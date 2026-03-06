"""
_model_registry.py  —  Central model loader for Study 2
========================================================
Usage:
    from _model_registry import load_model
    model = load_model("unet")        # or "segformer" / "changeformer"

All models share:
    • 4-class output  (no damage / minor / major / destroyed)
    • 4-channel input for U-Net & SegFormer  [R, G, B, SAR]
    • Siamese 3-channel input for ChangeFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 4

# ══════════════════════════════════════════════════════════════
# U-Net
# ══════════════════════════════════════════════════════════════
class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class _UNet(nn.Module):
    """
    Standard U-Net with 4-channel input.
    Input : (B, 4, H, W)  — pre-optical RGB + post-SAR
    Output: (B, N_CLASSES, H, W)
    """
    def __init__(self, in_channels: int = 4, n_classes: int = N_CLASSES,
                 features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(_ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        self.bottleneck = _ConvBlock(ch, ch * 2)

        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(_ConvBlock(f * 2, f))
            ch = f

        self.out_conv = nn.Conv2d(ch, n_classes, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                  align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.out_conv(x)


# ══════════════════════════════════════════════════════════════
# SegFormer-B2  (HuggingFace transformers)
# ══════════════════════════════════════════════════════════════
class _SegFormerWrapper(nn.Module):
    """
    Wraps HuggingFace SegFormerForSemanticSegmentation.
    Extends the patch-embedding from 3→4 channels by appending the mean
    of the RGB weights as the SAR-channel weight (warm-start).
    Output is upsampled from 1/4 resolution to full resolution.
    """
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError:
            raise ImportError("pip install transformers")

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )

        # Extend first conv from 3ch → 4ch
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(4, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding, bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3]  = old_conv.weight.mean(dim=1)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, x):
        out = self.model(pixel_values=x)
        logits = out.logits   # (B, n_classes, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


# ══════════════════════════════════════════════════════════════
# ChangeFormer  (Siamese Transformer for change detection)
# Bandara & Patel, IGARSS 2022
# ══════════════════════════════════════════════════════════════
class _MixedTransformerBlock(nn.Module):
    """Efficient self-attention with spatial reduction (MiT style)."""
    def __init__(self, dim, num_heads=1, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm_sr = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.sr_ratio = sr_ratio

    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.sr is not None:
            x2d = x.permute(0,2,1).reshape(B, C, H, W)
            x2d = self.sr(x2d).reshape(B, C, -1).permute(0,2,1)
            x2d = self.norm_sr(x2d)
            attn_out, _ = self.attn(x, x2d, x2d)
        else:
            attn_out, _ = self.attn(x, x, x)
        x = shortcut + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class _HierarchicalEncoder(nn.Module):
    """4-stage patch-embedding + transformer encoder."""
    def __init__(self, in_ch=3, embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        self.stages = nn.ModuleList()
        self.patch_embeds = nn.ModuleList()
        self.norms = nn.ModuleList()

        ch = in_ch
        for i, (dim, nh, sr) in enumerate(zip(embed_dims, num_heads, sr_ratios)):
            stride  = 4 if i == 0 else 2
            padding = 1
            self.patch_embeds.append(
                nn.Conv2d(ch, dim, kernel_size=3, stride=stride, padding=padding))
            self.stages.append(_MixedTransformerBlock(dim, nh, sr))
            self.norms.append(nn.LayerNorm(dim))
            ch = dim

        self.embed_dims = embed_dims

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for pe, stage, norm in zip(self.patch_embeds, self.stages, self.norms):
            x = pe(x)
            B, C, H, W = x.shape
            x_seq = x.flatten(2).permute(0, 2, 1)   # B N C
            x_seq = stage(x_seq, H, W)
            x_seq = norm(x_seq)
            x = x_seq.permute(0, 2, 1).reshape(B, C, H, W)
            outs.append(x)
        return outs


class _ChangeFormer(nn.Module):
    """
    Siamese Transformer for bi-temporal change detection.
    Input : (B, 3, H, W) × 2  —  pre-event and post-event (SAR replicated to 3ch)
    Output: (B, N_CLASSES, H, W)

    Note: replicating SAR to 3 channels is a known simplification;
    documented as a limitation in the paper.
    """
    def __init__(self, n_classes: int = N_CLASSES,
                 embed_dims=(32, 64, 160, 256)):
        super().__init__()
        self.encoder_pre  = _HierarchicalEncoder(3, embed_dims)
        self.encoder_post = _HierarchicalEncoder(3, embed_dims)

        # Difference-feature MLP decoder
        total_ch = sum(d * 2 for d in embed_dims)
        self.decoder = nn.Sequential(
            nn.Conv2d(total_ch, 256, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, n_classes, 1),
        )

    def forward(self, pre, post):
        feats_pre  = self.encoder_pre(pre)
        feats_post = self.encoder_post(post)

        target_size = feats_pre[0].shape[2:]
        diff_feats  = []
        for fp, fpo in zip(feats_pre, feats_post):
            diff = torch.abs(fp - fpo)
            diff = F.interpolate(diff, size=target_size, mode="bilinear",
                                 align_corners=False)
            diff_feats.append(diff)
            fp2  = F.interpolate(fp,  size=target_size, mode="bilinear",
                                 align_corners=False)
            diff_feats.append(fp2)

        x = torch.cat(diff_feats, dim=1)
        x = self.decoder(x)
        return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
MODEL_NAMES = ["unet", "segformer", "changeformer"]

def load_model(name: str) -> nn.Module:
    """
    Return an untrained model instance.

    Args:
        name: one of "unet", "segformer", "changeformer"
    """
    name = name.lower()
    if name == "unet":
        return _UNet(in_channels=4, n_classes=N_CLASSES)
    elif name == "segformer":
        return _SegFormerWrapper(n_classes=N_CLASSES)
    elif name == "changeformer":
        return _ChangeFormer(n_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose from {MODEL_NAMES}")
