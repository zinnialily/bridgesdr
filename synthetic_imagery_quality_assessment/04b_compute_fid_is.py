"""
04b_compute_fid_is.py  —  Study 1: FID & Inception Score for DisasterGAN
=========================================================================
Addresses Reviewer 1 Concern 5: "No standard GAN quality metrics (FID, IS)."

Computes:
  • FID  (Fréchet Inception Distance) — lower is better
  • IS   (Inception Score)            — higher is better

between REAL xBD held-out test images and SYNTHETIC DisasterGAN outputs.
Evaluation is performed in the OPTICAL domain only (xBD → xBD), consistent
with Study 1's clean two-study framework.

Results are reported:
  • Overall
  • Per income level (LIC / MIC / HIC using xBD event → income mapping)

Dependencies:
    pip install pytorch-fid

Outputs:
    results/study1/fid_is/fid_is_results.json
"""

import sys, json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

REAL_DIR  = PROJECT_ROOT / "data"    / "xbd" / "test" / "images"
SYNTH_DIR = PROJECT_ROOT / "results" / "study1" / "synthetic_images"
OUT_DIR   = PROJECT_ROOT / "results" / "study1" / "fid_is"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# xBD test event → broad income proxy
XBD_EVENT_INCOME = {
    "noto-earthquake":   "hic",
    "sunda-strait":      "mic",
    "marshall-wildfire": "hic",
    # add more if your test split differs
}

# ─────────────────────────────────────────────
# InceptionV3 feature extractor
# ─────────────────────────────────────────────
def _get_inception():
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        raise ImportError(
            "pytorch-fid is required: pip install pytorch-fid"
        )
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def _images_to_tensor(paths: list, device) -> torch.Tensor:
    """Load a list of image paths → N×3×299×299 float tensor in [0,1]."""
    tensors = []
    for p in tqdm(paths, desc="  Loading images", leave=False):
        img = Image.open(p).convert("RGB").resize((299, 299), Image.BILINEAR)
        t   = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        tensors.append(t.permute(2, 0, 1))
    return torch.stack(tensors).to(device)


def _get_activations(paths: list, model, batch_size: int = 32) -> np.ndarray:
    device = next(model.parameters()).device
    all_acts = []
    for i in range(0, len(paths), batch_size):
        batch = _images_to_tensor(paths[i : i + batch_size], device)
        with torch.no_grad():
            act = model(batch)[0].squeeze(-1).squeeze(-1)
        all_acts.append(act.cpu().numpy())
    return np.concatenate(all_acts, axis=0)


def _compute_fid(mu1, sigma1, mu2, sigma2) -> float:
    """Numpy FID computation."""
    from scipy.linalg import sqrtm
    diff   = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def _compute_is(paths: list, model, batch_size: int = 32,
                splits: int = 10) -> tuple:
    """Compute Inception Score (mean, std) over `splits` splits."""
    device = next(model.parameters()).device

    # We need softmax probabilities, not pool3 features.
    # Re-use InceptionV3 output[0] which is pool3; recompute with logits.
    # Use torchvision's inception instead for IS.
    import torchvision.models as tvm
    inc_v3 = tvm.inception_v3(pretrained=True, transform_input=False).eval().to(device)

    all_probs = []
    for i in range(0, len(paths), batch_size):
        batch = _images_to_tensor(paths[i : i + batch_size], device)
        with torch.no_grad():
            logits = inc_v3(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)   # N × 1000

    split_scores = []
    n = len(all_probs)
    for k in range(splits):
        part = all_probs[k * n // splits : (k + 1) * n // splits]
        p_y  = part.mean(axis=0, keepdims=True)
        kl   = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        split_scores.append(np.exp(kl.sum(axis=1).mean()))

    return float(np.mean(split_scores)), float(np.std(split_scores))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("04b  —  FID & IS Computation  (Study 1, xBD test set)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _get_inception()

    # Gather post-disaster image paths
    EXTS = {".png", ".jpg", ".jpeg"}

    real_paths  = sorted(p for p in REAL_DIR.glob("*_post_disaster.*")
                         if p.suffix.lower() in EXTS)
    synth_paths = sorted(p for p in SYNTH_DIR.rglob("*.png"))

    if not real_paths:
        print(f"  [error] No real test images found in {REAL_DIR}")
        sys.exit(1)
    if not synth_paths:
        print(f"  [error] No synthetic images found in {SYNTH_DIR}")
        sys.exit(1)

    print(f"  Real images    : {len(real_paths)}")
    print(f"  Synthetic images: {len(synth_paths)}")

    # ── Overall FID ──────────────────────────────────────────────────────
    print("\nComputing activations (real) …")
    acts_real  = _get_activations(real_paths,  model)
    print("Computing activations (synthetic) …")
    acts_synth = _get_activations(synth_paths, model)

    mu_r, sig_r = acts_real.mean(0),  np.cov(acts_real,  rowvar=False)
    mu_s, sig_s = acts_synth.mean(0), np.cov(acts_synth, rowvar=False)

    fid_overall = _compute_fid(mu_r, sig_r, mu_s, sig_s)
    print(f"\n  Overall FID : {fid_overall:.2f}")

    print("\nComputing IS (synthetic) …")
    is_mean, is_std = _compute_is(synth_paths, model)
    print(f"  IS : {is_mean:.2f} ± {is_std:.2f}")

    results = {
        "overall": {
            "FID": fid_overall,
            "IS_mean": is_mean,
            "IS_std":  is_std,
            "n_real":  len(real_paths),
            "n_synth": len(synth_paths),
        },
        "per_income": {}
    }

    # ── Per-income FID ───────────────────────────────────────────────────
    # Partition real images by event → income
    income_real: dict = {"lic": [], "mic": [], "hic": []}
    for p in real_paths:
        stem  = p.stem.replace("_post_disaster", "")
        event = "_".join(stem.split("_")[:-1])
        inc   = XBD_EVENT_INCOME.get(event)
        if inc:
            income_real[inc].append(p)

    for inc, rpaths in income_real.items():
        if len(rpaths) < 10:
            print(f"  [skip] {inc.upper()} has only {len(rpaths)} real images — skipping FID")
            continue
        # synthetic: use same income subfolder if present, else all
        sinc_dir = SYNTH_DIR / inc
        spaths   = sorted(sinc_dir.glob("*.png")) if sinc_dir.exists() else synth_paths
        if len(spaths) < 10:
            spaths = synth_paths

        acts_r = _get_activations(rpaths, model)
        acts_s = _get_activations(spaths, model)
        mu_r2, sig_r2 = acts_r.mean(0), np.cov(acts_r, rowvar=False)
        mu_s2, sig_s2 = acts_s.mean(0), np.cov(acts_s, rowvar=False)
        fid_inc = _compute_fid(mu_r2, sig_r2, mu_s2, sig_s2)
        results["per_income"][inc] = {"FID": fid_inc, "n_real": len(rpaths)}
        print(f"  {inc.upper()} FID : {fid_inc:.2f}  (n_real={len(rpaths)})")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "fid_is_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
