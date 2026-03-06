"""
02_preprocess_data.py  —  Dataset Preprocessing (Revised Methodology)
=======================================================================
BRIGHT correct statistics (Zenodo v1.0):
  • 14 disaster events total
  • 11 unrestricted events → 3,395 full pre+post+target triplets
  • 3  restricted events (no pre-event optical) → 851 annotation pairs
  • 4,246 total annotation pairs across all 14 events

Two-study separation:
  • Study 1 (DisasterGAN quality): xBD ONLY — optical→optical domain
  • Study 2 (BRIGHT segmentation): BRIGHT ONLY — no xBD mixing

Income classification by EVENT (not by country substring):
  LIC: haiti-earthquake, congo-volcano, myanmar-hurricane*
  MIC: turkey-earthquake, morocco-earthquake, libya-flood,
       bata-explosion, beirut-explosion, mexico-hurricane*, ukraine-conflict*
  HIC: hawaii-wildfire, la_palma-volcano, noto-earthquake, marshall-wildfire
  (* restricted — no pre-event optical)

Usage:
    python 02_preprocess_data.py [--bright-only | --xbd-only] [--verify]
"""

import os, sys, json, shutil, argparse, random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"

XBD_RAW        = DATA_DIR / "xbd_raw"
BRIGHT_RAW     = DATA_DIR / "bright_raw"
XBD_PROCESSED  = DATA_DIR / "xbd"
BRIGHT_DIR     = DATA_DIR / "bright"

SEED = 42
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# BRIGHT: event → income level  (14 events from Zenodo)
# ─────────────────────────────────────────────────────────────────────────────
EVENT_INCOME = {
    # LIC
    "haiti-earthquake":   "lic",
    "congo-volcano":      "lic",
    # MIC
    "turkey-earthquake":  "mic",
    "morocco-earthquake": "mic",
    "libya-flood":        "mic",
    "bata-explosion":     "mic",
    "beirut-explosion":   "mic",
    # HIC
    "hawaii-wildfire":    "hic",
    "la_palma-volcano":   "hic",
    "noto-earthquake":    "hic",
    "marshall-wildfire":  "hic",
    # Restricted (no pre-event optical → excluded from Study 2 triplets)
    "mexico-hurricane":   "mic",
    "myanmar-hurricane":  "lic",
    "ukraine-conflict":   "mic",
}

FULL_TRIPLET_EVENTS = {
    "haiti-earthquake", "congo-volcano",
    "turkey-earthquake", "morocco-earthquake", "libya-flood",
    "bata-explosion", "beirut-explosion",
    "hawaii-wildfire", "la_palma-volcano", "noto-earthquake", "marshall-wildfire",
}

RESTRICTED_EVENTS = {"mexico-hurricane", "myanmar-hurricane", "ukraine-conflict"}

# xBD: event-level split used for DisasterGAN training ONLY (Study 1)
XBD_TRAIN_EVENTS = {
    "guatemala-volcano", "hurricane-florence", "hurricane-harvey",
    "hurricane-matthew", "hurricane-michael", "joplin-tornado",
    "lower-puna-volcano", "mexico-earthquake", "midwest-flooding",
    "moore-tornado", "nepal-flooding", "nepal-earthquake",
    "palu-tsunami", "portugal-wildfire", "santa-rosa-wildfire",
    "socal-fire", "tuscaloosa-tornado", "woolsey-fire",
}
XBD_TEST_EVENTS = {
    "noto-earthquake", "sunda-strait", "marshall-wildfire",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _event_from_stem(stem: str) -> str:
    """
    Detect BRIGHT event name from a filename stem.
    Filenames follow: <event-name>_<tile-id>  e.g. haiti-earthquake_00001
    """
    for event in sorted(EVENT_INCOME.keys(), key=len, reverse=True):
        if stem.startswith(event) or event.replace("-", "_") in stem:
            return event
    # fallback: first two underscore-joined tokens
    parts = stem.replace("-", "_").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def _income(event: str) -> str:
    return EVENT_INCOME.get(event, "unknown")


def _save_manifest(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"  Manifest → {path}  ({len(records)} records)")

# ─────────────────────────────────────────────────────────────────────────────
# BRIGHT preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def build_bright_triplets() -> list:
    """
    Scan BRIGHT raw directories and build a list of (pre, post, target) triplets.
    Only full-triplet events (11 unrestricted) are included.
    Returns list of dicts with keys: event, income, pre_img, post_img, mask
    """
    # Locate pre / post / target directories
    pre_dir = post_dir = tgt_dir = None
    for d in BRIGHT_RAW.rglob("*"):
        if not d.is_dir():
            continue
        n = d.name.lower()
        if "pre" in n and pre_dir is None:
            pre_dir = d
        elif "post" in n and post_dir is None:
            post_dir = d
        elif ("target" in n or "label" in n) and tgt_dir is None:
            tgt_dir = d

    if not all([pre_dir, post_dir, tgt_dir]):
        print(f"  [warn] Could not locate all BRIGHT sub-dirs in {BRIGHT_RAW}")
        print(f"    pre={pre_dir}  post={post_dir}  target={tgt_dir}")
        return []

    # Index post and target by stem
    EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    post_idx = {f.stem: f for f in post_dir.iterdir() if f.suffix.lower() in EXTS}
    tgt_idx  = {f.stem: f for f in tgt_dir.iterdir()  if f.suffix.lower() in EXTS}

    triplets = []
    skipped  = 0

    for pre_file in sorted(pre_dir.iterdir()):
        if pre_file.suffix.lower() not in EXTS:
            continue
        stem  = pre_file.stem
        event = _event_from_stem(stem)

        if event not in FULL_TRIPLET_EVENTS:
            skipped += 1
            continue

        post_file = post_idx.get(stem)
        tgt_file  = tgt_idx.get(stem)

        if post_file is None or tgt_file is None:
            skipped += 1
            continue

        triplets.append({
            "event":    event,
            "income":   _income(event),
            "pre_img":  str(pre_file),
            "post_img": str(post_file),
            "mask":     str(tgt_file),
        })

    print(f"  Built {len(triplets)} full triplets ({skipped} skipped/restricted)")
    return triplets


def split_bright_by_income(triplets: list) -> dict:
    """
    Split triplets by income level.  Split is BY EVENT (not random tile split)
    so that no event appears in both train and val of the same stratum.
    Returns dict: income → {"train": [...], "val": [...]}
    """
    by_income: dict = defaultdict(list)
    for t in triplets:
        by_income[t["income"]].append(t)

    splits = {}
    for income, records in by_income.items():
        # Group by event, then put last event as val (ensures no data leakage)
        by_event: dict = defaultdict(list)
        for r in records:
            by_event[r["event"]].append(r)

        events = sorted(by_event.keys())
        val_events = events[-1:]          # last event alphabetically = val
        train_events = events[:-1]

        train = [r for e in train_events for r in by_event[e]]
        val   = [r for e in val_events   for r in by_event[e]]

        splits[income] = {"train": train, "val": val}
        print(f"  {income.upper()}: {len(train)} train tiles "
              f"({len(train_events)} events), {len(val)} val tiles "
              f"(event: {val_events})")

    return splits


def compute_bright_statistics(triplets: list):
    """Print corrected BRIGHT dataset statistics for the paper."""
    print("\n" + "=" * 60)
    print("BRIGHT Dataset Statistics (Zenodo v1.0)")
    print("=" * 60)
    by_income  = defaultdict(int)
    by_event   = defaultdict(int)
    for t in triplets:
        by_income[t["income"]] += 1
        by_event[t["event"]]   += 1

    print(f"Total full triplets (11 unrestricted events): {len(triplets)}")
    print(f"  Target: ~3,395 triplets")
    print(f"Restricted events (no pre-event optical):     {len(RESTRICTED_EVENTS)}")
    print(f"  → ~851 annotation pairs (post+target only)")
    print(f"Total annotation pairs across 14 events:      ~4,246\n")

    print("Per income level:")
    for inc in ["lic", "mic", "hic"]:
        print(f"  {inc.upper()}: {by_income[inc]} tiles")

    print("\nPer event:")
    for event, count in sorted(by_event.items()):
        print(f"  {event:<30s}  {count:>5d}  [{EVENT_INCOME[event].upper()}]")
    print("=" * 60)


def preprocess_bright():
    """Main BRIGHT preprocessing: build manifests per income stratum."""
    print("\n[BRIGHT] Preprocessing …")

    if not BRIGHT_RAW.exists():
        print(f"  [error] BRIGHT raw data not found at {BRIGHT_RAW}")
        print("  Run 01_download_datasets.sh first.")
        return False

    triplets = build_bright_triplets()
    if not triplets:
        print("  [error] No triplets found — check BRIGHT_RAW directory structure.")
        return False

    compute_bright_statistics(triplets)
    splits = split_bright_by_income(triplets)

    # Save per-income manifests
    for income, split_data in splits.items():
        out_dir = BRIGHT_DIR / income
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_manifest(split_data["train"], out_dir / "manifest_train.json")
        _save_manifest(split_data["val"],   out_dir / "manifest_val.json")
        # Combined manifest (used by 15b visualisation)
        _save_manifest(split_data["train"] + split_data["val"],
                       out_dir / "manifest.json")

    return True

# ─────────────────────────────────────────────────────────────────────────────
# xBD preprocessing  (Study 1 ONLY — DisasterGAN, optical domain)
# ─────────────────────────────────────────────────────────────────────────────
def _event_from_xbd_stem(stem: str) -> str:
    """Extract event name from xBD filename: disaster-location_NNNNNNNN_pre_disaster"""
    return "_".join(stem.split("_")[:-2]) if "_" in stem else stem


def preprocess_xbd():
    """
    Organize xBD into event-level train/test split for DisasterGAN.
    Study 1 only — xBD is NEVER mixed into BRIGHT segmentation (Study 2).
    """
    print("\n[xBD] Preprocessing for DisasterGAN (Study 1) …")

    if not XBD_RAW.exists():
        print(f"  [error] xBD raw data not found at {XBD_RAW}")
        return False

    EXTS = {".png", ".jpg", ".jpeg"}
    splits_out = {"train": XBD_PROCESSED / "train",
                  "test":  XBD_PROCESSED / "test"}
    for d in splits_out.values():
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)

    # Find all images recursively
    all_images = [f for f in XBD_RAW.rglob("*_pre_disaster.*")
                  if f.suffix.lower() in EXTS]
    print(f"  Found {len(all_images)} pre-disaster images in xBD")

    train_manifest, test_manifest = [], []

    for pre_path in tqdm(all_images, desc="  Organising xBD"):
        stem  = pre_path.stem.replace("_pre_disaster", "")
        event = _event_from_xbd_stem(stem)
        split = "test" if event in XBD_TEST_EVENTS else "train"
        out   = splits_out[split]

        post_path = pre_path.parent / f"{stem}_post_disaster{pre_path.suffix}"
        mask_path = pre_path.parent.parent / "labels" / f"{stem}_post_disaster.json"

        try:
            shutil.copy2(pre_path,  out / "images" / pre_path.name)
            if post_path.exists():
                shutil.copy2(post_path, out / "images" / post_path.name)
            if mask_path.exists():
                shutil.copy2(mask_path, out / "labels" / mask_path.name)
        except Exception as exc:
            print(f"  [warn] {pre_path.name}: {exc}")
            continue

        record = {
            "event":    event,
            "pre_img":  str(out / "images" / pre_path.name),
            "post_img": str(out / "images" / post_path.name) if post_path.exists() else "",
            "label":    str(out / "labels" / mask_path.name) if mask_path.exists() else "",
        }
        if split == "train":
            train_manifest.append(record)
        else:
            test_manifest.append(record)

    _save_manifest(train_manifest, XBD_PROCESSED / "train_manifest.json")
    _save_manifest(test_manifest,  XBD_PROCESSED / "test_manifest.json")

    print(f"  xBD train: {len(train_manifest)} tiles  "
          f"(events: {len(XBD_TRAIN_EVENTS)})")
    print(f"  xBD test:  {len(test_manifest)} tiles  "
          f"(events: {len(XBD_TEST_EVENTS)})")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
def verify():
    ok = True
    print("\n[Verify] Checking manifest files …")
    for income in ["lic", "mic", "hic"]:
        for split in ["train", "val"]:
            p = BRIGHT_DIR / income / f"manifest_{split}.json"
            if p.exists():
                with open(p) as fh:
                    n = len(json.load(fh))
                print(f"  BRIGHT/{income}/manifest_{split}.json  →  {n} records ✓")
            else:
                print(f"  [MISSING] {p}")
                ok = False

    for split in ["train", "test"]:
        p = XBD_PROCESSED / f"{split}_manifest.json"
        if p.exists():
            with open(p) as fh:
                n = len(json.load(fh))
            print(f"  xBD/{split}_manifest.json  →  {n} records ✓")
        # xBD manifests are optional (only needed for Study 1)
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess xBD and BRIGHT datasets")
    parser.add_argument("--bright-only", action="store_true")
    parser.add_argument("--xbd-only",    action="store_true")
    parser.add_argument("--verify",      action="store_true")
    args = parser.parse_args()

    do_bright = not args.xbd_only
    do_xbd    = not args.bright_only

    print("=" * 60)
    print("02  —  Dataset Preprocessing (Revised Methodology)")
    print("=" * 60)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"Process BRIGHT: {do_bright}")
    print(f"Process xBD  : {do_xbd} (Study 1 / DisasterGAN only)")

    if do_bright:
        preprocess_bright()
    if do_xbd:
        preprocess_xbd()
    if args.verify or True:   # always verify
        verify()

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
