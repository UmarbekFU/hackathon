#!/usr/bin/env python3
"""Quick checklist: Excel rows, mask count, binary values, sample size match."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Submission team folder (default: submission/<TEAM_NAME> from config)",
    )
    args = parser.parse_args()
    team = config.TEAM_NAME
    base = args.root or (ROOT / "submission" / team)
    xlsx = base / f"{team} test_ground_truth.xlsx"
    masks_dir = base / team
    data_test = config.SEGMENTATION_TEST_IMAGES

    errors = []
    if not xlsx.is_file():
        errors.append(f"Missing Excel: {xlsx}")
    if not masks_dir.is_dir():
        errors.append(f"Missing mask folder: {masks_dir}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(xlsx)
    if list(df.columns) != ["Image_ID", "Label"]:
        errors.append(f"Bad columns: {list(df.columns)} (expected Image_ID, Label)")
    n_rows = len(df)
    if n_rows != 1276:
        errors.append(f"Excel rows: {n_rows} (expected 1276)")
    if df["Label"].min() < 0 or df["Label"].max() > 11:
        errors.append("Label out of range 0..11")

    mask_paths = sorted(masks_dir.glob("*.png"))
    if len(mask_paths) != 200:
        errors.append(f"Mask PNG count: {len(mask_paths)} (expected 200)")

    stems_data = {p.stem for p in data_test.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    stems_mask = {p.stem for p in mask_paths}
    missing = stems_data - stems_mask
    extra = stems_mask - stems_data
    if missing:
        errors.append(f"Missing masks for stems (sample): {list(sorted(missing))[:5]}")
    if extra:
        errors.append(f"Extra mask stems (sample): {list(sorted(extra))[:5]}")

    for mp in mask_paths[: min(5, len(mask_paths))]:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            errors.append(f"Unreadable mask: {mp}")
            continue
        vals = set(m.flatten().tolist())
        if not vals.issubset({0, 255}):
            errors.append(f"{mp.name} not binary {{0,255}}: unique sample {sorted(vals)[:8]}")
        stem = mp.stem
        candidates = list(data_test.glob(stem + ".*"))
        img_p = next((c for c in candidates if c.suffix.lower() in {".png", ".jpg", ".jpeg"}), None)
        if img_p is None:
            continue
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        if im.shape[0] != m.shape[0] or im.shape[1] != m.shape[1]:
            errors.append(f"{mp.name} size {m.shape} != image {im.shape[:2]}")

    if errors:
        print("FAILED:")
        for e in errors:
            print(" ", e)
        sys.exit(1)

    print(f"OK — team {team!r}")
    print(f"  {xlsx.name}: {n_rows} rows, labels in 0..11")
    print(f"  {masks_dir.relative_to(base)}: {len(mask_paths)} binary PNGs, sample sizes match inputs")


if __name__ == "__main__":
    main()
