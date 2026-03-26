#!/usr/bin/env python3
"""
Build a submission tree: TeamName/Excel, TeamName/TeamName/*.png masks, TeamName/models/...
Copies checkpoints next to classify.py / segment.py for upload.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def resolve_tta_mode(explicit: str | None, legacy_h: bool, legacy_full: bool, default: str) -> str:
    if explicit is not None:
        return explicit
    if legacy_full:
        return "full"
    if legacy_h:
        return "h"
    return default


def main():
    parser = argparse.ArgumentParser(description="Build hackathon submission folder for TEAM_NAME in config.py")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="2-view TTA (horizontal flip) for classify + segment",
    )
    parser.add_argument(
        "--tta-full",
        action="store_true",
        dest="tta_full",
        help="4-view TTA (H+V+HV); slower, usually strongest for segmentation",
    )
    parser.add_argument(
        "--cls-tta-mode",
        choices=("none", "h", "full", "tencrop"),
        default=None,
        help="Explicit classification TTA mode override",
    )
    parser.add_argument(
        "--seg-tta-mode",
        choices=("none", "h", "full"),
        default=None,
        help="Explicit segmentation TTA mode override",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Segmentation probability threshold for export",
    )
    parser.add_argument(
        "--postprocess",
        choices=("none", "close3", "open3", "close5", "open5"),
        default="none",
        help="Optional binary morphology applied to exported masks",
    )
    args = parser.parse_args()
    cls_tta_mode = resolve_tta_mode(args.cls_tta_mode, args.tta, args.tta_full, default="none")
    seg_tta_mode = resolve_tta_mode(args.seg_tta_mode, args.tta, args.tta_full, default="full")

    team = config.TEAM_NAME
    base = ROOT / "submission" / team
    inner_masks = base / team
    base.mkdir(parents=True, exist_ok=True)
    inner_masks.mkdir(parents=True, exist_ok=True)

    xlsx_name = f"{team} test_ground_truth.xlsx"
    xlsx_path = base / xlsx_name

    cls_script = ROOT / "models" / "classification" / "classify.py"
    seg_script = ROOT / "models" / "segmentation" / "segment.py"

    if not config.CLS_CHECKPOINT.is_file():
        print(f"Missing classification checkpoint: {config.CLS_CHECKPOINT}", file=sys.stderr)
        sys.exit(1)
    if not config.SEG_CHECKPOINT.is_file():
        print(f"Missing segmentation checkpoint: {config.SEG_CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    cls_cmd = [
        sys.executable,
        str(cls_script),
        "--test_dir",
        str(config.CLASSIFICATION_TEST),
        "--checkpoint",
        str(config.CLS_CHECKPOINT),
        "--output",
        str(xlsx_path),
    ]
    if cls_tta_mode != "none":
        cls_cmd.extend(["--tta_mode", cls_tta_mode])

    print("Running classification on test set...")
    subprocess.check_call(cls_cmd, cwd=str(ROOT))

    seg_cmd = [
        sys.executable,
        str(seg_script),
        "--images_dir",
        str(config.SEGMENTATION_TEST_IMAGES),
        "--output_dir",
        str(inner_masks),
        "--checkpoint",
        str(config.SEG_CHECKPOINT),
        "--threshold",
        str(args.threshold),
        "--postprocess",
        args.postprocess,
    ]
    if seg_tta_mode != "none":
        seg_cmd.extend(["--tta_mode", seg_tta_mode])

    print("Running segmentation on test set...")
    subprocess.check_call(seg_cmd, cwd=str(ROOT))

    # Package models/ for organizers
    dest_models = base / "models"
    if dest_models.exists():
        shutil.rmtree(dest_models)
    dest_models.mkdir(parents=True)

    for task in ("classification", "segmentation"):
        src = ROOT / "models" / task
        dst = dest_models / task
        dst.mkdir(parents=True)
        py_name = "classify.py" if task == "classification" else "segment.py"
        shutil.copy2(src / py_name, dst / py_name)
        shutil.copy2(src / "requirements.txt", dst / "requirements.txt")
        ckpt_src = config.CLS_CHECKPOINT if task == "classification" else config.SEG_CHECKPOINT
        shutil.copy2(ckpt_src, dst / ckpt_src.name)

    print(f"Submission folder ready: {base}")
    print(f"  Excel: {xlsx_path}")
    print(f"  Masks: {inner_masks} ({len(list(inner_masks.glob('*.png')))} png)")
    print(f"  Settings: cls_tta={cls_tta_mode}, seg_tta={seg_tta_mode}, threshold={args.threshold}, postprocess={args.postprocess}")


if __name__ == "__main__":
    main()
