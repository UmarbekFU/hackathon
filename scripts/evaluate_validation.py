#!/usr/bin/env python3
"""Evaluate validation performance for the current checkpoint pair."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.classification_dataset import ClassificationDataset, gather_classification_samples
from data.segmentation_dataset import SegmentationDataset, build_seg_val_transforms, list_segmentation_pairs
from models.classification.classify import (
    TTA_MODE_CHOICES as CLS_TTA_MODE_CHOICES,
    build_val_transform,
    load_model as load_cls_model,
    tta_classification_logits,
)
from models.segmentation.segment import (
    POSTPROCESS_CHOICES,
    TTA_MODE_CHOICES as SEG_TTA_MODE_CHOICES,
    load_model as load_seg_model,
    prob_to_mask,
    tta_segmentation_probs,
)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classification_val_samples(seed: int) -> list[tuple[Path, int]]:
    all_samples = gather_classification_samples(config.CLASSIFICATION_TRAIN)
    labels = [label for _, label in all_samples]
    _, val_idx = train_test_split(
        range(len(all_samples)),
        test_size=config.CLS_VAL_FRACTION,
        stratify=labels,
        random_state=seed,
    )
    return [all_samples[i] for i in val_idx]


def evaluate_classification(
    checkpoint: Path,
    tta_mode: str,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    model, img_size = load_cls_model(checkpoint, device)
    val_samples = classification_val_samples(seed)
    loader = DataLoader(
        ClassificationDataset(val_samples, build_val_transform(img_size, tta_mode=tta_mode)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    correct = 0
    total = 0
    per_class = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = tta_classification_logits(model, batch_x, tta_mode)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
            for y_true, y_pred in zip(batch_y.tolist(), pred.tolist()):
                per_class[y_true][1] += 1
                per_class[y_true][0] += int(y_true == y_pred)

    return {
        "accuracy": correct / max(total, 1),
        "samples": total,
        "per_class_accuracy": {
            str(label): per_class[label][0] / max(per_class[label][1], 1) for label in sorted(per_class)
        },
        "tta_mode": tta_mode,
    }


def evaluate_segmentation(
    checkpoint: Path,
    tta_mode: str,
    threshold: float,
    postprocess: str,
    batch_size: int,
    device: torch.device,
) -> dict:
    model, img_size = load_seg_model(checkpoint, device)
    pairs = list_segmentation_pairs(config.SEGMENTATION_VAL_IMAGES, config.SEGMENTATION_VAL_MASKS)
    loader = DataLoader(
        SegmentationDataset(pairs, build_seg_val_transforms(img_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    inter = 0.0
    union = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            probs = tta_segmentation_probs(model, batch_x.to(device), tta_mode).cpu().numpy()
            target = (batch_y[:, 0].cpu().numpy() > 0.5).astype("uint8")
            for idx in range(len(probs)):
                mask = prob_to_mask(probs[idx, 0], threshold=threshold, postprocess=postprocess)
                pred = (mask > 0).astype("uint8")
                y_true = target[idx]
                inter += float((pred & y_true).sum())
                union += float((pred | y_true).sum())

    return {
        "mIoU": inter / max(union, 1.0),
        "samples": len(pairs),
        "tta_mode": tta_mode,
        "threshold": threshold,
        "postprocess": postprocess,
    }


def score_summary(cls_result: dict, seg_result: dict) -> dict:
    score = 30.0 * cls_result["accuracy"] + 40.0 * seg_result["mIoU"]
    return {
        "classification_accuracy": cls_result["accuracy"],
        "segmentation_mIoU": seg_result["mIoU"],
        "proxy_score_70": score,
    }


def print_tta_sweep(checkpoint: Path, batch_size: int, seed: int, device: torch.device) -> None:
    sweep = {}
    for tta_mode in CLS_TTA_MODE_CHOICES:
        sweep[tta_mode] = evaluate_classification(checkpoint, tta_mode, batch_size, seed, device)["accuracy"]
    print("Classification TTA sweep:")
    for mode, score in sweep.items():
        print(f"  {mode:8s} {score:.4f}")


def print_seg_postprocess_sweep(
    checkpoint: Path,
    tta_mode: str,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> None:
    print("Segmentation postprocess sweep:")
    scores = {}
    for mode in POSTPROCESS_CHOICES:
        scores[mode] = evaluate_segmentation(
            checkpoint,
            tta_mode=tta_mode,
            threshold=threshold,
            postprocess=mode,
            batch_size=batch_size,
            device=device,
        )["mIoU"]
        print(f"  {mode:8s} {scores[mode]:.4f}")
    baseline = scores["none"]
    best_mode = max(scores, key=scores.get)
    delta = scores[best_mode] - baseline
    verdict = "ACCEPT" if delta >= 0.003 else "KEEP none"
    print(f"  best={best_mode} delta={delta:.4f} verdict={verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation performance for current checkpoints")
    parser.add_argument("--cls-checkpoint", type=Path, default=config.CLS_CHECKPOINT)
    parser.add_argument("--seg-checkpoint", type=Path, default=config.SEG_CHECKPOINT)
    parser.add_argument("--cls-tta", choices=CLS_TTA_MODE_CHOICES, default="none")
    parser.add_argument("--seg-tta", choices=SEG_TTA_MODE_CHOICES, default="none")
    parser.add_argument("--seg-threshold", type=float, default=0.5)
    parser.add_argument("--seg-postprocess", choices=POSTPROCESS_CHOICES, default="none")
    parser.add_argument("--cls-batch-size", type=int, default=32)
    parser.add_argument("--seg-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--sweep-cls-tta", action="store_true")
    parser.add_argument("--sweep-seg-postprocess", action="store_true")
    args = parser.parse_args()

    device = choose_device()
    cls_result = evaluate_classification(
        args.cls_checkpoint,
        tta_mode=args.cls_tta,
        batch_size=args.cls_batch_size,
        seed=args.seed,
        device=device,
    )
    seg_result = evaluate_segmentation(
        args.seg_checkpoint,
        tta_mode=args.seg_tta,
        threshold=args.seg_threshold,
        postprocess=args.seg_postprocess,
        batch_size=args.seg_batch_size,
        device=device,
    )
    summary = score_summary(cls_result, seg_result)

    print("Validation summary:")
    print(json.dumps(summary, indent=2))
    print("Classification details:")
    print(json.dumps(cls_result, indent=2))
    print("Segmentation details:")
    print(json.dumps(seg_result, indent=2))

    if args.sweep_cls_tta:
        print_tta_sweep(args.cls_checkpoint, args.cls_batch_size, args.seed, device)
    if args.sweep_seg_postprocess:
        print_seg_postprocess_sweep(
            args.seg_checkpoint,
            tta_mode=args.seg_tta,
            threshold=args.seg_threshold,
            batch_size=args.seg_batch_size,
            device=device,
        )


if __name__ == "__main__":
    main()
