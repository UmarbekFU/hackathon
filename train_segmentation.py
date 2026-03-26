#!/usr/bin/env python3
"""Train binary segmentation (U-Net or DeepLabV3+); train/val folders from dataset."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.segmentation_dataset import (
    SegmentationDataset,
    build_seg_train_transforms,
    build_seg_val_transforms,
    list_segmentation_pairs,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_seg_model(arch: str, encoder: str, pretrained: bool = True) -> nn.Module:
    enc_w = "imagenet" if pretrained else None
    a = arch.lower().strip()
    if a == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=enc_w,
            in_channels=3,
            classes=1,
        )
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=enc_w,
        in_channels=3,
        classes=1,
    )


@torch.no_grad()
def mean_iou_logits(model, loader, device) -> float:
    model.eval()
    inter = 0.0
    union = 0.0
    eps = 1e-6
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        yb = (y > 0.5).float()
        inter += (pred * yb).sum().item()
        union += (pred + yb - pred * yb).clamp(0, 1).sum().item()
    return inter / (union + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=("unet", "deeplabv3plus"),
        help="Override config.SEG_ARCH",
    )
    args = parser.parse_args()
    batch_size = args.batch_size or config.SEG_BATCH_SIZE
    arch = (args.arch or config.SEG_ARCH).lower()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_pairs = list_segmentation_pairs(config.SEGMENTATION_TRAIN_IMAGES, config.SEGMENTATION_TRAIN_MASKS)
    val_pairs = list_segmentation_pairs(config.SEGMENTATION_VAL_IMAGES, config.SEGMENTATION_VAL_MASKS)
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    print(f"Architecture: {arch}, encoder: {config.SEG_ENCODER}")

    train_tf = build_seg_train_transforms(config.SEG_IMG_SIZE)
    val_tf = build_seg_val_transforms(config.SEG_IMG_SIZE)
    train_ds = SegmentationDataset(train_pairs, train_tf)
    val_ds = SegmentationDataset(val_pairs, val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.SEG_NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.SEG_NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = create_seg_model(arch, config.SEG_ENCODER, pretrained=True)
    model = model.to(device)

    dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
    bce = nn.BCEWithLogitsLoss()
    focal = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE, alpha=0.25, gamma=2.0)
    tversky = smp.losses.TverskyLoss(
        mode=smp.losses.BINARY_MODE,
        from_logits=True,
        alpha=config.SEG_TVERSKY_ALPHA,
        beta=config.SEG_TVERSKY_BETA,
    )

    wf = config.SEG_LOSS_FOCAL_WEIGHT
    wt = config.SEG_LOSS_TVERSKY_WEIGHT

    def seg_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return dice(logits, y) + bce(logits, y) + wf * focal(logits, y) + wt * tversky(logits, y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou = 0.0
    epochs_no_improve = 0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = seg_loss(logits, y)
            scaler.scale(loss).backward()
            if config.SEG_GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.SEG_GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{running/n:.4f}")

        scheduler.step()
        val_iou = mean_iou_logits(model, val_loader, device)
        print(f"Epoch {epoch} val_mIoU={val_iou:.4f}")
        if val_iou > best_iou + 1e-6:
            best_iou = val_iou
            epochs_no_improve = 0
            torch.save(
                {
                    "arch": arch,
                    "encoder_name": config.SEG_ENCODER,
                    "in_channels": 3,
                    "classes": 1,
                    "img_size": config.SEG_IMG_SIZE,
                    "state_dict": model.state_dict(),
                    "val_mIoU": val_iou,
                },
                config.SEG_CHECKPOINT,
            )
            print(f"Saved best checkpoint to {config.SEG_CHECKPOINT}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.SEG_EARLY_STOP_PATIENCE:
                print(f"Early stopping (no val mIoU gain for {config.SEG_EARLY_STOP_PATIENCE} epochs).")
                break

    print(f"Done. Best val_mIoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
