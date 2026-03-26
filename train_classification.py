#!/usr/bin/env python3
"""Train 12-class biopsy classifier; validation split from train only."""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandAugment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.classification_dataset import ClassificationDataset, gather_classification_samples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(
    img_size: int,
    train: bool,
    randaug_n: int,
    randaug_m: int,
    vflip_prob: float,
):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if train:
        ops = [transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0))]
        if randaug_n > 0:
            ops.append(
                RandAugment(
                    num_ops=randaug_n,
                    magnitude=randaug_m,
                    interpolation=InterpolationMode.BILINEAR,
                )
            )
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=vflip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return transforms.Compose(ops)
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def compute_class_weights(samples, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in samples:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    w = counts.sum() / (num_classes * counts.astype(np.float32))
    return torch.tensor(w, dtype=torch.float32)


def build_weighted_sampler(samples, num_classes: int) -> WeightedRandomSampler:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in samples:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    sample_weights = [1.0 / counts[y] for _, y in samples]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


@torch.no_grad()
def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k in msd:
        esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def classifier_module(model: nn.Module) -> nn.Module:
    head = model.get_classifier()
    if isinstance(head, nn.Module):
        return head
    raise TypeError(f"Unsupported classifier head for {type(model).__name__}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(trainable)
    if not trainable:
        for p in classifier_module(model).parameters():
            p.requires_grad_(True)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--model-name", type=str, default=config.CLS_MODEL_NAME)
    parser.add_argument("--img-size", type=int, default=config.CLS_IMG_SIZE)
    parser.add_argument("--label-smoothing", type=float, default=config.CLS_LABEL_SMOOTHING)
    parser.add_argument("--randaug-n", type=int, default=config.CLS_RANDAUG_N)
    parser.add_argument("--randaug-m", type=int, default=config.CLS_RANDAUG_M)
    parser.add_argument("--vflip-prob", type=float, default=0.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=config.CLS_EARLY_STOP_PATIENCE)
    parser.add_argument("--checkpoint-out", type=Path, default=config.CLS_CHECKPOINT)
    parser.add_argument("--weighted-sampler", action="store_true", help="Use inverse-frequency sampling")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA (use raw weights for val/save)")
    parser.add_argument("--no-mixup", action="store_true", help="Disable Mixup")
    args = parser.parse_args()
    batch_size = args.batch_size or config.CLS_BATCH_SIZE
    use_ema = config.CLS_USE_EMA and not args.no_ema
    use_mixup = config.CLS_MIXUP_PROB > 0 and not args.no_mixup

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    all_samples = gather_classification_samples(config.CLASSIFICATION_TRAIN)
    labels = [y for _, y in all_samples]
    train_idx, val_idx = train_test_split(
        range(len(all_samples)),
        test_size=config.CLS_VAL_FRACTION,
        stratify=labels,
        random_state=args.seed,
    )
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    train_ds = ClassificationDataset(
        train_samples,
        build_transforms(
            args.img_size,
            train=True,
            randaug_n=args.randaug_n,
            randaug_m=args.randaug_m,
            vflip_prob=args.vflip_prob,
        ),
    )
    val_ds = ClassificationDataset(
        val_samples,
        build_transforms(
            args.img_size,
            train=False,
            randaug_n=args.randaug_n,
            randaug_m=args.randaug_m,
            vflip_prob=args.vflip_prob,
        ),
    )

    sampler = build_weighted_sampler(train_samples, config.CLS_NUM_CLASSES) if args.weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.CLS_NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.CLS_NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=config.CLS_NUM_CLASSES,
    ).to(device)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, trainable=False)

    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    weights = compute_class_weights(train_samples, config.CLS_NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")
    frozen_phase_epochs = min(args.freeze_backbone_epochs, args.epochs)
    scheduler = build_scheduler(
        optimizer,
        frozen_phase_epochs if frozen_phase_epochs > 0 else args.epochs,
    )

    best_acc = 0.0
    epochs_no_improve = 0
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    eval_model = lambda: ema_model if use_ema else model

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(model, args.lr, args.weight_decay)
            scheduler = build_scheduler(optimizer, args.epochs - args.freeze_backbone_epochs)
            print("Unfroze backbone for full fine-tuning.")

        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            do_mixup = use_mixup and x.size(0) > 1 and random.random() < config.CLS_MIXUP_PROB
            with autocast(enabled=device.type == "cuda"):
                if do_mixup:
                    lam = float(np.random.beta(config.CLS_MIXUP_ALPHA, config.CLS_MIXUP_ALPHA))
                    idx = torch.randperm(x.size(0), device=device)
                    x_m = lam * x + (1.0 - lam) * x[idx]
                    ya, yb = y, y[idx]
                    logits = model(x_m)
                    loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if use_ema:
                update_ema(model, ema_model, config.CLS_EMA_DECAY)

            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{running / max(n, 1):.4f}")

        scheduler.step()
        m = eval_model()
        m.eval()
        val_acc = evaluate(m, val_loader, device)
        print(f"Epoch {epoch} val_acc={val_acc:.4f}{' (EMA)' if use_ema else ''}")
        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            epochs_no_improve = 0
            save_sd = ema_model.state_dict() if use_ema else model.state_dict()
            torch.save(
                {
                    "model_name": args.model_name,
                    "num_classes": config.CLS_NUM_CLASSES,
                    "img_size": args.img_size,
                    "state_dict": save_sd,
                    "val_acc": val_acc,
                    "ema": use_ema,
                },
                args.checkpoint_out,
            )
            print(f"Saved best checkpoint to {args.checkpoint_out}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping (no val improvement for {args.early_stop_patience} epochs).")
                break

    print(f"Done. Best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
