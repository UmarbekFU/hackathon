#!/usr/bin/env python3
"""
Inference-only classification: test image directory + checkpoint -> Excel.
Organizer verification script (no training).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

TTA_MODE_CHOICES = ("none", "h", "full", "tencrop")


class ImageFolderPredict(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), p.stem


def load_model(checkpoint_path: Path, device: torch.device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt["model_name"]
    num_classes = int(ckpt["num_classes"])
    img_size = int(ckpt.get("img_size", 300))
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, img_size


def build_val_transform(img_size: int, tta_mode: str = "none"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    if tta_mode == "tencrop":
        return transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.1)),
                transforms.TenCrop(img_size),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            normalize,
        ]
    )


def collect_image_paths(test_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    paths = sorted([p for p in test_dir.iterdir() if p.suffix in exts])
    return paths


@torch.no_grad()
def tta_classification_logits(model: torch.nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    if x.ndim == 5:
        batch_size, n_views, channels, height, width = x.shape
        logits = model(x.reshape(batch_size * n_views, channels, height, width))
        return logits.reshape(batch_size, n_views, -1).mean(dim=1)
    if mode == "none":
        return model(x)
    logits = [model(x)]
    logits.append(model(torch.flip(x, dims=[3])))
    if mode == "full":
        logits.append(model(torch.flip(x, dims=[2])))
        logits.append(model(torch.flip(x, dims=[2, 3])))
    return torch.stack(logits, dim=0).mean(dim=0)


@torch.no_grad()
def run_classification(
    test_dir: Path,
    checkpoint: Path,
    output_xlsx: Path,
    batch_size: int = 32,
    tta_mode: str = "none",
    num_workers: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, img_size = load_model(checkpoint, device)
    tfm = build_val_transform(img_size, tta_mode=tta_mode)
    paths = collect_image_paths(test_dir)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images in {test_dir}")

    ds = ImageFolderPredict(paths, tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: list[tuple[str, int]] = []
    for batch_x, stems in tqdm(loader, desc="Classify"):
        batch_x = batch_x.to(device)
        logits = tta_classification_logits(model, batch_x, tta_mode)
        pred = logits.argmax(dim=1).cpu().tolist()
        for s, y in zip(stems, pred):
            rows.append((s, int(y)))

    df = pd.DataFrame(rows, columns=["Image_ID", "Label"])
    df = df.sort_values("Image_ID").reset_index(drop=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_xlsx, index=False)


def main():
    parser = argparse.ArgumentParser(description="Classification inference for hackathon verification")
    parser.add_argument("--test_dir", type=Path, required=True, help="Directory with test PNG/JPG images")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to classification .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output .xlsx path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for inference")
    parser.add_argument(
        "--tta_mode",
        choices=TTA_MODE_CHOICES,
        default=None,
        help="Explicit TTA mode override",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="2-view TTA: average logits with horizontal flip",
    )
    parser.add_argument(
        "--tta_full",
        action="store_true",
        help="4-view TTA: identity + H + V + HV (slower, can help accuracy)",
    )
    parser.add_argument(
        "--tta_tencrop",
        action="store_true",
        help="10-crop TTA using torchvision.TenCrop",
    )
    args = parser.parse_args()
    if args.tta_mode is not None:
        tta_mode = args.tta_mode
    elif args.tta_tencrop:
        tta_mode = "tencrop"
    else:
        tta_mode = "full" if args.tta_full else ("h" if args.tta else "none")
    run_classification(
        args.test_dir,
        args.checkpoint,
        args.output,
        args.batch_size,
        tta_mode=tta_mode,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
