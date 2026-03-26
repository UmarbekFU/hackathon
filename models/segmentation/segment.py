#!/usr/bin/env python3
"""
Inference-only segmentation: images directory + checkpoint -> binary PNG masks.
Output masks match input filename and spatial size (organizer requirement).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

POSTPROCESS_CHOICES = ("none", "close3", "open3", "close5", "open5")
TTA_MODE_CHOICES = ("none", "h", "full")


def create_seg_model(arch: str, encoder: str, in_ch: int, classes: int) -> torch.nn.Module:
    a = (arch or "unet").lower().strip()
    if a == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=in_ch,
            classes=classes,
        )
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_ch,
        classes=classes,
    )


def load_model(checkpoint_path: Path, device: torch.device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    encoder_name = ckpt["encoder_name"]
    in_ch = int(ckpt["in_channels"])
    classes = int(ckpt["classes"])
    img_size = int(ckpt.get("img_size", 320))
    arch = ckpt.get("arch", "unet")
    model = create_seg_model(arch, encoder_name, in_ch, classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, img_size


def preprocess(rgb: np.ndarray, img_size: int) -> torch.Tensor:
    """rgb uint8 HxWx3 -> 1x3xSxS normalized tensor."""
    r = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    r = (r - mean) / std
    t = torch.from_numpy(np.transpose(r, (2, 0, 1))).unsqueeze(0)
    return t


@torch.no_grad()
def _forward_prob(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model(x))


@torch.no_grad()
def tta_segmentation_probs(model: torch.nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """mode: none | h | full — average aligned probability maps."""
    if mode == "none":
        return _forward_prob(model, x)
    probs = [_forward_prob(model, x)]
    xh = torch.flip(x, dims=[3])
    probs.append(torch.flip(_forward_prob(model, xh), dims=[3]))
    if mode == "full":
        xv = torch.flip(x, dims=[2])
        probs.append(torch.flip(_forward_prob(model, xv), dims=[2]))
        xhv = torch.flip(x, dims=[2, 3])
        probs.append(torch.flip(_forward_prob(model, xhv), dims=[2, 3]))
    return torch.stack(probs, dim=0).mean(dim=0)


def apply_binary_postprocess(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mask
    if mode not in POSTPROCESS_CHOICES:
        raise ValueError(f"Unsupported postprocess mode: {mode}")
    kernel_size = 3 if mode.endswith("3") else 5
    op = cv2.MORPH_CLOSE if mode.startswith("close") else cv2.MORPH_OPEN
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask, op, kernel)


def prob_to_mask(prob: np.ndarray, threshold: float = 0.5, postprocess: str = "none") -> np.ndarray:
    mask = ((prob > threshold).astype(np.uint8)) * 255
    return apply_binary_postprocess(mask, postprocess)


@torch.no_grad()
def run_segmentation(
    images_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    tta_mode: str = "none",
    threshold: float = 0.5,
    postprocess: str = "none",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, img_size = load_model(checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    paths = sorted([p for p in images_dir.iterdir() if p.suffix in exts])
    if not paths:
        raise FileNotFoundError(f"No images in {images_dir}")

    for p in tqdm(paths, desc="Segment"):
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = preprocess(rgb, img_size).to(device)
        prob = tta_segmentation_probs(model, x, tta_mode)[0, 0].cpu().numpy()
        prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = prob_to_mask(prob_full, threshold=threshold, postprocess=postprocess)
        out_path = output_dir / f"{p.stem}.png"
        cv2.imwrite(str(out_path), mask)


def main():
    parser = argparse.ArgumentParser(description="Segmentation inference for hackathon verification")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for predicted masks")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to segmentation .pth checkpoint")
    parser.add_argument(
        "--tta_mode",
        choices=TTA_MODE_CHOICES,
        default=None,
        help="Explicit TTA mode override",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="2-view TTA: average with horizontal flip (aligned)",
    )
    parser.add_argument(
        "--tta_full",
        action="store_true",
        help="4-view TTA: identity + H + V + HV flips (slower, often best mIoU)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary mask generation",
    )
    parser.add_argument(
        "--postprocess",
        choices=POSTPROCESS_CHOICES,
        default="none",
        help="Optional binary morphology applied after thresholding",
    )
    args = parser.parse_args()
    if args.tta_mode is not None:
        tta_mode = args.tta_mode
    else:
        tta_mode = "full" if args.tta_full else ("h" if args.tta else "none")
    run_segmentation(
        args.images_dir,
        args.output_dir,
        args.checkpoint,
        tta_mode=tta_mode,
        threshold=args.threshold,
        postprocess=args.postprocess,
    )


if __name__ == "__main__":
    main()
