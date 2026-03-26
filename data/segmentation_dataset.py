from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def list_segmentation_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair image and mask by same stem; only include if both exist."""
    pairs: List[Tuple[Path, Path]] = []
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix not in exts:
            continue
        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.is_file():
            mask_alt = masks_dir / img_path.name
            if mask_alt.is_file():
                mask_path = mask_alt
            else:
                continue
        pairs.append((img_path, mask_path))
    return pairs


def _mask_to_binary01(mask: np.ndarray) -> np.ndarray:
    """H,W -> float32 {0,1}."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = (m > 127).astype(np.float32)
    else:
        m = (m > 0.5).astype(np.float32)
    return m


def build_seg_train_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop((image_size, image_size), scale=(0.75, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5
            ),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.3),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.15),
            A.GridDistortion(
                num_steps=4,
                distort_limit=0.04,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.2,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_seg_val_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        transform: Optional[Callable] = None,
    ):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(img_path)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(mask_path)
        mask_bin = _mask_to_binary01(mask_gray)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask_bin)
            image_t = out["image"]
            mask_t = out["mask"]
        else:
            image_t = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
            mask_t = torch.from_numpy(mask_bin).unsqueeze(0).float()

        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        mask_t = mask_t.float()
        return image_t, mask_t
