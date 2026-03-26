from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def gather_classification_samples(train_root: Path) -> List[Tuple[Path, int]]:
    """Collect (image_path, label) for subfolders 0..11."""
    samples: List[Tuple[Path, int]] = []
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
            for p in class_dir.glob(ext):
                samples.append((p, label))
    samples.sort(key=lambda x: (x[1], str(x[0])))
    return samples


class ClassificationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Callable | None = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
