"""Step 2 — dataset + augmentation."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), rotate=(-20, 20),
                 border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    # Kvasir masks are JPEGs, so edges carry compression noise — threshold at mid-grey
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def read_split(splits_dir: str | Path, name: str) -> list[str]:
    return [line for line in Path(splits_dir, f"{name}.txt").read_text().split() if line]


class PolypDataset(Dataset):
    def __init__(self, root: str | Path, ids: list[str], transform: A.Compose):
        self.root = Path(root)
        self.ids = ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        id_ = self.ids[i]
        img = read_image(self.root / "images" / f"{id_}.jpg")
        mask = read_mask(self.root / "masks" / f"{id_}.jpg")
        out = self.transform(image=img, mask=mask)
        return {
            "image": out["image"],
            "mask": out["mask"].unsqueeze(0).float(),
            "id": id_,
        }


def denormalize(t: torch.Tensor) -> np.ndarray:
    """CHW normalized tensor -> HWC uint8 RGB."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(3, 1, 1)
    img = (t * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)
