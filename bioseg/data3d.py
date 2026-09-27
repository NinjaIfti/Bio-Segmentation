"""Step 2 (3D) — MONAI dataset and transforms for MSD volumes.

Every volume is centre-padded to a fixed size so it can be batched. The Hippocampus
volumes are all smaller than the configured patch, so this pads only and never crops:
no voxel is thrown away, and evaluation can undo the padding exactly.
"""
from pathlib import Path

import torch
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
)

KEYS = ["image", "label"]


def case_files(root: str | Path, ids: list[str]) -> list[dict]:
    root = Path(root)
    return [{"image": str(root / "imagesTr" / f"{i}.nii.gz"),
             "label": str(root / "labelsTr" / f"{i}.nii.gz"),
             "id": i} for i in ids]


def base_transforms(patch: list[int]) -> list:
    return [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        EnsureTyped(keys=KEYS),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=patch),
    ]


def train_transforms(patch: list[int]) -> Compose:
    return Compose(base_transforms(patch) + [
        RandFlipd(keys=KEYS, spatial_axis=0, prob=0.5),
        RandFlipd(keys=KEYS, spatial_axis=1, prob=0.5),
        RandFlipd(keys=KEYS, spatial_axis=2, prob=0.5),
        RandAffined(keys=KEYS, prob=0.3, rotate_range=(0.15, 0.15, 0.15),
                    scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
    ])


def eval_transforms(patch: list[int]) -> Compose:
    return Compose(base_transforms(patch))


def build_dataset(root, ids, transform, cache_rate: float = 1.0) -> CacheDataset:
    # The whole task is ~30 MB, so caching every decoded volume in RAM is free speed.
    return CacheDataset(case_files(root, ids), transform=transform, cache_rate=cache_rate, progress=False)


def undo_center_pad(volume: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
    """Inverse of ResizeWithPadOrCropd's symmetric padding, for a (H,W,D) volume.

    MONAI pads symmetrically with the extra voxel going at the end, so the original data
    starts at floor((padded - original) / 2) along each axis.
    """
    slices = []
    for padded, orig in zip(volume.shape, original_shape):
        start = (padded - orig) // 2
        if start < 0:
            raise ValueError(f"volume {volume.shape} is smaller than the original {original_shape}; "
                             "increase data.patch_size so the transform only pads")
        slices.append(slice(start, start + orig))
    return volume[tuple(slices)]
