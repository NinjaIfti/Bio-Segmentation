"""Step 3 — model, loss, metrics."""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(arch: str, encoder: str, encoder_weights: str | None) -> nn.Module:
    return smp.create_model(arch, encoder_name=encoder, encoder_weights=encoder_weights,
                            in_channels=3, classes=1)


class DiceBCELoss(nn.Module):
    """BCE gives stable per-pixel gradients; Dice directly optimizes overlap and handles
    the foreground/background imbalance (polyps are often <15% of the frame)."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.w = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w * self.bce(logits, target) + (1 - self.w) * self.dice(logits, target)


@torch.no_grad()
def per_image_scores(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> dict[str, torch.Tensor]:
    """pred/target: (N,1,H,W) binary {0,1}. Returns per-image Dice, IoU, precision, recall.

    Per-image averaging (mDice/mIoU) is the convention in the polyp segmentation literature.
    An image where both prediction and ground truth are empty counts as a perfect score.
    """
    pred = pred.flatten(1).float()
    target = target.flatten(1).float()
    tp = (pred * target).sum(1)
    fp = (pred * (1 - target)).sum(1)
    fn = ((1 - pred) * target).sum(1)
    both_empty = (tp + fp + fn) == 0

    dice = torch.where(both_empty, 1.0, 2 * tp / (2 * tp + fp + fn + eps))
    iou = torch.where(both_empty, 1.0, tp / (tp + fp + fn + eps))
    precision = torch.where(both_empty, 1.0, tp / (tp + fp + eps))
    recall = torch.where(both_empty, 1.0, tp / (tp + fn + eps))
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}
