"""Per-class segmentation metrics for multi-class 3D volumes.

Overlap metrics (Dice, IoU) and boundary metrics (HD95, ASSD, NSD) answer different
questions, which is why both get reported: a prediction can have a high Dice and still
have a badly wrong boundary in a few slices, and that shows up in HD95.

All boundary metrics are computed in millimetres using the voxel spacing of the volume,
so they are comparable across cases with different resolutions.
"""
import numpy as np
import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
    compute_iou,
    compute_surface_dice,
)

# Boundary metrics are undefined when a structure is missing from the prediction or the
# reference, because one of the two surfaces does not exist. MONAI returns NaN there.
BOUNDARY_METRICS = ("hd95_mm", "assd_mm", "nsd")
OVERLAP_METRICS = ("dice", "iou")
ALL_METRICS = OVERLAP_METRICS + BOUNDARY_METRICS


def one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(H,W,D) integer labels -> (1,C,H,W,D) one-hot float, background as channel 0."""
    oh = torch.nn.functional.one_hot(label.long(), num_classes)      # (H,W,D,C)
    return oh.permute(3, 0, 1, 2).unsqueeze(0).float()


def per_class_scores(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    spacing: tuple[float, ...],
    nsd_tolerance_mm: list[float],
) -> dict[str, np.ndarray]:
    """Metrics for one case. pred/gt are integer label volumes of the same shape.

    Returns one array of length num_classes-1 per metric (foreground classes only —
    background Dice is ~0.99 by construction and carries no information).
    """
    p, g = one_hot(pred, num_classes), one_hot(gt, num_classes)
    kw = dict(include_background=False)
    sp = list(spacing)
    out = {
        "dice": compute_dice(p, g, **kw),
        "iou": compute_iou(p, g, **kw),
        "hd95_mm": compute_hausdorff_distance(p, g, percentile=95, spacing=sp, **kw),
        "assd_mm": compute_average_surface_distance(p, g, symmetric=True, spacing=sp, **kw),
        "nsd": compute_surface_dice(p, g, class_thresholds=list(nsd_tolerance_mm), spacing=sp, **kw),
    }
    return {k: v.squeeze(0).cpu().numpy() for k, v in out.items()}


def summarize(rows: list[dict], class_names: list[str]) -> dict:
    """Aggregate per-case rows into a per-class table.

    Boundary metrics are averaged with nanmean and the number of cases that actually
    contributed is reported, so a mean over 38 of 39 cases is never silently presented
    as a mean over all of them.
    """
    table = {}
    for ci, name in enumerate(class_names):
        entry = {}
        for metric in ALL_METRICS:
            vals = np.array([r[metric][ci] for r in rows], dtype=float)
            valid = np.isfinite(vals)
            entry[metric] = float(np.nanmean(vals[valid])) if valid.any() else float("nan")
            entry[f"{metric}_std"] = float(np.nanstd(vals[valid])) if valid.any() else float("nan")
            if metric in BOUNDARY_METRICS:
                entry[f"{metric}_n_valid"] = int(valid.sum())
        entry["n_cases"] = len(rows)
        table[name] = entry

    mean_row = {m: float(np.nanmean([table[c][m] for c in class_names])) for m in ALL_METRICS}
    return {"per_class": table, "mean_over_classes": mean_row}


def format_table(summary: dict) -> str:
    """Markdown table, ready to paste into a report."""
    head = f"| {'Class':<12} | Dice | IoU | HD95 (mm) | ASSD (mm) | NSD |\n|{'-' * 14}|------|-----|-----------|-----------|-----|"
    lines = [head]
    for name, e in summary["per_class"].items():
        lines.append(
            f"| {name:<12} | {e['dice']:.4f} | {e['iou']:.4f} | {e['hd95_mm']:.3f} | "
            f"{e['assd_mm']:.3f} | {e['nsd']:.4f} |"
        )
    m = summary["mean_over_classes"]
    lines.append(f"| **Mean**     | {m['dice']:.4f} | {m['iou']:.4f} | {m['hd95_mm']:.3f} | "
                 f"{m['assd_mm']:.3f} | {m['nsd']:.4f} |")
    return "\n".join(lines)
