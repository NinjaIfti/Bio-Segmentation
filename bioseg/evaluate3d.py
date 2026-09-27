"""Step 5 (3D) — per-class quantitative and qualitative analysis on the test set.

Quantitative: Dice, IoU, HD95, ASSD and NSD for every foreground class, per case and
aggregated. Boundary metrics use the volume's real voxel spacing, so they are in mm.

Qualitative: per-class contour overlays for the best and worst cases, and per-class
metric distributions, so the numbers in the table can be checked against the images.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader

from bioseg.data3d import build_dataset, eval_transforms, undo_center_pad
from bioseg.metrics3d import ALL_METRICS, BOUNDARY_METRICS, format_table, per_class_scores, summarize
from bioseg.msd import class_names
from bioseg.train3d import build_unet, read_split
from bioseg.utils import get_device, load_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CLASS_COLORS = ["#e6194b", "#4363d8", "#3cb44b", "#f58231", "#911eb4"]


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    names = ckpt["class_names"]
    model = build_unet(cfg["model"], len(names) + 1)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, names


@torch.no_grad()
def predict_case(model, image: torch.Tensor, device: torch.device, original_shape) -> torch.Tensor:
    """Padded (1,H,W,D) image -> integer label volume cropped back to the original shape."""
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(image.unsqueeze(0).to(device))
    pred = logits.float().argmax(1)[0].cpu()
    return undo_center_pad(pred, original_shape)


def best_slice(label: np.ndarray, num_fg_classes: int) -> tuple[int, int]:
    """Axis and index of the most informative slice.

    Anterior and Posterior sit at opposite ends of the structure's long axis, so a slice
    taken across that axis shows only one of them. Prefer an axis whose best slice shows
    every class, and among those take the slice with the most foreground.
    """
    best = None
    for axis in range(label.ndim):
        others = tuple(a for a in range(label.ndim) if a != axis)
        for idx in range(label.shape[axis]):
            sl = label.take(idx, axis=axis)
            n_classes = sum(int((sl == c + 1).any()) for c in range(num_fg_classes))
            area = int((sl > 0).sum())
            if best is None or (n_classes, area) > best[0]:
                best = ((n_classes, area), axis, idx)
    _, axis, idx = best
    return axis, idx


def plot_cases(rows: list[dict], root: Path, preds: dict, names: list[str], out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.5, 3.8 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        case = row["case"]
        img = nib.load(root / "imagesTr" / f"{case}.nii.gz").get_fdata()
        gt = nib.load(root / "labelsTr" / f"{case}.nii.gz").get_fdata().astype(int)
        pred = preds[case]
        axis, idx = best_slice(gt, len(names))
        sl = [slice(None)] * 3
        sl[axis] = idx
        img2d, gt2d, pred2d = img[tuple(sl)].T, gt[tuple(sl)].T, pred[tuple(sl)].T

        for c, (mask, label) in enumerate(((gt2d, "Ground truth"), (pred2d, "Prediction"))):
            ax = axes[r][c]
            ax.imshow(img2d, cmap="gray", origin="lower")
            for k, name in enumerate(names):
                m = (mask == k + 1).astype(float)
                if m.any():
                    ax.contour(m, levels=[0.5], colors=CLASS_COLORS[k], linewidths=1.6)
            ax.set_title(f"{case} — {label}" if c == 0 else
                         f"{label}  (Dice " + ", ".join(f"{n}:{row[f'dice_{n}']:.2f}" for n in names) + ")",
                         fontsize=9)
            ax.axis("off")
    handles = [Line2D([0], [0], color=CLASS_COLORS[k], lw=2, label=n) for k, n in enumerate(names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(names), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, names: list[str], out_png: Path) -> None:
    metrics = [("dice", "Dice"), ("iou", "IoU"), ("hd95_mm", "HD95 (mm)"),
               ("assd_mm", "ASSD (mm)"), ("nsd", "NSD")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 4))
    for ax, (key, label) in zip(axes, metrics):
        data = [df[f"{key}_{n}"].replace([np.inf, -np.inf], np.nan).dropna().values for n in names]
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True, widths=0.55)
        for patch, color in zip(bp["boxes"], CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-class metric distribution over the test set")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_curves(log_csv: Path, names: list[str], out_png: Path) -> None:
    log = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(log.epoch, log.train_loss, label="train")
    axes[0].plot(log.epoch, log.val_loss, label="val")
    axes[0].set(title="Loss (Dice + cross-entropy)", xlabel="epoch")
    for k, n in enumerate(names):
        axes[1].plot(log.epoch, log[f"val_dice_{n}"], label=n, color=CLASS_COLORS[k])
    axes[1].plot(log.epoch, log.val_dice, label="mean", color="black", ls="--")
    axes[1].set(title="Validation Dice per class", xlabel="epoch", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hippocampus_unet3d")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    run = Path(args.run)
    device = get_device()
    model, cfg, names = load_model(run / "best.pt", device)
    dcfg = cfg["data"]
    root = Path(dcfg["root"])
    num_classes = len(names) + 1
    ids = read_split(dcfg["splits_dir"], args.split)

    ds = build_dataset(root, ids, eval_transforms(dcfg["patch_size"]))
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    rows, score_rows, preds = [], [], {}
    for batch, case in zip(loader, ids):
        nii = nib.load(root / "labelsTr" / f"{case}.nii.gz")
        gt = torch.from_numpy(np.asanyarray(nii.dataobj).astype(np.int64))
        spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])

        # Self-check: undoing the padding on the transformed label must reproduce the
        # label on disk exactly. If it does not, predictions are misaligned with the
        # reference and every metric below would be quietly wrong.
        label_roundtrip = undo_center_pad(batch["label"][0, 0].long(), tuple(gt.shape))
        if not torch.equal(label_roundtrip, gt):
            raise RuntimeError(f"{case}: padding round-trip does not match the label on disk")

        pred = predict_case(model, batch["image"][0], device, tuple(gt.shape))
        preds[case] = pred.numpy()
        s = per_class_scores(pred, gt, num_classes, spacing, cfg["eval"]["nsd_tolerance_mm"])
        score_rows.append(s)
        rows.append({"case": case} | {f"{m}_{n}": s[m][k] for m in ALL_METRICS for k, n in enumerate(names)})

    df = pd.DataFrame(rows)
    df["dice_mean"] = df[[f"dice_{n}" for n in names]].mean(axis=1)
    df = df.sort_values("dice_mean")

    summary = summarize(score_rows, names)
    summary["split"] = args.split
    summary["nsd_tolerance_mm"] = cfg["eval"]["nsd_tolerance_mm"]
    summary["spacing_note"] = "HD95 and ASSD in mm, from each volume's own voxel spacing"

    eval_dir = run / f"eval_{args.split}"
    eval_dir.mkdir(exist_ok=True)
    df.to_csv(eval_dir / "per_case.csv", index=False)
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (eval_dir / "metrics_table.md").write_text(format_table(summary) + "\n")

    plot_curves(run / "log.csv", names, run / "curves.png")
    plot_distributions(df, names, eval_dir / "per_class_distributions.png")
    recs = df.to_dict("records")
    plot_cases(recs[:3], root, preds, names, eval_dir / "worst.png", f"{args.split}: 3 worst cases")
    plot_cases(recs[-3:][::-1], root, preds, names, eval_dir / "best.png", f"{args.split}: 3 best cases")

    print(format_table(summary))
    for metric in BOUNDARY_METRICS:
        for n in names:
            e = summary["per_class"][n]
            if e[f"{metric}_n_valid"] < e["n_cases"]:
                print(f"note: {metric} for {n} averaged over {e[f'{metric}_n_valid']}/{e['n_cases']} cases "
                      "(undefined where the class was absent from the prediction)")
    print(f"\noutputs in {eval_dir}")


if __name__ == "__main__":
    main()
