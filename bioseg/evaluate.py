"""Step 5 — test-set evaluation at original resolution, training curves, qualitative figures."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bioseg.data import eval_transforms, read_image, read_mask, read_split
from bioseg.model import build_model, per_image_scores
from bioseg.utils import get_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_model(ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt["config"]["model"]
    model = build_model(mcfg["arch"], mcfg["encoder"], encoder_weights=None)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["config"]


@torch.no_grad()
def predict_prob(model, img: np.ndarray, size: int, device: torch.device, tta: bool) -> np.ndarray:
    """RGB uint8 HxWx3 -> foreground probability HxW at the image's original resolution."""
    h, w = img.shape[:2]
    x = eval_transforms(size)(image=img)["image"].unsqueeze(0).to(device)
    views = [(), (3,), (2,)] if tta else [()]
    prob = torch.zeros(1, 1, size, size, device=device)
    for dims in views:
        xi = x.flip(dims) if dims else x
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = model(xi).float().sigmoid()
        prob += p.flip(dims) if dims else p
    prob /= len(views)
    return F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Green = ground truth contour, red = prediction contour."""
    out = img.copy()
    for m, color in ((gt, (0, 255, 0)), (pred, (255, 0, 0))):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness=max(2, img.shape[0] // 200))
    return out


def plot_curves(log_csv: Path, out_png: Path) -> None:
    log = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(log.epoch, log.train_loss, label="train")
    axes[0].plot(log.epoch, log.val_loss, label="val")
    axes[0].set(title="Loss (0.5·BCE + 0.5·Dice)", xlabel="epoch")
    axes[1].plot(log.epoch, log.val_dice, label="val Dice")
    axes[1].plot(log.epoch, log.val_iou, label="val IoU")
    axes[1].set(title="Validation overlap", xlabel="epoch", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_examples(rows: list[dict], root: Path, preds: dict[str, np.ndarray], out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
    for r, row in enumerate(rows):
        img = read_image(root / "images" / f"{row['id']}.jpg")
        gt = read_mask(root / "masks" / f"{row['id']}.jpg")
        pred = preds[row["id"]]
        for c, (im, t) in enumerate([(img, row["id"]), (gt, "ground truth"),
                                     (overlay(img, gt, pred), f"Dice {row['dice']:.3f}  (green=GT, red=pred)")]):
            axes[r, c].imshow(im, cmap="gray" if im.ndim == 2 else None)
            axes[r, c].set_title(t, fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kvasir_unet_r34")
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    device = get_device()
    model, cfg = load_model(run / "best.pt", device)
    dcfg, ecfg = cfg["data"], cfg["eval"]
    tta = ecfg["tta"] and not args.no_tta
    root = Path(dcfg["root"])
    ids = read_split(dcfg["splits_dir"], args.split)

    rows, preds = [], {}
    for id_ in ids:
        img = read_image(root / "images" / f"{id_}.jpg")
        gt = read_mask(root / "masks" / f"{id_}.jpg")
        pred = (predict_prob(model, img, dcfg["img_size"], device, tta) > ecfg["threshold"]).astype(np.uint8)
        preds[id_] = pred
        s = per_image_scores(torch.from_numpy(pred)[None, None], torch.from_numpy(gt)[None, None])
        rows.append({"id": id_, **{k: v.item() for k, v in s.items()}, "gt_area_frac": gt.mean()})

    df = pd.DataFrame(rows).sort_values("dice")
    eval_dir = run / f"eval_{args.split}{'' if tta else '_notta'}"
    eval_dir.mkdir(exist_ok=True)
    df.to_csv(eval_dir / "per_image.csv", index=False)

    metrics = ["dice", "iou", "precision", "recall"]
    summary = {"split": args.split, "n_images": len(df), "tta": tta, "threshold": ecfg["threshold"],
               **{f"m{k}": float(df[k].mean()) for k in metrics},
               **{f"std_{k}": float(df[k].std()) for k in metrics},
               "n_dice_below_0.5": int((df.dice < 0.5).sum())}
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_curves(run / "log.csv", run / "curves.png")
    recs = df.to_dict("records")
    plot_examples(recs[:4], root, preds, eval_dir / "worst.png", f"{args.split}: 4 worst cases")
    plot_examples(recs[-4:][::-1], root, preds, eval_dir / "best.png", f"{args.split}: 4 best cases")

    print(json.dumps(summary, indent=2))
    print(f"outputs in {eval_dir}")


if __name__ == "__main__":
    main()
