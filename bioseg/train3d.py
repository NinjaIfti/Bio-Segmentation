"""Step 4 (3D) — train a 3D U-Net for multi-class segmentation.

Validation tracks mean foreground Dice (background excluded), which is what the
checkpoint is selected on.
"""
import argparse
import csv
import math
import shutil
import time
from pathlib import Path

import torch
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from tqdm import tqdm

from bioseg.data3d import build_dataset, eval_transforms, train_transforms
from bioseg.metrics3d import one_hot
from bioseg.msd import class_names
from bioseg.utils import get_device, load_config, seed_everything
from monai.metrics import compute_dice


def read_split(splits_dir: str | Path, name: str) -> list[str]:
    return [line for line in Path(splits_dir, f"{name}.txt").read_text().split() if line]


def build_unet(mcfg: dict, num_classes: int) -> UNet:
    return UNet(spatial_dims=3, in_channels=1, out_channels=num_classes,
                channels=tuple(mcfg["channels"]), strides=tuple(mcfg["strides"]),
                num_res_units=mcfg["num_res_units"], dropout=mcfg["dropout"])


def warmup_cosine(total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


@torch.no_grad()
def validate(model, loader, loss_fn, device, num_classes: int, amp: bool) -> dict:
    model.eval()
    total_loss, n, dice_per_class = 0.0, 0, []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        logits = logits.float()
        total_loss += loss_fn(logits, y).item() * len(x)
        n += len(x)
        pred = logits.argmax(1)
        for i in range(len(x)):
            p = one_hot(pred[i], num_classes)
            g = one_hot(y[i, 0], num_classes)
            dice_per_class.append(compute_dice(p, g, include_background=False).squeeze(0).cpu())
    dice = torch.stack(dice_per_class).nanmean(0)
    return {"val_loss": total_loss / n, "val_dice": dice.nanmean().item(),
            "val_dice_per_class": dice.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hippocampus_unet3d.yaml")
    ap.add_argument("--epochs", type=int, help="override config (e.g. 2 for a smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    if args.epochs:
        tcfg["epochs"] = args.epochs
    seed_everything(tcfg["seed"])
    device = get_device()
    amp = tcfg["amp"] and device.type == "cuda"
    out = Path(tcfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")

    names = class_names(Path(dcfg["root"]))
    num_classes = len(names) + 1
    patch = dcfg["patch_size"]
    train_ds = build_dataset(dcfg["root"], read_split(dcfg["splits_dir"], "train"), train_transforms(patch))
    val_ds = build_dataset(dcfg["root"], read_split(dcfg["splits_dir"], "val"), eval_transforms(patch))
    # persistent workers matter here: the volumes are tiny, so without this the per-epoch
    # worker startup costs more than the epoch itself
    dl_kw = dict(num_workers=tcfg["num_workers"], pin_memory=True,
                 persistent_workers=tcfg["num_workers"] > 0)
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=tcfg["batch_size"], **dl_kw)
    print(f"device={device} classes={names} train={len(train_ds)} val={len(val_ds)}")

    model = build_unet(mcfg, num_classes).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total_steps, warmup_steps=max(1, 3 * len(train_dl))))

    fields = ["epoch", "train_loss", "val_loss", "val_dice"] + [f"val_dice_{n}" for n in names] + ["lr", "time_s"]
    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_dice, bad_epochs = -1.0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0
        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{tcfg['epochs']}", leave=False):
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(x)
            seen += len(x)

        val = validate(model, val_dl, loss_fn, device, num_classes, amp)
        row = {"epoch": epoch, "train_loss": running / seen, "val_loss": val["val_loss"],
               "val_dice": val["val_dice"], "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
        row |= {f"val_dice_{n}": d for n, d in zip(names, val["val_dice_per_class"])}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch,
                "val_dice": row["val_dice"], "class_names": names}
        torch.save(ckpt, out / "last.pt")
        improved = row["val_dice"] > best_dice
        if improved:
            best_dice, bad_epochs = row["val_dice"], 0
            torch.save(ckpt, out / "best.pt")
        else:
            bad_epochs += 1
        per_cls = "  ".join(f"{n} {d:.4f}" for n, d in zip(names, val["val_dice_per_class"]))
        print(f"epoch {epoch:3d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} | "
              f"dice {row['val_dice']:.4f} ({per_cls}) | {row['time_s']:.0f}s" + ("  *best*" if improved else ""))
        if bad_epochs >= tcfg["early_stop_patience"]:
            print(f"early stop: no val improvement for {bad_epochs} epochs")
            break

    print(f"best mean foreground val dice {best_dice:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
