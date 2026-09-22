"""Step 4 — training loop with validation, checkpointing and early stopping."""
import argparse
import csv
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bioseg.data import PolypDataset, eval_transforms, read_split, train_transforms
from bioseg.model import DiceBCELoss, build_model, per_image_scores
from bioseg.utils import get_device, load_config, seed_everything


def warmup_cosine(total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp: bool) -> dict[str, float]:
    model.eval()
    losses, scores = [], {"dice": [], "iou": []}
    for batch in loader:
        x, y = batch["image"].to(device), batch["mask"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        logits = logits.float()
        losses.append(loss_fn(logits, y).item() * len(x))
        s = per_image_scores((logits.sigmoid() > 0.5), y)
        for k in scores:
            scores[k].append(s[k])
    n = len(loader.dataset)
    return {
        "val_loss": sum(losses) / n,
        "val_dice": torch.cat(scores["dice"]).mean().item(),
        "val_iou": torch.cat(scores["iou"]).mean().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kvasir_unet.yaml")
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

    train_ds = PolypDataset(dcfg["root"], read_split(dcfg["splits_dir"], "train"), train_transforms(dcfg["img_size"]))
    val_ds = PolypDataset(dcfg["root"], read_split(dcfg["splits_dir"], "val"), eval_transforms(dcfg["img_size"]))
    loader_kw = dict(num_workers=tcfg["num_workers"], pin_memory=True, persistent_workers=tcfg["num_workers"] > 0)
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, **loader_kw)
    print(f"device={device} ({torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}) "
          f"train={len(train_ds)} val={len(val_ds)}")

    model = build_model(mcfg["arch"], mcfg["encoder"], mcfg["encoder_weights"]).to(device)
    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total_steps, warmup_steps=2 * len(train_dl)))

    log_path = out / "log.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "time_s"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_dice, bad_epochs = -1.0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{tcfg['epochs']}", leave=False):
            x, y = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(x)

        row = {"epoch": epoch, "train_loss": running / (len(train_dl) * tcfg["batch_size"])}
        row |= validate(model, val_dl, loss_fn, device, amp)
        row |= {"lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_dice": row["val_dice"]}
        torch.save(ckpt, out / "last.pt")
        improved = row["val_dice"] > best_dice
        if improved:
            best_dice, bad_epochs = row["val_dice"], 0
            torch.save(ckpt, out / "best.pt")
        else:
            bad_epochs += 1
        print(f"epoch {epoch:3d} | train_loss {row['train_loss']:.4f} | val_loss {row['val_loss']:.4f} | "
              f"val_dice {row['val_dice']:.4f} | val_iou {row['val_iou']:.4f} | {row['time_s']:.0f}s"
              + ("  *best*" if improved else ""))
        if bad_epochs >= tcfg["early_stop_patience"]:
            print(f"early stop: no val improvement for {bad_epochs} epochs")
            break

    print(f"best val dice {best_dice:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
