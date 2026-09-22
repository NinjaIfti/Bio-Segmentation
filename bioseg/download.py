"""Step 1 — download Kvasir-SEG and write a fixed train/val/test split.

Kvasir-SEG (Jha et al., MMM 2020): 1000 colonoscopy images with pixel-level polyp masks.
https://datasets.simula.no/kvasir-seg/
"""
import argparse
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from bioseg.utils import load_config

URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="kvasir-seg.zip") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def make_splits(root: Path, splits_dir: Path, val_frac: float, test_frac: float, seed: int) -> None:
    ids = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    missing = [i for i in ids if not (root / "masks" / f"{i}.jpg").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} images have no mask, e.g. {missing[:3]}")

    random.Random(seed).shuffle(ids)
    n_test = round(len(ids) * test_frac)
    n_val = round(len(ids) * val_frac)
    splits = {
        "test": ids[:n_test],
        "val": ids[n_test:n_test + n_val],
        "train": ids[n_test + n_val:],
    }
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (splits_dir / f"{name}.txt").write_text("\n".join(sorted(items)) + "\n", newline="\n")
        print(f"{name:5s}: {len(items)} images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kvasir_unet.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)["data"]

    root = Path(cfg["root"])
    if not (root / "images").exists():
        zip_path = root.parent / "kvasir-seg.zip"
        if not zip_path.exists():
            download(URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root.parent)
    print(f"dataset at {root}: {len(list((root / 'images').glob('*.jpg')))} images")

    make_splits(root, Path(cfg["splits_dir"]), cfg["val_frac"], cfg["test_frac"], cfg["split_seed"])


if __name__ == "__main__":
    main()
