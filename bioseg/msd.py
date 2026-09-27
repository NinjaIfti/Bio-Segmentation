"""Step 1 (3D) — download a Medical Segmentation Decathlon task and write case-level splits.

MSD (Antonelli et al., Nature Communications 2022): http://medicaldecathlon.com/
Task04 Hippocampus: 260 labelled MRI volumes, 1 mm isotropic, classes
0 = background, 1 = Anterior, 2 = Posterior.
"""
import argparse
import json
import random
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from bioseg.utils import load_config


def case_ids(root: Path) -> list[str]:
    """Labelled cases, skipping the ._* resource-fork entries the MSD tarballs carry."""
    return sorted(p.name[: -len(".nii.gz")] for p in (root / "imagesTr").glob("*.nii.gz")
                  if not p.name.startswith("._"))


def class_names(root: Path) -> list[str]:
    """Foreground class names in label order, e.g. ['Anterior', 'Posterior']."""
    labels = json.loads((root / "dataset.json").read_text())["labels"]
    return [labels[k] for k in sorted(labels, key=int) if int(k) != 0]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hippocampus_unet3d.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)["data"]

    root = Path(cfg["root"])
    if not (root / "imagesTr").exists():
        tar_path = root.parent / f"{root.name}.tar"
        if not tar_path.exists():
            download(cfg["task_url"], tar_path)
        with tarfile.open(tar_path) as t:
            t.extractall(root.parent, filter="data")

    ids = case_ids(root)
    missing = [i for i in ids if not (root / "labelsTr" / f"{i}.nii.gz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} images have no label, e.g. {missing[:3]}")
    print(f"{root.name}: {len(ids)} labelled cases, classes {class_names(root)}")

    random.Random(cfg["split_seed"]).shuffle(ids)
    n_test = round(len(ids) * cfg["test_frac"])
    n_val = round(len(ids) * cfg["val_frac"])
    splits = {"test": ids[:n_test], "val": ids[n_test:n_test + n_val], "train": ids[n_test + n_val:]}

    splits_dir = Path(cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (splits_dir / f"{name}.txt").write_text("\n".join(sorted(items)) + "\n", newline="\n")
        print(f"{name:5s}: {len(items)} cases")


if __name__ == "__main__":
    main()
