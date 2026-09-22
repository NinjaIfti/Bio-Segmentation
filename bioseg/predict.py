"""Step 6 — inference on new images (any folder of .jpg/.png)."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bioseg.data import read_image
from bioseg.evaluate import load_model, predict_prob
from bioseg.utils import get_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kvasir_unet_r34")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--output", default="predictions")
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_model(Path(args.run) / "best.pt", device)
    src = Path(args.input)
    paths = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for p in paths:
        img = read_image(p)
        mask = (predict_prob(model, img, cfg["data"]["img_size"], device, cfg["eval"]["tta"])
                > cfg["eval"]["threshold"]).astype(np.uint8)
        cv2.imwrite(str(out / f"{p.stem}_mask.png"), mask * 255)
        tinted = img.copy()
        tinted[mask == 1] = (0.55 * tinted[mask == 1] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR))
    print(f"wrote {len(paths)} masks + overlays to {out}")


if __name__ == "__main__":
    main()
