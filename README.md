# Bio Segmentation — Polyp segmentation on Kvasir-SEG

End-to-end binary medical image segmentation pipeline: public dataset → split → augmentation → U-Net training → test evaluation → inference.

## Dataset

**Kvasir-SEG** (Jha et al., *MMM 2020*): 1000 colonoscopy frames with pixel-level polyp masks, resolutions from 332×487 to 1920×1072. Free, no login: <https://datasets.simula.no/kvasir-seg/>.

Split (seed 42, fixed in `data/splits/*.txt`): **800 train / 100 val / 100 test**. The test set is used once, only for the final numbers.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1. Data | [bioseg/download.py](bioseg/download.py) | Downloads + extracts the zip, checks every image has a mask, writes the train/val/test split |
| 2. Dataset | [bioseg/data.py](bioseg/data.py) | Reads RGB image + JPEG mask (threshold at 127 to remove compression noise), resizes to 352², augments (flips, rot90, affine, colour jitter, blur), ImageNet normalization |
| 3. Model | [bioseg/model.py](bioseg/model.py) | U-Net with ImageNet-pretrained ResNet34 encoder (`segmentation_models_pytorch`), loss = 0.5·BCE + 0.5·Dice, per-image Dice / IoU / precision / recall |
| 4. Train | [bioseg/train.py](bioseg/train.py) | AdamW, 2-epoch warmup + cosine LR, bf16 autocast, keeps the best checkpoint by val Dice, early stopping, `log.csv` |
| 5. Evaluate | [bioseg/evaluate.py](bioseg/evaluate.py) | Test set at **original resolution** (prediction upsampled back), flip TTA, `summary.json`, `per_image.csv`, training curves, best/worst case figures |
| 6. Predict | [bioseg/predict.py](bioseg/predict.py) | Masks + overlays for any image folder |

All hyperparameters are in [configs/kvasir_unet.yaml](configs/kvasir_unet.yaml). To try another architecture, change `model.arch` (e.g. `UnetPlusPlus`, `DeepLabV3Plus`) or `model.encoder`.

## Run

Requires [uv](https://docs.astral.sh/uv/). `pyproject.toml` pulls CUDA 12.8 PyTorch wheels, which RTX 50-series GPUs need.

```bash
uv sync                                         # install
uv run python -m bioseg.download                # 1. data + splits
uv run python -m bioseg.train --epochs 2        #    (optional smoke test)
uv run python -m bioseg.train                   # 4. train  -> runs/kvasir_unet_r34/
uv run python -m bioseg.evaluate                # 5. test metrics + figures
uv run python -m bioseg.predict --input path/to/images --output predictions   # 6. inference
```

## Results (U-Net / ResNet34, 352², RTX 5060 Ti)

Early stopping ended training at epoch 41. The best checkpoint is epoch 26 (val mDice 0.913). Each epoch takes about 6 s, so the whole run takes about 4.5 min.

| Test set (100 images, original resolution) | mDice | mIoU | Precision | Recall | Dice < 0.5 |
|---|---|---|---|---|---|
| no TTA | 0.902 | 0.843 | — | — | — |
| **flip TTA** | **0.910** | **0.855** | 0.924 | 0.924 | 4 / 100 |

For context, published U-Net-style baselines on Kvasir-SEG are usually around 0.82–0.90 mDice. Papers use different splits, though, so these numbers are not directly comparable.

**Failure modes** (see `runs/kvasir_unet_r34/eval_test/worst.png`): most of the worst cases are pedunculated polyps. The annotation sometimes includes the stalk and sometimes only the head, and the model is inconsistent in the same way. The rest are flat, large lesions where the model finds only part of the lesion. Train and val loss track each other closely, so there is no sign of overfitting.

Ideas to try next: a bigger input size (448–512), a stronger encoder (e.g. `efficientnet-b4`, `mit_b2`), `UnetPlusPlus`, or tuning the threshold on the validation set.

## Metrics

Dice and IoU are computed **per image and then averaged** (mDice / mIoU), which is the usual convention in polyp segmentation papers. Pooling all pixels into one global Dice would let the large polyps dominate the score.
