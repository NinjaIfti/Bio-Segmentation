# Bio Segmentation

Two end-to-end medical image segmentation pipelines on public datasets:

1. **[Kvasir-SEG](#part-1--polyp-segmentation-on-kvasir-seg)** — 2D binary polyp segmentation (U-Net / ResNet34)
2. **[MSD Hippocampus](#part-2--multi-class-3d-segmentation-on-msd-hippocampus)** — 3D multi-class segmentation with **per-class Dice, IoU, HD95, ASSD and NSD**

---

# Part 1 — Polyp segmentation on Kvasir-SEG

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

---

# Part 2 — Multi-class 3D segmentation on MSD Hippocampus

Per-class quantitative analysis (Dice, IoU, HD95, ASSD, NSD) plus per-class qualitative analysis.

## Dataset

**MSD Task04 Hippocampus** (Antonelli et al., *Nature Communications* 2022): 260 labelled 3D MRI volumes at 1 mm isotropic spacing, <http://medicaldecathlon.com/>. Free, no login, downloads in seconds.

Classes: `0 background`, `1 Anterior`, `2 Posterior`. Split (seed 42): **182 train / 39 val / 39 test**, by case.

This task was chosen over Kvasir-SEG for the per-class analysis for two reasons:

- Kvasir-SEG is **binary** — one foreground class. A per-class table would have a background row whose Dice is ~0.98 by construction and carries no information.
- Kvasir-SEG images are JPEGs with **no physical voxel spacing**, so HD95 and ASSD could only be reported in pixels. Here spacing is real, and all distances are in **millimetres**.

## Results

| Class | Dice | IoU | HD95 (mm) | ASSD (mm) | NSD @1 mm |
|---|---|---|---|---|---|
| Anterior | 0.8651 | 0.7645 | 1.600 | 0.571 | 0.9065 |
| Posterior | 0.8488 | 0.7403 | 1.751 | 0.546 | 0.9193 |
| **Mean** | **0.8569** | **0.7524** | **1.676** | **0.558** | **0.9129** |

39 test cases, all 39 valid for every boundary metric. Standard deviations are in `eval_test/summary.json`, per-case values in `eval_test/per_case.csv`.

Training converged at val Dice 0.866 after 120 epochs (~1 s/epoch). For reference, nnU-Net reports ~0.90 Dice on this task, so a plain 3D U-Net at 0.857 is in a sensible range.

### Metric definitions

| Metric | Measures | Units |
|---|---|---|
| Dice, IoU | Volumetric overlap | — |
| **HD95** | 95th percentile of surface-to-surface distance — the worst-case boundary error, with the top 5% of outliers trimmed | mm |
| **ASSD** | Average symmetric surface distance — the typical boundary error | mm |
| **NSD** | Normalized Surface Dice — the fraction of the boundary within a clinical tolerance (1 mm here, i.e. one voxel) | — |

Overlap and boundary metrics answer different questions: a prediction can score a high Dice and still be badly wrong along part of the boundary, which is what HD95 exposes.

### Two reporting decisions worth stating in a write-up

**Boundary metrics are undefined when a class is absent.** If the model predicts nothing for a class, there is no predicted surface, so no distance exists — MONAI returns NaN. Those cases are excluded from the mean *and counted*, via the `hd95_mm_n_valid` field. A mean over 37 of 39 cases is never silently presented as a mean over 39. (On this run all 39 were valid.)

**Geometry is verified, not assumed.** Volumes are centre-padded to 48×64×48 for batching; every case is cropped back to its original shape before scoring. Evaluation asserts per case that padding the reference label and undoing it reproduces the label on disk exactly — if predictions were ever misaligned with the reference, the run fails instead of reporting quietly wrong numbers.

### Qualitative analysis

`eval_test/` holds:

| File | Contents |
|---|---|
| `best.png` / `worst.png` | Per-class contour overlays, ground truth next to prediction. The slice is picked to show *all* classes — Anterior and Posterior sit at opposite ends of the structure's long axis, so a slice across that axis would show only one of them |
| `per_class_distributions.png` | Box plots of all five metrics per class over the 39 test cases |
| `curves.png` (in run dir) | Loss and per-class validation Dice per epoch |

**What the worst cases show:** the outer boundary of the hippocampus is tracked well; the error concentrates at the **internal Anterior/Posterior division**, which the model places a slice or two away from the reference. This is consistent with the numbers — Posterior scores lower on Dice yet has the *better* NSD, i.e. its boundary is mostly within tolerance and the loss is coming from that shared internal face, not from the outer surface.

## Run

```bash
uv sync
uv run python -m bioseg.msd                     # download + case splits
uv run python -m tests.test_metrics3d           # known-answer checks on the metrics
uv run python -m bioseg.train3d                 # train  -> runs/hippocampus_unet3d/
uv run python -m bioseg.evaluate3d              # per-class table + figures
```

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1. Data | [bioseg/msd.py](bioseg/msd.py) | Downloads/extracts an MSD task, reads class names from `dataset.json`, writes case-level splits |
| 2. Dataset | [bioseg/data3d.py](bioseg/data3d.py) | MONAI transforms: intensity normalization, centre pad to a fixed size, flips/affine/intensity augmentation. Volumes are cached in RAM |
| 3. Metrics | [bioseg/metrics3d.py](bioseg/metrics3d.py) | Per-class Dice, IoU, HD95, ASSD, NSD with spacing; aggregation that tracks valid-case counts; markdown table output |
| 4. Train | [bioseg/train3d.py](bioseg/train3d.py) | 3D residual U-Net (MONAI), Dice+CE loss, warmup + cosine LR, bf16, best checkpoint by mean foreground Dice |
| 5. Evaluate | [bioseg/evaluate3d.py](bioseg/evaluate3d.py) | Per-case scoring at original geometry, per-class table, box plots, contour overlays |
| Tests | [tests/test_metrics3d.py](tests/test_metrics3d.py) | Known-answer checks: a 2-voxel shift gives HD95 = 2.000 mm; at 3 mm spacing the same shift gives 6.000 mm; NSD tolerance behaves; absent classes give NaN and are counted |

Config: [configs/hippocampus_unet3d.yaml](configs/hippocampus_unet3d.yaml). The code reads class names from the MSD `dataset.json`, so pointing `data.root` at another MSD task (e.g. Task01 BrainTumour, 3 classes) works without code changes.

---

## Committed results

Figures and tables are checked into [results/](results/) so they can be viewed without re-running anything (`runs/` itself is gitignored):

- [results/hippocampus/](results/hippocampus/) — per-class metrics table, summary, per-case CSV, box plots, best/worst overlays, training curves
- [results/kvasir/](results/kvasir/) — best/worst overlays, training curves
