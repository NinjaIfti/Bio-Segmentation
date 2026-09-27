"""Known-answer checks for the per-class metrics.

Run: uv run python -m tests.test_metrics3d
"""
import numpy as np
import torch

from bioseg.metrics3d import per_class_scores, summarize

NUM_CLASSES = 3  # background + 2 foreground, same as the Hippocampus task


def volume(boxes: dict[int, tuple[slice, slice, slice]], shape=(40, 40, 40)) -> torch.Tensor:
    v = torch.zeros(shape, dtype=torch.long)
    for label, sl in boxes.items():
        v[sl] = label
    return v


CUBE_A = (slice(10, 20), slice(10, 20), slice(10, 20))
CUBE_B = (slice(25, 33), slice(25, 33), slice(25, 33))


def test_perfect_prediction():
    gt = volume({1: CUBE_A, 2: CUBE_B})
    s = per_class_scores(gt.clone(), gt, NUM_CLASSES, (1.0, 1.0, 1.0), [1.0, 1.0])
    assert np.allclose(s["dice"], 1.0), s["dice"]
    assert np.allclose(s["iou"], 1.0), s["iou"]
    assert np.allclose(s["hd95_mm"], 0.0), s["hd95_mm"]
    assert np.allclose(s["assd_mm"], 0.0), s["assd_mm"]
    assert np.allclose(s["nsd"], 1.0), s["nsd"]
    print("perfect prediction -> Dice 1, HD95 0, NSD 1  OK")


def test_shifted_cube_distance_matches_shift():
    """A cube shifted 2 voxels along x has a 2 mm Hausdorff distance at 1 mm spacing."""
    gt = volume({1: CUBE_A})
    pred = volume({1: (slice(12, 22), CUBE_A[1], CUBE_A[2])})
    s = per_class_scores(pred, gt, NUM_CLASSES, (1.0, 1.0, 1.0), [1.0, 1.0])
    assert np.isclose(s["hd95_mm"][0], 2.0, atol=1e-6), s["hd95_mm"]
    assert 0.0 < s["assd_mm"][0] < 2.0, s["assd_mm"]
    assert s["dice"][0] < 1.0
    print(f"2-voxel shift @1mm -> HD95 {s['hd95_mm'][0]:.3f} mm, ASSD {s['assd_mm'][0]:.3f} mm  OK")


def test_spacing_scales_distances():
    """Same 2-voxel shift, but 3 mm voxels along that axis -> distances triple."""
    gt = volume({1: CUBE_A})
    pred = volume({1: (slice(12, 22), CUBE_A[1], CUBE_A[2])})
    iso = per_class_scores(pred, gt, NUM_CLASSES, (1.0, 1.0, 1.0), [1.0, 1.0])
    aniso = per_class_scores(pred, gt, NUM_CLASSES, (3.0, 1.0, 1.0), [1.0, 1.0])
    assert np.isclose(aniso["hd95_mm"][0], 3 * iso["hd95_mm"][0], rtol=1e-6), (iso, aniso)
    # Dice is spacing-independent: it counts voxels, not millimetres
    assert np.isclose(aniso["dice"][0], iso["dice"][0])
    print(f"3 mm voxels -> HD95 {aniso['hd95_mm'][0]:.3f} mm (3x the 1 mm case)  OK")


def test_nsd_tolerance_behaviour():
    """NSD is 1 when the whole boundary sits inside the tolerance, and drops below it
    once the tolerance is tighter than the error."""
    gt = volume({1: CUBE_A})
    pred = volume({1: (slice(12, 22), CUBE_A[1], CUBE_A[2])})
    loose = per_class_scores(pred, gt, NUM_CLASSES, (1.0,) * 3, [5.0, 5.0])["nsd"][0]
    tight = per_class_scores(pred, gt, NUM_CLASSES, (1.0,) * 3, [0.5, 0.5])["nsd"][0]
    assert np.isclose(loose, 1.0), loose
    assert tight < loose, (tight, loose)
    print(f"NSD: tolerance 5 mm -> {loose:.3f}, tolerance 0.5 mm -> {tight:.3f}  OK")


def test_missing_class_gives_nan_boundary_but_zero_dice():
    """If the model predicts nothing for a class, Dice is 0 but the boundary metrics are
    undefined — there is no predicted surface to measure a distance to."""
    gt = volume({1: CUBE_A, 2: CUBE_B})
    pred = volume({1: CUBE_A})           # class 2 missing entirely
    s = per_class_scores(pred, gt, NUM_CLASSES, (1.0,) * 3, [1.0, 1.0])
    assert np.isclose(s["dice"][1], 0.0), s["dice"]
    assert not np.isfinite(s["hd95_mm"][1]), s["hd95_mm"]
    print("missing class -> Dice 0, HD95 undefined (NaN)  OK")


def test_summary_reports_how_many_cases_were_valid():
    """The undefined case must be excluded from the mean AND counted, not hidden."""
    gt = volume({1: CUBE_A, 2: CUBE_B})
    good = per_class_scores(gt.clone(), gt, NUM_CLASSES, (1.0,) * 3, [1.0, 1.0])
    bad = per_class_scores(volume({1: CUBE_A}), gt, NUM_CLASSES, (1.0,) * 3, [1.0, 1.0])
    summary = summarize([good, good, bad], ["Anterior", "Posterior"])

    post = summary["per_class"]["Posterior"]
    assert post["n_cases"] == 3
    assert post["hd95_mm_n_valid"] == 2, post          # only 2 of 3 cases had a surface
    assert np.isclose(post["hd95_mm"], 0.0)            # mean over the 2 valid ones
    assert np.isclose(post["dice"], 2 / 3)             # Dice averages over all 3
    print("summary: HD95 over 2/3 valid cases, Dice over 3/3  OK")


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall metric checks passed")


if __name__ == "__main__":
    main()
