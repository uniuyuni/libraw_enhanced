import json
import os
import time
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW_FIXTURE = ROOT / "tests" / "fixtures" / "x1d-xcd45-04.3FR"
RESULT_DIR = ROOT / "tests" / "results"


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _box_blur_2d(image, radius):
    if radius <= 0:
        return np.asarray(image, dtype=np.float32)

    src = np.asarray(image, dtype=np.float32)
    padded = np.pad(src, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(integral, axis=0, dtype=np.float32)
    integral = np.cumsum(integral, axis=1, dtype=np.float32)

    size = 2 * radius + 1
    window_sum = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return window_sum / float(size * size)


def _chroma_residuals(rgb, radius=2):
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise AssertionError(f"expected HxWx3 RGB image, got {arr.shape}")

    arr = np.maximum(arr, 0.0)
    finite = np.isfinite(arr).all(axis=2)
    finite_values = arr[finite]
    eps = 1e-8
    if finite_values.size:
        eps = max(float(np.percentile(finite_values, 99.0)) * 1e-6, eps)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    log_rg = np.log((r + eps) / (g + eps))
    log_bg = np.log((b + eps) / (g + eps))

    residual_rg = log_rg - _box_blur_2d(log_rg, radius)
    residual_bg = log_bg - _box_blur_2d(log_bg, radius)
    return np.sqrt(residual_rg * residual_rg + residual_bg * residual_bg).astype(
        np.float32, copy=False
    )


def _smooth_midtones_mask(rgb):
    arr = np.asarray(rgb, dtype=np.float32)
    finite = np.isfinite(arr).all(axis=2)
    arr = np.maximum(arr, 0.0)
    luma = 0.25 * arr[:, :, 0] + 0.50 * arr[:, :, 1] + 0.25 * arr[:, :, 2]

    valid_luma = luma[finite]
    if valid_luma.size == 0:
        raise AssertionError("image has no finite pixels")

    lo = max(float(np.percentile(valid_luma, 1.0)), 1e-6)
    hi = float(np.percentile(valid_luma, 99.5))
    valid = finite & (luma > lo) & (luma < hi)

    local_luma = np.abs(luma - _box_blur_2d(luma, radius=2))
    edge_limit = float(np.percentile(local_luma[valid], 65.0))
    smooth = valid & (local_luma <= edge_limit)

    if np.count_nonzero(smooth) < max(1024, smooth.size // 200):
        raise AssertionError("not enough smooth midtone pixels for chroma-noise analysis")
    return smooth


def _summarize_residuals(residual, mask):
    samples = np.asarray(residual[mask], dtype=np.float32)
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95.0)),
        "p99": float(np.percentile(samples, 99.0)),
        "max": float(np.max(samples)),
    }


def _process_x1d(all_corrections, **overrides):
    import libraw_enhanced as lre

    params = {
        "use_camera_wb": True,
        "use_auto_wb": False,
        "half_size": not _env_bool("LRE_X1D_FULL_RES"),
        "output_bps": 32,
        "demosaic_algorithm": lre.DemosaicAlgorithm.AMaZE,
        "output_color": lre.ColorSpace.Raw,
        "gamma": (1.0, 1.0),
        "highlight_mode": 5,
        "no_auto_bright": True,
        "use_gpu_acceleration": _env_bool("LRE_X1D_USE_GPU"),
    }
    if all_corrections:
        params.update(
            {
                "defringe": True,
                "defringe_green": True,
                "lateral_ca_correction": True,
                "axial_ca_correction": True,
            }
        )
    params.update(overrides)

    started = time.perf_counter()
    with lre.imread(str(RAW_FIXTURE)) as raw:
        image = raw.postprocess(**params)
    elapsed = time.perf_counter() - started

    return np.asarray(image, dtype=np.float32), elapsed, params


def _tile_density(mask, tile_size=256):
    height, width = mask.shape
    max_density = 0.0
    max_count = 0
    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            tile = mask[y0:y1, x0:x1]
            count = int(np.count_nonzero(tile))
            if count == 0:
                continue
            density = count / float(tile.size)
            if density > max_density:
                max_density = density
                max_count = count
    return max_density, max_count


@pytest.mark.skipif(not RAW_FIXTURE.exists(), reason=f"fixture missing: {RAW_FIXTURE}")
@pytest.mark.skipif(
    not _env_bool("LRE_RUN_HEAVY_RAW_TESTS"),
    reason="set LRE_RUN_HEAVY_RAW_TESTS=1 to run the large X1D RAW diagnostic",
)
def test_x1d_all_corrections_do_not_increase_smooth_chroma_noise():
    baseline, baseline_seconds, baseline_params = _process_x1d(all_corrections=False)
    corrected, corrected_seconds, corrected_params = _process_x1d(all_corrections=True)

    assert baseline.shape == corrected.shape
    assert baseline.dtype == corrected.dtype == np.float32
    assert np.isfinite(baseline).all()
    assert np.isfinite(corrected).all()

    mask = _smooth_midtones_mask(baseline)
    baseline_residual = _chroma_residuals(baseline)
    corrected_residual = _chroma_residuals(corrected)

    baseline_stats = _summarize_residuals(baseline_residual, mask)
    corrected_stats = _summarize_residuals(corrected_residual, mask)

    increase = corrected_residual[mask] - baseline_residual[mask]
    baseline_p95 = max(baseline_stats["p95"], 1e-8)
    p95_ratio = corrected_stats["p95"] / baseline_p95
    p99_ratio = corrected_stats["p99"] / max(baseline_stats["p99"], 1e-8)
    speckle_fraction = float(
        np.mean(increase > max(0.010, baseline_stats["p95"] * 0.50))
    )

    metrics = {
        "fixture": str(RAW_FIXTURE),
        "shape": list(baseline.shape),
        "mask_fraction": float(np.mean(mask)),
        "baseline_seconds": float(baseline_seconds),
        "corrected_seconds": float(corrected_seconds),
        "baseline_params": baseline_params,
        "corrected_params": corrected_params,
        "baseline_chroma_residual": baseline_stats,
        "corrected_chroma_residual": corrected_stats,
        "p95_ratio": float(p95_ratio),
        "p99_ratio": float(p99_ratio),
        "speckle_fraction": speckle_fraction,
    }

    RESULT_DIR.mkdir(exist_ok=True)
    metrics_path = RESULT_DIR / "x1d_all_corrections_chroma_noise_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    max_p95_ratio = float(os.environ.get("LRE_X1D_MAX_CHROMA_NOISE_P95_RATIO", "1.35"))
    max_p99_ratio = float(os.environ.get("LRE_X1D_MAX_CHROMA_NOISE_P99_RATIO", "1.50"))
    max_speckle_fraction = float(
        os.environ.get("LRE_X1D_MAX_CHROMA_SPECKLE_FRACTION", "0.015")
    )

    assert p95_ratio <= max_p95_ratio, (
        f"smooth-area chroma noise p95 increased {p95_ratio:.3f}x; "
        f"metrics: {metrics_path}"
    )
    assert p99_ratio <= max_p99_ratio, (
        f"smooth-area chroma noise p99 increased {p99_ratio:.3f}x; "
        f"metrics: {metrics_path}"
    )
    assert speckle_fraction <= max_speckle_fraction, (
        f"new chroma speckles in {speckle_fraction:.3%} of smooth pixels; "
        f"metrics: {metrics_path}"
    )


@pytest.mark.skipif(not RAW_FIXTURE.exists(), reason=f"fixture missing: {RAW_FIXTURE}")
@pytest.mark.skipif(
    not _env_bool("LRE_RUN_HEAVY_RAW_TESTS"),
    reason="set LRE_RUN_HEAVY_RAW_TESTS=1 to run the large X1D RAW diagnostic",
)
def test_x1d_defringe_green_scope_stays_sparse():
    green_off, off_seconds, off_params = _process_x1d(
        all_corrections=True,
        half_size=False,
        defringe_green=False,
    )
    green_on, on_seconds, on_params = _process_x1d(
        all_corrections=True,
        half_size=False,
        defringe_green=True,
    )

    assert green_off.shape == green_on.shape
    assert green_off.dtype == green_on.dtype == np.float32

    g_drop = green_off[:, :, 1] - green_on[:, :, 1]
    green_changed = np.abs(g_drop) > 1e-8
    changed_fraction = float(np.mean(green_changed))
    max_tile_density, max_tile_count = _tile_density(green_changed)

    metrics = {
        "fixture": str(RAW_FIXTURE),
        "shape": list(green_off.shape),
        "off_seconds": float(off_seconds),
        "on_seconds": float(on_seconds),
        "green_off_params": off_params,
        "green_on_params": on_params,
        "changed_fraction": changed_fraction,
        "changed_count": int(np.count_nonzero(green_changed)),
        "max_tile_density": float(max_tile_density),
        "max_tile_count": int(max_tile_count),
        "g_drop": _summarize_residuals(np.abs(g_drop), green_changed),
    }
    RESULT_DIR.mkdir(exist_ok=True)
    metrics_path = RESULT_DIR / "x1d_defringe_green_scope_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    max_changed_fraction = float(
        os.environ.get("LRE_X1D_MAX_DEFRINGE_GREEN_CHANGED_FRACTION", "0.0015")
    )
    max_tile_density_limit = float(
        os.environ.get("LRE_X1D_MAX_DEFRINGE_GREEN_TILE_DENSITY", "0.08")
    )

    assert changed_fraction <= max_changed_fraction, (
        f"defringe_green changed {changed_fraction:.3%} of X1D pixels; "
        f"metrics: {metrics_path}"
    )
    assert max_tile_density <= max_tile_density_limit, (
        f"defringe_green concentrated in a {max_tile_density:.2%} tile; "
        f"metrics: {metrics_path}"
    )
