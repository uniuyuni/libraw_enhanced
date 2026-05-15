#!/usr/bin/env python3
"""Analyze defringe before/after changes on known ROI pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ROIS = {
    "bike_edge": ("5DS Bike", (3346, 2400, 3375, 2428)),
    "bike_silver": ("5DS Bike", (3589, 2371, 3604, 2378)),
    "bike_faint": ("5DS Bike", (3671, 2264, 3681, 2270)),
    "bike_red": ("5DS Bike", (3836, 2329, 3856, 2337)),
    "room_ref": ("X-T5 Room", (6114, 2736, 6121, 2834)),
}

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def image_paths(result_dir: Path, stem: str) -> tuple[Path, Path]:
    return (
        result_dir / f"{stem}_no defringe.jpg",
        result_dir / f"{stem}_defringe.jpg",
    )


def chroma(rgb: np.ndarray) -> np.ndarray:
    return rgb.max(axis=2) - rgb.min(axis=2)


def saturation(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    return np.where(mx > 1e-6, (mx - mn) / mx, 0.0)


def stats(values: np.ndarray) -> str:
    return (
        f"mean={values.mean(): .6f} "
        f"med={np.quantile(values, 0.5): .6f} "
        f"p10={np.quantile(values, 0.1): .6f} "
        f"p90={np.quantile(values, 0.9): .6f} "
        f"min={values.min(): .6f} "
        f"max={values.max(): .6f}"
    )


def read_pgm_with_max(path: Path) -> np.ndarray:
    data = path.read_bytes()
    tokens: list[str] = []
    i = 0
    while len(tokens) < 4:
        while data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while data[i : i + 1] != b"\n":
                i += 1
            continue
        start = i
        while not data[i : i + 1].isspace():
            i += 1
        tokens.append(data[start:i].decode("ascii"))

    _, w, h, _ = tokens
    while data[i : i + 1].isspace():
        i += 1
    arr = np.frombuffer(data[i:], dtype=">u2").astype(np.float32)
    arr = arr.reshape(int(h), int(w)) / 65535.0
    max_path = Path(str(path) + ".max.txt")
    if max_path.exists():
        arr *= float(max_path.read_text().strip())
    return arr


def report_image_roi(name: str, result_dir: Path, stem: str, roi: tuple[int, int, int, int]) -> None:
    base_path, def_path = image_paths(result_dir, stem)
    base = read_image(base_path)
    corrected = read_image(def_path)
    x1, y1, x2, y2 = roi
    before = base[y1:y2, x1:x2]
    after = corrected[y1:y2, x1:x2]
    delta = after - before

    luma_delta = after @ LUMA - before @ LUMA
    chroma_delta = chroma(after) - chroma(before)
    sat_delta = saturation(after) - saturation(before)
    green_cast = after[:, :, 1] - np.maximum(after[:, :, 0], after[:, :, 2])
    warm_loss = (after[:, :, 0] - after[:, :, 1]) - (before[:, :, 0] - before[:, :, 1])

    print(f"\n[{name}] {stem} roi={roi} size={x2 - x1}x{y2 - y1}")
    for metric, values in (
        ("dR", delta[:, :, 0]),
        ("dG", delta[:, :, 1]),
        ("dB", delta[:, :, 2]),
        ("dLuma", luma_delta),
        ("dChroma", chroma_delta),
        ("dSat", sat_delta),
        ("dWarm(R-G)", warm_loss),
        ("greenCast", green_cast),
    ):
        print(f"  {metric:12s} {stats(values)}")

    for threshold in (0.02, 0.05, 0.10):
        print(
            f"  counts @{threshold:.2f}: "
            f"chroma_drop={(chroma_delta < -threshold).sum():4d} "
            f"sat_drop={(sat_delta < -threshold).sum():4d} "
            f"green_cast={(green_cast > threshold).sum():4d}"
        )


def report_debug_roi(name: str, debug_dir: Path, roi: tuple[int, int, int, int]) -> None:
    if not debug_dir.exists():
        return
    x1, y1, x2, y2 = roi
    maps = [
        "09_warm_protect",
        "11_amount",
        "12_rgb_delta",
        "13_rgb_delta_r",
        "14_rgb_delta_g",
        "15_rgb_delta_b",
        "22_red_strength",
        "23_red_purple",
        "25_line_purple_relief",
        "27_blue_lift",
        "30_red_purple_context",
        "34_neutral_gray_edge",
    ]
    print(f"  debug maps:")
    for map_name in maps:
        path = debug_dir / f"{map_name}.pgm"
        if not path.exists():
            continue
        values = read_pgm_with_max(path)[y1:y2, x1:x2]
        print(f"    {map_name:22s} {stats(values)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("tests/results"))
    parser.add_argument("--bike-debug-dir", type=Path, default=Path("tests/results/bike_defringe_debug"))
    parser.add_argument("--room-debug-dir", type=Path, default=Path("tests/results/room_defringe_debug"))
    args = parser.parse_args()

    for name, (stem, roi) in DEFAULT_ROIS.items():
        report_image_roi(name, args.result_dir, stem, roi)
        debug_dir = args.bike_debug_dir if stem.startswith("5DS") else args.room_debug_dir
        report_debug_roi(name, debug_dir, roi)


if __name__ == "__main__":
    main()
