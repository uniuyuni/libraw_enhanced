#!/usr/bin/env python3
"""Convert defringe debug PGM maps to viewable PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def convert_file(path: Path, colorize: bool) -> Path:
    image = Image.open(path)
    if image.mode not in {"I", "I;16", "I;16B", "L"}:
        image = image.convert("I")

    # Normalize each diagnostic map independently. The C++ dump already does
    # this, but autocontrast makes PNG viewers display 16-bit maps consistently.
    image_8 = ImageOps.autocontrast(image.convert("L"))
    out_path = path.with_suffix(".png")
    image_8.save(out_path)

    if colorize:
        colored = ImageOps.colorize(image_8, black="#000000", white="#ffffff", mid="#ff4fd8")
        colored.save(path.with_name(f"{path.stem}_color.png"))

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert LIBRAW_ENHANCED_DEFRINGE_DEBUG_DIR .pgm maps to .png."
    )
    parser.add_argument(
        "debug_dir",
        nargs="?",
        default="tests/results",
        help="Directory containing defringe debug .pgm files.",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Also write *_color.png versions for quick visual scanning.",
    )
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    pgms = sorted(debug_dir.glob("*.pgm"))
    if not pgms:
        raise SystemExit(f"No .pgm files found in {debug_dir}")

    for pgm in pgms:
        out_path = convert_file(pgm, args.color)
        print(f"{pgm.name} -> {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
