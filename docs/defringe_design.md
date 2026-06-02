# Defringe Design Notes

This is a short note for the current implementation. It is not a roadmap.

## Goal

Remove narrow chromatic fringes without damaging real color texture.

The defringe pass runs on linear RGB before output color-space conversion and gamma. This keeps detection independent of display tone curves and lets `output_color=Raw` remain a true no-matrix path.

## Current Algorithm

The implementation uses green as the structural guide:

1. Build local edge evidence from the green channel.
2. Compare local `log(R/G)` and `log(B/G)` against blurred neighborhood ratios.
3. Treat narrow high-chroma deviations near strong structure as fringe candidates.
4. Move purple fringe R/B values toward the local green-guided estimate.
5. Optionally process green fringes with stricter gates.

This is not the old Lab/darktable-style design. That approach was useful as a reference, but the current code works directly in linear RGB to avoid repeated color-space conversions and to keep RAW-output analysis stable.

## Green Defringe

`defringe_green=False` is the safer default.

Green fringes are harder to distinguish from real scene detail: foliage, stone texture, fabric, and fine midtone patterns can all contain weak green dominance. The green branch therefore requires stronger chroma and clearer line/edge evidence than the purple branch.

Use `defringe_green=True` only when green fringing is visible. If color texture starts to look thin or speckled, test with green defringe disabled first.

## Processing Order

The relevant post-demosaic order is:

1. Highlight recovery
2. Tone mapping / detail-preserving tone map
3. Micro-contrast
4. Lateral CA registration
5. Axial CA cleanup
6. Defringe
7. Output color conversion
8. Gamma correction
9. Integer-output clamp for `output_bps=8` or `16`

`HighlightMode.RebuildAndMicroContrast` enables steps 1 and 3.
`HighlightMode.RebuildAndDetailToneMap` enables steps 1, 2 using the detail-preserving tone map, and 3.
`HighlightMode.RebuildAndToneMap` enables steps 1, 2 using the standard tone map, and 3.

`output_bps=32` skips the final integer clamp and can preserve HDR/superwhite float values.

## Tests

The lightweight synthetic tests live in `tests/test_defringe.py`.

The large RAW diagnostics are opt-in because they are slow:

```bash
LRE_RUN_HEAVY_RAW_TESTS=1 pixi run pytest tests/test_x1d_color_noise_regression.py -q -s
```

They monitor smooth-area chroma noise and the spatial scope of `defringe_green` on real fixtures.
