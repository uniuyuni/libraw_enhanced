# LibRaw Enhanced

Enhanced LibRaw Python wrapper with Apple Silicon Metal GPU acceleration.  
Version **0.11.7**

## Features

- rawpy-compatible API (`imread`, `postprocess`, context manager)
- Apple Silicon Metal GPU acceleration for demosaicing, defringing, CA correction, tone mapping, and more
- Pyramidal Lucas-Kanade lateral chromatic aberration (CA) correction (CPU)
- Guided-filter axial CA correction (CPU + Metal GPU via MPS)
- Edge-gated Gaussian chroma-suppression defringe — purple/green fringe removal (CPU + Metal GPU, ~3.7× speedup)
- Detail tone mapping (CPU + Metal GPU, ~3.7× speedup on 51 MP)
- Micro-contrast enhancement (CPU + Metal GPU)
- Full Bayer (linear, AMaZE) and X-Trans (1-pass, 3-pass) demosaicing on Metal GPU
- Standalone post-processing methods that operate on float32 numpy arrays

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import libraw_enhanced as lre

with lre.imread('image.CR2') as raw:
    rgb = raw.postprocess(
        use_camera_wb=True,
        use_gpu_acceleration=True,   # enable Metal GPU pipeline
        defringe=True,               # remove purple/green fringes
        lateral_ca_correction=True,  # sub-pixel lateral CA registration
        axial_ca_correction=True,    # guided-filter axial CA cleanup
    )
```

## Requirements

- Python 3.8+
- LibRaw (system library)
  - macOS: `brew install libraw`
  - Ubuntu/Debian: `sudo apt-get install libraw-dev`
- macOS 12+ with Apple Silicon (M1/M2/M3) for Metal acceleration
- pybind11 ≥ 2.10.0, NumPy ≥ 1.19.0

## API Reference

### `lre.imread(filepath)` / `lre.imread_buffer(buffer)`

Opens a RAW file (or in-memory buffer) and returns a `RawImage` object.  
Use as a context manager to ensure resources are released.

### `RawImage.postprocess(**kwargs) → np.ndarray`

Develops the RAW image and returns an `(H, W, 3)` NumPy array.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_camera_wb` | bool | `True` | Use embedded camera white balance |
| `use_auto_wb` | bool | `False` | Automatic white balance |
| `user_wb` | tuple[float,float,float,float] | `None` | Manual WB multipliers (R,G,B,G) |
| `half_size` | bool | `False` | Half-size output for speed |
| `four_color_rgb` | bool | `False` | Separate interpolation for two green channels |
| `output_bps` | int | `16` | Output bit depth (8 or 16) |
| `user_flip` | int | `None` | Rotation override (-1=auto, 0/1/2/3) |
| `demosaic_algorithm` | DemosaicAlgorithm | `VNG` | Demosaicing algorithm |
| `dcb_iterations` | int | `0` | DCB interpolation iterations |
| `dcb_enhance` | bool | `False` | DCB color enhancement |
| `fbdd_noise_reduction` | FBDDNoiseReduction | `Off` | FBDD noise reduction level |
| `noise_thr` | float | `None` | Wavelet denoising threshold |
| `median_filter_passes` | int | `0` | Median filter passes |
| `output_color` | ColorSpace | `sRGB` | Output color space |
| `bright` | float | `1.0` | Brightness multiplier |
| `no_auto_bright` | bool | `False` | Disable automatic brightness |
| `auto_bright_thr` | float | `None` | Auto-brightness clip threshold |
| `adjust_maximum_thr` | float | `0.75` | Maximum adjustment threshold |
| `highlight_mode` | HighlightMode | `Clip` | Highlight recovery mode |
| `exp_shift` | float | `1.0` | Exposure shift in linear scale (0.25–8.0) |
| `exp_preserve_highlights` | float | `0.0` | Highlight preservation (0.0–1.0) |
| `gamma` | tuple[float,float] | `(0.0, 0.0)` | Gamma curve (power, slope) |
| `no_auto_scale` | bool | `False` | Disable automatic scaling |
| `chromatic_aberration` | tuple[float,float] | `None` | Legacy RAW-space radial (R_scale, B_scale); prefer `lateral_ca_correction` |
| `user_black` | int | `None` | Custom black level |
| `user_sat` | int | `None` | Custom saturation level |
| `bad_pixels_path` | str | `None` | Path to bad-pixel correction file |
| `use_gpu_acceleration` | bool | `False` | Enable Metal GPU pipeline (Apple Silicon) |
| `preprocess` | bool | `False` | Stop before demosaic and return Bayer data |
| **Defringe** | | | |
| `defringe` | bool | `False` | Enable chroma-suppression fringe removal |
| `defringe_radius` | float | `10.0` | Gaussian blur radius for fringe detection (px) |
| `defringe_strength` | float | `10.0` | Correction strength / detection sensitivity |
| `defringe_green` | bool | `False` | Also correct green fringes (off by default to protect natural highlights) |
| `defringe_green_strength` | float | `0.3` | Green fringe correction strength |
| **Lateral CA** | | | |
| `lateral_ca_correction` | bool | `False` | Pyramidal LK sub-pixel lateral CA registration (post-demosaic) |
| `lateral_ca_cell_size` | int | `96` | Cell size for flow estimation grid |
| `lateral_ca_max_iterations` | int | `3` | LK solver iterations per pyramid level |
| `lateral_ca_max_shift` | float | `6.0` | Maximum allowed shift in pixels |
| `lateral_ca_min_confidence` | float | `0.02` | Minimum flow confidence threshold |
| `lateral_ca_pyramid_levels` | int | `3` | Number of coarse-to-fine pyramid levels |
| **Axial CA** | | | |
| `axial_ca_correction` | bool | `False` | Guided-filter axial CA cleanup (runs after lateral CA) |
| `axial_ca_radius` | int | `6` | Guided filter radius |
| `axial_ca_epsilon` | float | `1e-4` | Guided filter regularisation |
| `axial_ca_strength` | float | `0.3` | Axial CA correction strength |

### Standalone methods

These operate on an already-processed `(H, W, 3) float32` NumPy array (linear RGB, 0.0–1.0):

```python
with lre.imread('image.CR2') as raw:
    rgb = raw.postprocess(use_camera_wb=True)

    # Detail tone mapping (ACES-based local contrast)
    tonemapped = raw.tone_mapping(rgb, after_scale=1.0)

    # Micro-contrast enhancement
    enhanced = raw.enhance_micro_contrast(
        rgb, threshold=-1.0, strength=8.0, target_contrast=0.06
    )

    # Defringe (also available standalone)
    defringed = raw.defringe(rgb, radius=10.0, strength=10.0, defringe_green=False)
```

### Platform detection

```python
lre.is_apple_silicon()   # True on M1/M2/M3
lre.is_metal_available() # True when Metal GPU pipeline is usable
lre.get_platform_info()  # dict with OS, CPU, Python version details
lre.get_version_info()   # package + core version strings
```

## Metal GPU Acceleration

On Apple Silicon, enabling `use_gpu_acceleration=True` routes the following pipeline stages to Metal compute kernels:

| Stage | Metal kernel | Notes |
|-------|-------------|-------|
| White balance | `apply_white_balance_bayer` / `_xtrans` | |
| Demosaicing – Bayer linear | `demosaic_bayer_linear` | |
| Demosaicing – Bayer AMaZE | `demosaic_bayer_amaze` | Multi-pass |
| Demosaicing – X-Trans 1-pass | `demosaic_xtrans_1pass` | |
| Demosaicing – X-Trans 3-pass | `demosaic_xtrans_3pass` | |
| Detail tone mapping | `detail_tonemap` | ~3.7× vs CPU |
| Micro-contrast enhance | `enhance_micro_contrast` | |
| Defringe | `defringe` | ~3.7× vs CPU |
| Axial CA (guided filter) | MPS ImageBox + `axial_ca` | |
| Color space conversion | `convert_color_space` | |
| Gamma correction | `gamma_correct` | |

Lateral CA correction currently runs on CPU (the Metal kernel `lateral_ca` exists but GPU dispatch is disabled pending further tuning).

## ICC Profiles

ICC profiles are not included in this repository. Download them from Google Drive and place them in the `icc/` directory:

[Download ICC profiles](https://drive.google.com/drive/folders/1dWrL7ciw5DWlk9zFEBf63Gz9uKsWjJ_W)

```
icc/
├── sRGB IEC61966-2.1.icc
├── Adobe RGB (1998).icc
├── Display P3.icc
├── ACEScg.icc
├── ACES2065-1.icc
├── ITU-R BT.709.icc
├── ITU-R BT.2020.icc
├── ProPhoto RGB.icc
├── WideGamut RGB.icc
└── XYZD65.icc
```

## Development

```bash
# Build and install in editable mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
pytest tests/ -m "not slow" -v       # skip slow tests
pytest tests/ -m metal -v            # Metal-specific tests

# Code quality
black libraw_enhanced/ tests/
flake8 libraw_enhanced/ tests/
mypy libraw_enhanced/

# Clean rebuild
pip uninstall libraw_enhanced
rm -rf build/ *.so libraw_enhanced/_core.*
pip install -e ".[dev]"
```

Set `LIBRAW_ENHANCED_DEBUG=1` for verbose initialisation output.
