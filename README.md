# LibRaw Enhanced

Enhanced LibRaw Python wrapper with Apple Silicon optimization

## Features

- rawpy compatible API
- Apple Silicon Metal Performance Shaders acceleration
- Custom processing pipeline support
- Enhanced RAW image processing capabilities

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import libraw_enhanced as lre

with lre.imread('image.CR2') as raw:
    rgb = raw.postprocess(use_camera_wb=True, metal_acceleration=True)
```

## Requirements

- Python 3.8+
- LibRaw library
- macOS 12+ (for optimal Apple Silicon support)

## ICC Profiles

ICC profiles are not included in this repository. Download them from Google Drive and place in the `icc/` directory:

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
pip install -e ".[dev]"
pytest tests/
```
