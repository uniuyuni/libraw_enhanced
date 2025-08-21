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

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```
