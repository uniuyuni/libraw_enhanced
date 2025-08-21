# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LibRaw Enhanced is a high-performance Python wrapper for LibRaw with Apple Silicon Metal acceleration. It provides rawpy-compatible API with additional Metal Performance Shaders acceleration for Apple Silicon Macs.

### Key Architecture

- **Core C++ Extension**: Located in `core/` directory with pybind11 bindings
- **Python API Layer**: Located in `libraw_enhanced/` with high-level API and constants
- **Apple Silicon Optimization**: GPU/CPU acceleration via `core/accelerator.cpp`
- **Build System**: Dual build system using both setuptools (`setup.py`) and CMake (`CMakeLists.txt`)

## Common Development Commands

### Building and Installation
```bash
# Development installation
pip install -e ".[dev]"

# Build with CMake (alternative)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Clean build (recommended for development)
pip uninstall libraw_enhanced
rm -rf build/ *.so libraw_enhanced/_core.*
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m unit -v
pytest tests/ -m integration -v
pytest tests/ -m performance -v
pytest tests/ -m "not slow" -v

# Run tests with coverage
pytest tests/ -v --cov=libraw_enhanced

# Run single test file
pytest tests/test_basic_functionality.py -v

# Tox testing across Python versions
tox
tox -e py311  # specific Python version
tox -e lint   # linting only
```

### Code Quality
```bash
# Format code
black libraw_enhanced/ tests/

# Check formatting
black --check libraw_enhanced/ tests/

# Lint code
flake8 libraw_enhanced/ tests/

# Type checking
mypy libraw_enhanced/

# Import sorting
isort libraw_enhanced/ tests/
isort --check-only libraw_enhanced/ tests/

# All linting at once (via tox)
tox -e lint
```

## Build Dependencies

### Required System Libraries
- **LibRaw**: Core RAW processing library
  - macOS: `brew install libraw`
  - Ubuntu/Debian: `sudo apt-get install libraw-dev`
  - CentOS/RHEL: `sudo yum install LibRaw-devel`

### Apple-Specific Dependencies
- **Apple Silicon Macs**: Metal, MetalPerformanceShaders, Accelerate frameworks (system provided)
- **Intel Macs**: Limited Metal support, fallback CPU processing

### Python Build Dependencies
- pybind11 >= 2.10.0
- numpy >= 1.19.0
- setuptools >= 45

## Project Structure

```
libraw_enhanced/
├── core/                          # C++ extension module
│   ├── libraw_wrapper.cpp/.h     # Main LibRaw interface
│   ├── accelerator.cpp/.h        # Unified GPU/CPU acceleration
│   ├── python_bindings.cpp       # pybind11 Python bindings
│   └── metal/                     # Metal shaders and utilities
├── libraw_enhanced/               # Python package
│   ├── __init__.py               # Main module with platform detection
│   ├── high_level_api.py         # rawpy-compatible high-level API
│   └── constants.py              # Enums and constants
├── tests/                         # Test suite and test files
│   ├── fixtures/                 # Test RAW image files
│   ├── results/                  # Test output files and results
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── performance/              # Performance benchmarks
└── docs/                          # Documentation, analysis reports, and specifications
```

## Apple Silicon Optimizations

### Metal Acceleration
- Enabled via `metal_acceleration=True` parameter in processing functions
- Requires macOS 12+ and Apple Silicon (M1/M2/M3) chips
- Fallback to CPU processing on Intel Macs and other platforms

### Detection Functions
- `libraw_enhanced.is_apple_silicon()`: Detects Apple Silicon hardware
- `libraw_enhanced.is_metal_available()`: Checks Metal acceleration availability
- `libraw_enhanced.get_platform_info()`: Returns comprehensive platform information

### Platform-Specific Build Flags
- Apple Silicon: `-mcpu=apple-a14` optimization
- Objective-C++ support with `-fobjc-arc -fmodules`
- Metal framework linking automatically detected and configured

## Testing Strategy

### Test Categories (pytest markers)
- `unit`: Fast unit tests
- `integration`: Integration tests requiring file I/O
- `performance`: Performance benchmarks
- `slow`: Long-running tests
- `apple_silicon`: Tests requiring Apple Silicon hardware
- `metal`: Tests requiring Metal support
- `requires_raw_files`: Tests needing actual RAW image files

### Performance Testing
- `test_metal_performance.py`: Metal vs CPU performance comparisons
- Benchmark tests use `pytest-benchmark` for reliable measurements
- Performance tests can be skipped with `-m "not performance"`

## API Compatibility

### rawpy Compatibility
The package maintains rawpy API compatibility for common use cases:
```python
import libraw_enhanced as lre

with lre.imread('image.CR2') as raw:
    rgb = raw.postprocess(use_camera_wb=True, metal_acceleration=True)
```

### Enhanced Features
- Metal acceleration parameter in processing functions
- Enhanced platform detection and capability reporting
- Custom processing pipeline support (when available)
- Comprehensive processing statistics

## Development Tips

### Debug Mode
Set `LIBRAW_ENHANCED_DEBUG=1` environment variable for verbose debugging output during package initialization.

### Metal Development
- Metal shaders located in `core/metal/`
- Metal validation target: `make validate_metal_shaders` (CMake build)
- Apple Silicon optimization reports available via CMake custom targets

### Fallback Behavior
The package gracefully handles missing dependencies:
- Core module import failures provide helpful error messages
- Platform-specific features fallback to CPU implementations
- Missing optional dependencies (pipeline, Metal) are handled with warnings

## File Organization Guidelines

### Test Files and Scripts
- **Location**: All test-related files should be placed in the `tests/` directory
- **Test Scripts**: Individual test scripts (e.g., `test_cpu_gpu_comparison.py`, `test_image_diff.py`) go in `tests/`
- **Test Results**: Output files, performance results, and generated images go in `tests/results/`
- **Test Data**: RAW image files and fixtures go in `tests/fixtures/`

### Documentation and Analysis
- **Location**: All documentation, analysis reports, and specifications should be placed in the `docs/` directory
- **Analysis Reports**: Performance analysis, GPU implementation status, and technical investigations go in `docs/`
- **Implementation Documentation**: Implementation plans, status reports, and architectural decisions go in `docs/`
- **Examples**: Code examples and usage guides can be placed in `docs/`
- **LibRaw Research**: `docs/LibRaw.md` - Comprehensive analysis of LibRaw's internal structure and image processing pipeline

### Directory Structure
```bash
tests/
├── fixtures/           # Test RAW files (CR2, CR3, RAF, etc.)
├── results/           # Test output images, performance logs, comparison results
├── test_*.py          # Individual test scripts
└── conftest.py        # pytest configuration

docs/
├── *.md              # Analysis reports and documentation
├── examples/          # Code examples and usage guides
└── specifications/    # Technical specifications and requirements
```