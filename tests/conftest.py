"""
pytest configuration: pre-load libraw.dylib so the _core extension can be imported.
This runs before any test module import, ensuring the dynamic library is resolved.
"""
import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBRAW = os.path.realpath(
    os.path.join(_HERE, "..", "third_party", "libraw-install", "lib", "libraw.24.dylib")
)

_CORE_AVAILABLE = False

if os.path.exists(_LIBRAW):
    try:
        ctypes.CDLL(_LIBRAW)
        # Verify that the module can actually be imported
        import libraw_enhanced._core  # noqa: F401
        _CORE_AVAILABLE = True
    except (OSError, ImportError):
        pass
