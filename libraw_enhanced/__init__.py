#!/usr/bin/env python3
"""
LibRaw Enhanced - High-performance RAW image processing with Metal acceleration

Apple Silicon最適化とMetal Performance Shadersを活用した
高速RAW画像処理ライブラリ
"""

__version__ = "0.1.0"
__author__ = "LibRaw Enhanced Team"

import warnings
import platform

# Core extension import with fallback
try:
    from . import _core
    _CORE_AVAILABLE = True
except ImportError as e:
    _core = None
    _CORE_AVAILABLE = False
    warnings.warn(
        f"LibRaw Enhanced core module not available: {e}\n"
        "Please ensure the package was built correctly for your platform."
    )

# High-level API imports
try:
    from .high_level_api import (
        RawImage, 
        imread,
        imread_buffer,
    )
except ImportError as e:
    # Fallback implementations or warnings
    warnings.warn(f"High-level API not available: {e}")
    
    # Minimal fallback implementations
    def RawImage(*args, **kwargs):
        raise RuntimeError("LibRaw Enhanced not properly installed")
    
    def imread(*args, **kwargs):
        raise RuntimeError("LibRaw Enhanced not properly installed")

# Core classes (low-level API)
if _CORE_AVAILABLE:
    try:
        Accelerator = _core.Accelerator
        ImageBufferFloat = _core.ImageBufferFloat
        ProcessingParams = _core.ProcessingParams
    except AttributeError:
        # These classes might not be available in all builds
        Accelerator = None
        ImageBufferFloat = None
        ProcessingParams = None
else:
    Accelerator = None
    ImageBufferFloat = None
    ProcessingParams = None

# Constants and enums
from .constants import (
    ColorSpace,
    HighlightMode, 
    FBDDNoiseReduction,
    NoiseReduction,
    DemosaicAlgorithm
)

# Platform and capability detection
def get_platform_info():
    """プラットフォーム情報の取得"""
    import platform  # Re-import to ensure availability
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "core_available": _CORE_AVAILABLE,
    }
    
    if _CORE_AVAILABLE:
        try:
            build_info = _core.get_build_info()
            info.update(build_info)
            
            info["apple_silicon"] = _core.is_apple_silicon()
            info["available"] = _core.is_available()
            
            if info["available"]:
                info["devices"] = _core.get_device_list()
            
        except (AttributeError, RuntimeError):
            pass
    
    return info

def is_apple_silicon():
    """Apple Silicon環境の検出"""
    if _CORE_AVAILABLE:
        try:
            return _core.is_apple_silicon()
        except AttributeError:
            pass
    
    # Fallback detection
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def is_available():
    """Metal加速の可用性チェック"""
    if _CORE_AVAILABLE:
        try:
            return _core.is_available()
        except AttributeError:
            pass
    
    return False

def get_version_info():
    """バージョン情報の取得"""
    info = {
        "package_version": __version__,
        "platform_info": get_platform_info()
    }
    
    if _CORE_AVAILABLE:
        try:
            info["core_version"] = _core.get_version_info()
        except AttributeError:
            pass
    
    return info

# rawpy互換関数のエクスポート
__all__ = [
    # Core classes (high-level API)
    "RawImage", 
    "imread",
    "imread_buffer",
    
    # Core classes (low-level API)
    "Accelerator", 
    "ImageBufferFloat",
    "ProcessingParams",
    
    # Platform detection  
    "is_apple_silicon",
    "get_platform_info",
    "get_version_info",
    
    # Constants
    "ColorSpace",
    "HighlightMode",
    "FBDDNoiseReduction", 
    "NoiseReduction",
    "DemosaicAlgorithm",
]

# Initialization and startup checks
def _initialize_package():
    """パッケージ初期化時のチェック"""
    
    # 基本的な可用性チェック
    if not _CORE_AVAILABLE:
        warnings.warn(
            "LibRaw Enhanced core module is not available. "
            "Some functionality will be limited. "
            "Please check your installation."
        )
        return
    
    # Apple Silicon特有の最適化情報
    if is_apple_silicon() and is_available():
        # デバッグモード以外では非表示
        import os
        if os.environ.get("LIBRAW_ENHANCED_DEBUG"):
            print("LibRaw Enhanced: Metal acceleration available on Apple Silicon")
    
    # Performance hint for Intel Macs
    elif platform.system() == "Darwin" and platform.machine() == "x86_64":
        if os.environ.get("LIBRAW_ENHANCED_DEBUG"): 
            print("LibRaw Enhanced: Running on Intel Mac, Metal features limited")

# パッケージ初期化実行
_initialize_package()

# Clean up module namespace
del warnings, platform