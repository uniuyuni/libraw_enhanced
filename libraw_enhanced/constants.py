#!/usr/bin/env python3
"""
LibRaw Enhanced - Constants and Enums

rawpy互換性とMetal拡張機能のための定数定義
"""

from enum import IntEnum, Enum


# Color Spaces
class ColorSpace(IntEnum):
    """カラースペース定数（rawpy互換・LibRaw互換）"""
    Raw = 0              # LIBRAW_COLORSPACE_NotFound
    sRGB = 1            # LIBRAW_COLORSPACE_sRGB
    AdobeRGB = 2        # LIBRAW_COLORSPACE_AdobeRGB  
    WideGamutRGB = 3    # LIBRAW_COLORSPACE_WideGamutRGB
    ProPhotoRGB = 4     # LIBRAW_COLORSPACE_ProPhotoRGB
    XYZ = 5             # rawpy互換値（LibRawには直接対応なし）
    ACES = 6            # rawpy拡張
    P3D65 = 7           # rawpy拡張
    Rec2020 = 8         # rawpy拡張


# Highlight handling modes
class HighlightMode(IntEnum):
    """ハイライト処理モード（LibRaw互換）"""
    Clip = 0        # クリッピング
    Unclip = 1      # アンクリップ
    Blend = 2       # ブレンド
    Rebuild = 3     # 再構築


# FBDD Noise Reduction
class FBDDNoiseReduction(IntEnum):
    """FBDD ノイズ除去設定（rawpy互換）"""
    Off = 0         # 無効
    Light = 1       # 弱い
    Full = 2        # 強い


# Enhanced Noise Reduction (LibRaw Enhanced extension)
class NoiseReduction(Enum):
    """拡張ノイズ除去設定"""
    Off = "off"
    Standard = "standard"
    Advanced = "advanced"
    MLBased = "ml_based"
    Neural = "neural"


# Demosaic Algorithms
class DemosaicAlgorithm(IntEnum):
    """デモザイクアルゴリズム（rawpy互換・LibRaw互換）"""
    # rawpy/LibRaw標準アルゴリズム（quality値に対応）
    Linear = 0          # Linear/Bilinear interpolation (LibRaw quality=0)
    VNG = 1             # Variable Number of Gradients (LibRaw quality=1)
    PPG = 2             # Patterned Pixel Grouping (LibRaw quality=2)
    AHD = 3             # Adaptive Homogeneity-Directed (LibRaw quality=3)
    DCB = 4             # DCB (Dave Coffin's method) (LibRaw quality=4)
    ModifiedAHD = 5     # Modified AHD (requires GPL2 pack)
    AFD = 6             # AFD (Adaptive Filtered Demosaicing) (requires GPL2 pack)
    VCD = 7             # VCD (Variable Color Demosaicing) (requires GPL2 pack)
    MixedVCDModifiedAHD = 8  # Mixed VCD and Modified AHD (requires GPL2 pack)
    LMMSE = 9           # LMMSE (Linear Minimum Mean Square Error) (requires GPL2 pack)
    AMaZE = 10          # AMaZE (Aliasing minimization and zipper elimination) (requires GPL3 pack)
    DHT = 11            # DHT interpolation (LibRaw quality=11)
    AAHD = 12           # AAHD (Modified AHD variant) (LibRaw quality=12)


# Output bit depths
class OutputBitDepth(IntEnum):
    """出力ビット深度"""
    UInt8 = 8
    UInt16 = 16
    Float32 = 32

# White Balance modes
class WhiteBalance(IntEnum):
    """ホワイトバランスモード"""
    Camera = 0      # カメラ設定
    Auto = 1        # 自動
    Manual = 2      # マニュアル


# Image flip modes
class FlipMode(IntEnum):
    """画像反転モード"""
    None_ = -1      # 反転なし
    Horizontal = 0  # 水平反転
    Vertical = 1    # 垂直反転
    Rotate180 = 2   # 180度回転
    Rotate90CW = 3  # 90度時計回り
    Rotate90CCW = 5 # 90度反時計回り
    Rotate270CW = 6 # 270度時計回り


# Metal specific enums
class MetalPrecision(Enum):
    """Metal処理精度"""
    Auto = "auto"
    Float16 = "float16"
    Float32 = "float32"


class ProcessingMethod(Enum):
    """処理方法"""
    Auto = "auto"
    CPU = "cpu"
    Metal = "metal"
    NeuralEngine = "neural_engine"


# Error codes (LibRaw compatible)
class LibRawError(IntEnum):
    """LibRawエラーコード"""
    SUCCESS = 0
    UNSPECIFIED_ERROR = -1
    FILE_UNSUPPORTED = -2
    REQUEST_FOR_NONEXISTENT_IMAGE = -3
    OUT_OF_ORDER_CALL = -4
    NO_THUMBNAIL = -5
    UNSUPPORTED_THUMBNAIL = -6
    INPUT_CLOSED = -7
    INSUFFICIENT_MEMORY = -100007
    DATA_ERROR = -100008
    IO_ERROR = -100009
    CANCELLED_BY_CALLBACK = -100010
    BAD_CROP = -100011


# File format constants
SUPPORTED_RAW_EXTENSIONS = {
    '.cr2', '.cr3',        # Canon
    '.nef', '.nrw',        # Nikon
    '.arw', '.srf', '.sr2',# Sony
    '.orf',                # Olympus
    '.pef', '.ptx',        # Pentax
    '.raf',                # Fujifilm
    '.rw2',                # Panasonic
    '.dng',                # Adobe DNG
    '.3fr',                # Hasselblad
    '.bay',                # Casio
    '.bmq',                # NuCore
    '.cine',               # Phantom Software
    '.cs1',                # Sinar
    '.dc2',                # Kodak
    '.dcr',                # Kodak
    '.drf',                # Kodak
    '.dsc',                # Kodak
    '.erf',                # Epson
    '.fff',                # Imacon
    '.iiq',                # Phase One
    '.k25',                # Kodak
    '.kdc',                # Kodak
    '.mdc',                # Minolta
    '.mef',                # Mamiya
    '.mos',                # Leaf
    '.mrw',                # Minolta
    '.nrw',                # Nikon
    '.obm',                # Olympus
    '.pxn',                # Logitech
    '.qtk',                # Apple QuickTake
    '.r3d',                # RED
    '.rwl',                # Leica
    '.rwz',                # Rawzor
    '.x3f',                # Sigma
    '.yin',                # Alcatel
}

# Apple Silicon specific constants
APPLE_SILICON_CHIPS = {
    'M1', 'M1 Pro', 'M1 Max', 'M1 Ultra',
    'M2', 'M2 Pro', 'M2 Max', 'M2 Ultra', 
    'M3', 'M3 Pro', 'M3 Max', 'M3 Ultra',
}

# Metal Performance Shaders constants  
METAL_MAX_TEXTURE_SIZE = 16384      # Metal maximum texture dimension
METAL_TILE_SIZE = 16                # Optimal Metal tile size for Apple Silicon
METAL_BUFFER_ALIGNMENT = 256        # Metal buffer alignment requirement

# Performance targets
PERFORMANCE_TARGETS = {
    'apple_silicon_speedup': 3.0,    # 300% speedup target
    'intel_mac_speedup': 2.0,        # 200% speedup target  
    'memory_reduction': 0.6,         # 60% memory usage reduction
    'processing_consistency': 0.05,  # 5% variation tolerance
}

# Utility functions
def is_raw_file(filepath):
    """RAWファイルかどうかを判定"""
    from pathlib import Path
    return Path(filepath).suffix.lower() in SUPPORTED_RAW_EXTENSIONS

def validate_color_space(color_space):
    """カラースペースの有効性チェック"""
    if isinstance(color_space, int):
        return color_space in [cs.value for cs in ColorSpace]
    elif isinstance(color_space, ColorSpace):
        return True
    return False

def validate_demosaic_algorithm(algorithm):
    """デモザイクアルゴリズムの有効性チェック"""
    if isinstance(algorithm, int):
        return algorithm in [da.value for da in DemosaicAlgorithm]
    elif isinstance(algorithm, DemosaicAlgorithm):
        return True
    return False

# Export all constants for wildcard import
__all__ = [
    # Enum classes
    'ColorSpace',
    'HighlightMode',
    'FBDDNoiseReduction',
    'NoiseReduction', 
    'DemosaicAlgorithm',
    'OutputBitDepth',
    'WhiteBalance',
    'FlipMode',
    'MetalPrecision',
    'ProcessingMethod',
    'LibRawError',
    
    # Constants
    'SUPPORTED_RAW_EXTENSIONS',
    'APPLE_SILICON_CHIPS',
    'METAL_MAX_TEXTURE_SIZE',
    'METAL_TILE_SIZE',
    'METAL_BUFFER_ALIGNMENT',
    'PERFORMANCE_TARGETS',
    
    # Utility functions
    'is_raw_file',
    'get_default_params',
    'validate_color_space',
    'validate_demosaic_algorithm',
]