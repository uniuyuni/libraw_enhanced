"""
LibRaw Enhanced - High-Level API Implementation
rawpy互換のユーザーフレンドリーなAPIを提供
"""

import os
import numpy as np
from typing import Optional, Union, Tuple, Any, Dict
from pathlib import Path

try:
    from ._core import (
        LibRawWrapper,
        ImageInfo,
    )
    _CORE_AVAILABLE = True
except ImportError as e:
    _CORE_AVAILABLE = False
    import warnings
    warnings.warn(f"Core module not available: {e}")
    
    # Create dummy classes for type annotations when core not available
    class LibRawWrapper: pass
    class ImageInfo: pass

from .constants import (
    FBDDNoiseReduction,
    HighlightMode,
    ColorSpace,
    DemosaicAlgorithm,
)


class RawImage:
    """
    RAW画像を扱うためのメインクラス
    
    rawpyのRawPyオブジェクトとの互換性を提供しながら、
    LibRaw Enhancedの拡張機能にもアクセス可能。
    
    使用例:
        with libraw_enhanced.imread('image.CR2') as raw:
            rgb = raw.postprocess(use_camera_wb=True)
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self._last_timing_info = None  # 最後の処理の計測情報
        """
        RawImageオブジェクトを初期化
        
        Args:
            filepath: RAWファイルのパス（オプション）
        """
        if not _CORE_AVAILABLE:
            raise RuntimeError("LibRaw Enhanced core module not available")
            
        self._wrapper = LibRawWrapper()
        self._filepath = None
        self._is_loaded = False
        self._image_info = None
        self._raw_data = None
        
        if filepath is not None:
            self.load_file(filepath)
    
    def load_file(self, filepath: str):
        """
        RAWファイルを読み込み（rawpy互換：自動でunpackも実行）
        
        Args:
            filepath: RAWファイルのパス
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            RuntimeError: ファイル読み込みに失敗した場合
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"RAW file not found: {filepath}")
        
        # ファイル読み込み
        load_result = self._wrapper.load_file(filepath)
        if load_result != 0:
            raise RuntimeError(f"Failed to load RAW file: {load_result}")
        
        # rawpy互換：自動でunpackを実行
        unpack_result = self._wrapper.unpack()
        if unpack_result != 0:
            raise RuntimeError(f"Failed to unpack RAW data: {unpack_result}")
        
        self._filepath = str(Path(filepath).resolve())
        self._is_loaded = True
        self._image_info = None  # 次回アクセス時に更新
        self._raw_data = None   # 次回アクセス時に更新
    
    def load_buffer(self, buffer: Union[bytes, np.ndarray]):
        """
        メモリバッファからRAWデータを読み込み
        
        Args:
            buffer: RAWデータのバッファ（bytes またはnumpy配列）
            
        Raises:
            RuntimeError: データ読み込みに失敗した場合
        """
        if isinstance(buffer, np.ndarray):
            buffer = buffer.tobytes()
        
        # Convert bytes to list for C++ std::vector<uint8_t>
        buffer_list = list(buffer)
        self._wrapper.load_buffer(buffer_list)
        self._filepath = "<buffer>"
        self._is_loaded = True
        self._image_info = None
    
    def postprocess(self,
                   # Basic processing parameters (rawpy compatible)
                   use_camera_wb: bool = True,
                   half_size: bool = False,
                   four_color_rgb: bool = False,
                   output_bps: int = 16,
                   user_flip: Optional[int] = None,
                   
                   # Demosaicing parameters
                   demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.VNG,
                   dcb_iterations: int = 0,
                   dcb_enhance: bool = False,
                   
                   # Noise reduction parameters
                   fbdd_noise_reduction: FBDDNoiseReduction = FBDDNoiseReduction.Off,
                   noise_thr: Optional[float] = None,
                   median_filter_passes: int = 0,
                   
                   # White balance parameters
                   use_auto_wb: bool = False,
                   user_wb: Optional[Tuple[float, float, float, float]] = None,
                   
                   # Color space and output parameters
                   output_color: ColorSpace = ColorSpace.sRGB,
                   
                   # Brightness and exposure parameters
                   bright: float = 1.0,
                   no_auto_bright: bool = False,
                   auto_bright_thr: Optional[float] = None,
                   adjust_maximum_thr: float = 0.75,
                   
                   # Highlight processing
                   highlight_mode: HighlightMode = HighlightMode.Clip,
                   
                   # NEW: Exposure correction parameters (rawpy compatible)
                   exp_shift: float = 1.0,
                   exp_preserve_highlights: float = 0.0,
                   
                   # Gamma and scaling
                   gamma: Tuple[float, float] = (2.222, 4.5),
                   no_auto_scale: bool = False,
                   
                   # NEW: Color correction parameters (rawpy compatible)
                   chromatic_aberration: Optional[Tuple[float, float]] = None,
                   
                   # User adjustments
                   user_black: Optional[int] = None,
                   user_sat: Optional[int] = None,
                   
                   # NEW: File-based corrections (rawpy compatible)
                   bad_pixels_path: Optional[str] = None,
                   
                   # LibRaw Enhanced extensions
                   use_gpu_acceleration: Optional[bool] = False) -> np.ndarray:
        """
        RAW画像の現像処理を実行 (rawpy完全互換 + 拡張機能)
        
        This method provides full rawpy.postprocess() compatibility with all
        parameters, plus LibRaw Enhanced extensions.
        
        Args:
            # Basic processing parameters
            use_camera_wb: Use camera white balance settings
            half_size: Output half-size image for speed
            four_color_rgb: Use separate interpolation for two green components
            output_bps: Output bit depth (8 or 16)
            user_flip: Manual image rotation (-1=auto, 0,1,2,3=angles)
            
            # Demosaicing parameters
            demosaic_algorithm: Demosaicing algorithm (VNG, AHD, etc.)
            dcb_iterations: DCB interpolation iterations
            dcb_enhance: DCB color enhancement
            
            # Noise reduction parameters  
            fbdd_noise_reduction: FBDD noise reduction level
            noise_thr: Wavelet denoising threshold
            median_filter_passes: Median filter passes
            
            # White balance parameters
            use_auto_wb: Use automatic white balance
            user_wb: Custom white balance multipliers (R,G,B,G)
            
            # Color and output parameters
            output_color: Output color space (sRGB, Adobe RGB, etc.)
            
            # Brightness and exposure parameters
            bright: Brightness multiplier
            no_auto_bright: Disable automatic brightness adjustment
            auto_bright_thr: Auto brightness threshold (default: 0.01)
            adjust_maximum_thr: Maximum adjustment threshold
            
            # Highlight processing
            highlight_mode: Highlight recovery mode (clip, blend, rebuild)
            
            # Exposure correction (NEW - rawpy compatible)
            exp_shift: Exposure shift in linear scale (0.25-8.0)
            exp_preserve_highlights: Highlight preservation (0.0-1.0)
            
            # Gamma and scaling
            gamma: Gamma curve (power, slope) tuple
            no_auto_scale: Disable automatic scaling
            
            # Color correction (NEW - rawpy compatible)
            chromatic_aberration: Chromatic aberration correction (red, blue scales)
            
            # User adjustments
            user_black: Custom black level (-1=auto)
            user_sat: Custom saturation level (-1=auto)
            
            # File-based corrections (NEW - rawpy compatible)
            bad_pixels_path: Path to bad pixels file
            
            # LibRaw Enhanced extensions
            metal_acceleration: Use Metal Performance Shaders (Apple Silicon)
            use_gpu_acceleration: Alternative name for metal_acceleration (overrides if specified)
            custom_pipeline: Custom processing pipeline (future)
            
        Returns:
            numpy.ndarray: Processed RGB image array (height, width, channels)
            
        Raises:
            RuntimeError: If no RAW file loaded or processing fails
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded. Call load_file() first.")
        
        # Build processing parameters from all rawpy-compatible parameters
        from .constants import DemosaicAlgorithm
        
        # Handle GPU acceleration parameters (allow use_gpu_acceleration to override)
        gpu_acceleration = use_gpu_acceleration
        
        # Create parameters structure (assuming it will be passed to C++ layer)
        params = {
            # Basic processing parameters
            'use_camera_wb': use_camera_wb,
            'half_size': half_size,
            'four_color_rgb': four_color_rgb,
            'output_bps': output_bps,
            'user_flip': user_flip if user_flip is not None else -1,
            
            # Demosaicing parameters
            'demosaic_algorithm': int(demosaic_algorithm),
            'dcb_iterations': dcb_iterations,
            'dcb_enhance': dcb_enhance,
            
            # Noise reduction parameters
            'fbdd_noise_reduction': int(fbdd_noise_reduction),
            'noise_thr': noise_thr if noise_thr is not None else 0.0,
            'median_filter_passes': median_filter_passes,
            
            # White balance parameters
            'use_auto_wb': use_auto_wb,
            'user_wb': list(user_wb) if user_wb is not None else [1.0, 1.0, 1.0, 1.0],
            
            # Color space and output parameters
            'output_color_space': int(output_color),
            
            # Brightness and exposure parameters
            'bright': bright,
            'no_auto_bright': no_auto_bright,
            'auto_bright_thr': auto_bright_thr if auto_bright_thr is not None else 0.01,
            'adjust_maximum_thr': adjust_maximum_thr,
            
            # Highlight processing
            'highlight_mode': int(highlight_mode),
            
            # NEW: Exposure correction parameters
            'exp_shift': exp_shift,
            'exp_preserve_highlights': exp_preserve_highlights,
            
            # Gamma and scaling
            'gamma_power': gamma[0],
            'gamma_slope': gamma[1],
            'no_auto_scale': no_auto_scale,
            
            # NEW: Color correction parameters
            'chromatic_aberration_red': chromatic_aberration[0] if chromatic_aberration else 1.0,
            'chromatic_aberration_blue': chromatic_aberration[1] if chromatic_aberration else 1.0,
            
            # User adjustments
            'user_black': user_black if user_black is not None else -1,
            'user_sat': user_sat if user_sat is not None else -1,
            
            # NEW: File-based corrections
            'bad_pixels_path': bad_pixels_path if bad_pixels_path is not None else '',
            
            # LibRaw Enhanced extensions
            'use_gpu_acceleration': gpu_acceleration,
        }
                
        # Convert parameter dict to the format expected by C++
        float_params = {}
        int_params = {}
        bool_params = {}
        string_params = {}
        
        # Categorize parameters by type
        for key, value in params.items():
            if isinstance(value, bool):
                bool_params[key] = value
            elif isinstance(value, int):
                int_params[key] = value
            elif isinstance(value, float):
                float_params[key] = value
            elif isinstance(value, str):
                string_params[key] = value
            elif isinstance(value, list) and len(value) == 4:
                # Handle user_wb array
                if key == 'user_wb':
                    for i, val in enumerate(value):
                        float_params[f'user_wb_{i}'] = float(val)
        
        # Execute processing with categorized parameters
        result_image = self._wrapper.process_with_dict(
            float_params, int_params, bool_params, string_params
        )
        
        # Save timing information for profiling
        if hasattr(result_image, 'timing_info'):
            self._last_timing_info = result_image.timing_info
        
        # Convert to NumPy array and return
        return result_image.to_numpy()
    
    def unpack(self):
        """
        RAWデータの展開処理
        
        通常は自動的に実行されるため、明示的に呼び出す必要はありません。
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded")
        
        self._wrapper.unpack()
    
    def get_timing_info(self):
        """
        最後の処理の詳細計測情報を取得
        
        Returns:
            ProcessingTimes: 処理時間の詳細情報 (利用可能な場合)
            None: 計測情報が利用できない場合
        """
        return self._last_timing_info
    
    def close(self):
        """リソースの解放"""
        if self._wrapper:
            self._wrapper.close()
        self._is_loaded = False
        self._filepath = None
        self._image_info = None
    
    # コンテキストマネージャサポート
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # プロパティ（rawpy互換）
    @property
    def raw_image(self) -> np.ndarray:
        """
        展開されたRAW画像データを取得
        
        Returns:
            numpy.ndarray: RAW画像データ配列
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded")
        
        # キャッシュされたRAWデータがあれば返す
        if self._raw_data is not None:
            return self._raw_data
        
        # LibRawWrapperから直接numpy配列として取得
        try:
            raw_array = self._wrapper.get_raw_image_as_numpy()
            self._raw_data = raw_array
            return raw_array
        except Exception as e:
            # フォールバック：基本情報のみ返す
            info = self.sizes
            print(f"⚠️ RAW data access failed: {e}, returning zeros")
            return np.zeros((info.raw_height, info.raw_width), dtype=np.uint16)
    
    @property
    def sizes(self) -> ImageInfo:
        """
        画像サイズ情報を取得
        
        Returns:
            ImageInfo: 画像メタデータ情報
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded")
        
        if self._image_info is None:
            self._image_info = self._wrapper.get_image_info()
        
        return self._image_info
    
    @property
    def color(self) -> Dict[str, Any]:
        """
        色情報を取得（rawpy互換性のため）
        
        Returns:
            dict: 色情報の辞書
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded")
        
        info = self.sizes
        return {
            'cam_mul': list(info.cam_mul),
            'pre_mul': list(info.pre_mul),
            'rgb_cam': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # 簡略化
        }
    
    @property
    def params(self) -> Dict[str, Any]:
        """
        現在の処理パラメータを取得（rawpy互換性のため）
        
        Returns:
            dict: パラメータの辞書
        """
        return {
            'use_camera_wb': True,
            'output_color': 1,  # sRGB
            'output_bps': 8,
        }
    
    # LibRaw Enhanced拡張プロパティ
    @property
    def camera_info(self) -> Dict[str, str]:
        """
        カメラ情報を取得
        
        Returns:
            dict: カメラメーカー、モデル等の情報
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded")
        
        return {
            'make': self._wrapper.get_camera_make(),
            'model': self._wrapper.get_camera_model(),
            'filepath': self._filepath,
        }
    
    @property
    def optimization_info(self) -> Dict[str, bool]:
        """
        最適化機能の利用可能性を取得
        
        Returns:
            dict: 最適化機能の利用可能性
        """
        from . import is_gpu_available, is_apple_selicon
        
        return {
            'accelerate_available': True,  # Accelerate framework is always available on macOS
            'apple_silicon': is_apple_selicon(),
            'gpu_available': is_gpu_available(),
        }
    
    def __repr__(self):
        if self._is_loaded:
            try:
                camera = f"{self._wrapper.get_camera_make()} {self._wrapper.get_camera_model()}"
                return f"<RawImage: {camera}, {self._filepath}>"
            except:
                return f"<RawImage: {self._filepath}>"
        else:
            return "<RawImage: not loaded>"


def imread(filepath: str) -> RawImage:
    """
    RAWファイルを読み込み、RawImageオブジェクトを作成
    
    rawpy.imread()と互換性のあるファクトリ関数
    
    Args:
        filepath: RAWファイルのパス
        
    Returns:
        RawImage: 読み込まれたRAW画像オブジェクト
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        RuntimeError: ファイル読み込みに失敗した場合
        
    Usage:
        # 基本的な使用方法
        with libraw_enhanced.imread('image.CR2') as raw:
            rgb = raw.postprocess()
        
        # rawpyからの移行例
        import libraw_enhanced as lre  # 元々は import rawpy
        
        with lre.imread('image.CR2') as raw:  # rawpy.imread と同じ
            rgb = raw.postprocess(use_camera_wb=True)  # rawpy と同じパラメータ
    """
    raw_image = RawImage()
    raw_image.load_file(filepath)
    return raw_image


def imread_buffer(buffer: Union[bytes, np.ndarray]) -> RawImage:
    """
    メモリバッファからRAWデータを読み込み
    
    Args:
        buffer: RAWデータのバッファ
        
    Returns:
        RawImage: 読み込まれたRAW画像オブジェクト
        
    Usage:
        with open('image.CR2', 'rb') as f:
            buffer = f.read()
        
        with libraw_enhanced.imread_buffer(buffer) as raw:
            rgb = raw.postprocess()
    """
    raw_image = RawImage()
    raw_image.load_buffer(buffer)
    return raw_image