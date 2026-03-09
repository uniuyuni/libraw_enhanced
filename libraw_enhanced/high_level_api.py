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
        self._color_matrix: Optional[np.ndarray] = None
        
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
                   gamma: Tuple[float, float] = (0.0, 0.0), # (2.222, 4.5),
                   no_auto_scale: bool = False,
                   
                   # NEW: Color correction parameters (rawpy compatible)
                   chromatic_aberration: Optional[Tuple[float, float]] = None,
                   
                   # User adjustments
                   user_black: Optional[int] = None,
                   user_sat: Optional[int] = None,
                   
                   # NEW: File-based corrections (rawpy compatible)
                   bad_pixels_path: Optional[str] = None,
                   
                   # LibRaw Enhanced extensions
                   use_gpu_acceleration: Optional[bool] = False,
                   preprocess: bool = False) -> np.ndarray:
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
            preprocess: If true, stops processing before demosaicing and returns the raw/modified bayer data.
            
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
            'preprocess': preprocess,
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
        int_params['preprocess'] = preprocess
        
        # Execute processing with categorized parameters
        result_image = self._wrapper.process_with_dict(
            float_params, int_params, bool_params, string_params
        )
        
        # Save timing information for profiling

        if hasattr(result_image, 'timing_info'):
            self._last_timing_info = result_image.timing_info
            
        # Save camera color transformation matrix if available
        if hasattr(result_image, 'color_matrix'):
            self._color_matrix = np.array(result_image.color_matrix, dtype=np.float32)
        
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
    
    def get_threshold(self) -> float:
        """
        threshold = maximum / data_maximum の値を返す。

        RAWファイルの処理後にハイライトリカバリやマイクロコントラスト強調に
        使うデフォルトの閾値として利用できる。

        Returns:
            float: threshold 値（data_maximum が 0 の場合は 1.0）

        Raises:
            RuntimeError: RAW ファイルが読み込まれていない場合
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded. Call load_file() first.")
        return self._wrapper.get_threshold()

    def recover_highlights(self,
                           image: np.ndarray,
                           threshold: float = -1.0) -> np.ndarray:
        """
        ハイライトリカバリを実行し、新しい numpy 配列を返す。

        Args:
            image: 入力画像 (H, W, 3) float32 numpy 配列, 値域 0.0-1.0
            threshold: 飽和閾値。-1 のとき maximum/data_maximum を自動使用。

        Returns:
            numpy.ndarray: 処理後の新しい (H, W, 3) float32 配列

        Raises:
            RuntimeError: RAW ファイルが読み込まれていない場合
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded. Call load_file() first.")
        arr = np.ascontiguousarray(image, dtype=np.float32)
        return self._wrapper.recover_highlights(arr, threshold)

    def tone_mapping(self,
                     image: np.ndarray,
                     after_scale: float = 1.0) -> np.ndarray:
        """
        トーンマッピングを実行し、新しい numpy 配列を返す。

        Args:
            image: 入力画像 (H, W, 3) float32 numpy 配列, 値域 0.0-1.0
            after_scale: 処理後のスケール係数（デフォルト 1.0）

        Returns:
            numpy.ndarray: 処理後の新しい (H, W, 3) float32 配列

        Raises:
            RuntimeError: RAW ファイルが読み込まれていない場合
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded. Call load_file() first.")
        arr = np.ascontiguousarray(image, dtype=np.float32)
        return self._wrapper.tone_mapping(arr, after_scale)

    def enhance_micro_contrast(self,
                               image: np.ndarray,
                               threshold: float = -1.0,
                               strength: float = 8.0,
                               target_contrast: float = 0.06) -> np.ndarray:
        """
        マイクロコントラスト強調を実行し、新しい numpy 配列を返す。

        Args:
            image: 入力画像 (H, W, 3) float32 numpy 配列, 値域 0.0-1.0
            threshold: 処理対象の閾値。-1 のとき maximum/data_maximum を自動使用。
            strength: 強調の強さ（デフォルト 8.0）
            target_contrast: 目標コントラスト値（デフォルト 0.06）

        Returns:
            numpy.ndarray: 処理後の新しい (H, W, 3) float32 配列

        Raises:
            RuntimeError: RAW ファイルが読み込まれていない場合
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded. Call load_file() first.")
        arr = np.ascontiguousarray(image, dtype=np.float32)
        return self._wrapper.enhance_micro_contrast(arr, threshold, strength,
                                                   target_contrast)

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
    def is_xtrans(self) -> bool:
        """
        画像がX-Transセンサーか判定
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded")
        return self.sizes.is_xtrans
        
    @property
    def is_bayer(self) -> bool:
        """
        画像がBayer配列か判定 (X-Transでない場合はBayerとみなす)
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded")
        return not self.sizes.is_xtrans
    
    @property
    def color_desc(self) -> bytes:
        """
        RAW画像のBayer配列の色配置パターン記述子を取得します。
        例: b'RGBG'
        """
        if not self._is_loaded:
            raise RuntimeError("No RAW file loaded")
        # pybind11経由で抽出された文字列(str)を利用してbytesに変換（上位互換のため）
        return self.sizes.color_desc.encode('ascii')
        
    def get_bayer_pattern_offset(self, img_pre: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        DemosaicNet等で必要となる固定レイアウト(GRBG)にアラインメントするための
        (offset_y, offset_x) を計算して返します。
        
        Bayer配列画像に対して機能します。X-Trans画像の場合は (0, 0) を返します。
        内部で前処理画像(preprocess=True)の赤チャンネルの位相を判定します。
        
        Args:
            img_pre (numpy.ndarray, optional): 既に `postprocess(preprocess=True)` で
                取得済みの画像があれば渡すことで再計算を防げます。
                
        Returns:
            Tuple[int, int]: numpy.roll(img, shift=(offset_y, offset_x), axis=(0, 1)) に使用するシフト量
        """
        if self.is_xtrans:
            return (0, 0)
            
        try:
            # 前処理画像が渡されていなければ取得する
            if img_pre is None:
                img_pre = self.postprocess(preprocess=True, no_auto_scale=True, use_camera_wb=False)
            
            # preprocess=True出力は (H, W, 3) のスパースBayerデータ
            # 各位置はRGBいずれか1チャンネルのみ非ゼロ
            # Rチャンネルの2x2グリッド上での非ゼロ位置を特定する
            r_ch = img_pre[..., 0].astype(np.float64)
            
            # 2x2位相ごとのR値の合計を計算する
            # (0,0): 偶数行・偶数列, (0,1): 偶数行・奇数列
            # (1,0): 奇数行・偶数列, (1,1): 奇数行・奇数列
            sums = {
                (0, 0): r_ch[0::2, 0::2].sum(),  # RGGB layout
                (0, 1): r_ch[0::2, 1::2].sum(),  # GRBG layout
                (1, 0): r_ch[1::2, 0::2].sum(),  # GBRG layout
                (1, 1): r_ch[1::2, 1::2].sum(),  # BGGR layout
            }
            
            # 最も赤チャンネルの値の合計が大きい位相をRの2D空間位置とみなす
            # NOTE: color_desc文字列はLibRaw内部の順序であり、2D空間レイアウトと一致しない場合がある
            # このため実際のピクセルデータから位相を検出する
            ry, rx = max(sums, key=sums.get)
            
            # アラインメント要件: DemosaicNetは R が (0, 1) の位置にあることを期待 (GRBG)
            # DemosaicNet bayer_mosaic: mask[0, 0::2, 1::2] = 1  # Red at (even row, odd col)
            # 必要なシフト量 = DemosaicNet期待位置 - 現在位置 (mod 2)
            dy = (0 - ry) % 2
            dx = (1 - rx) % 2
            
            return (dy, dx)
            
        except Exception as e:
            # Fallback for unexpected errors
            import warnings
            warnings.warn(f"Failed to calculate bayer offset: {e}")
            return (0, 0)

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
    def color_matrix(self) -> Optional[np.ndarray]:
        """
        Color transformation matrix (3x4) used to convert camera color space to output color space.
        Available only after `postprocess()` is called.
        
        Returns:
            numpy.ndarray: 3x4 layout matrix, or None if `postprocess()` has not been called.
        """
        return self._color_matrix
        
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
        from . import is_available, is_apple_selicon
        
        return {
            'accelerate_available': True,  # Accelerate framework is always available on macOS
            'apple_silicon': is_apple_selicon(),
            'available': is_available(),
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