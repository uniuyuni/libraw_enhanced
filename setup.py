#!/usr/bin/env python3
"""
LibRaw Enhanced - Setup script with Metal support

Objective-C++サポートとApple Siliconフレームワーク統合を含む
クロスプラットフォーム対応ビルドスクリプト
"""

import os
import sys
import platform
import subprocess
import sysconfig
import shutil
import warnings
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from distutils.unixccompiler import UnixCCompiler


def is_apple_platform():
    """Apple プラットフォームの検出"""
    return platform.system() == "Darwin"

def is_apple_silicon():
    """Apple Silicon の検出"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_libomp_paths():
    """OpenMP（llvm-openmp）。pixi / conda 環境を優先し、無ければ Homebrew。"""
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("PREFIX", "").strip()
    if conda:
        include_path = os.path.join(conda, "include")
        lib_path = os.path.join(conda, "lib")
        omp_h = os.path.join(include_path, "omp.h")
        if os.path.isfile(omp_h) and os.path.isdir(lib_path):
            print(f"Found libomp from conda/pixi: {conda}")
            return include_path, lib_path
    try:
        brew_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        include_path = os.path.join(brew_prefix, "include")
        lib_path = os.path.join(brew_prefix, "lib")
        if os.path.exists(include_path) and os.path.exists(lib_path):
            print(f"Found libomp via Homebrew: {brew_prefix}")
            return include_path, lib_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    warnings.warn("libomp not found (pixi の llvm-openmp か brew install libomp)。OpenMP が無効になる場合があります。")
    return None, None

def _default_libraw_install_prefix():
    """platypus/third_party/libraw-install（setup.py は libraw_enhanced/ 直下）"""
    return Path(__file__).resolve().parent.parent / "third_party" / "libraw-install"


def find_libraw():
    """LibRaw は pixi でビルドした third_party/libraw-install のみ（システム・Homebrew は使わない）。"""
    libraries = ["raw"]
    candidates = []
    env_p = os.environ.get("LIBRAW_LOCAL_PREFIX", "").strip()
    if env_p:
        candidates.append(Path(env_p))
    candidates.append(_default_libraw_install_prefix())

    for prefix in candidates:
        inc = prefix / "include"
        lib = prefix / "lib"
        pc = lib / "pkgconfig" / "libraw.pc"
        has_pc = pc.is_file()
        has_headers = (inc / "libraw" / "libraw.h").is_file()
        has_lib = (
            (lib / "libraw.dylib").is_file()
            or (lib / "libraw.so").is_file()
            or (lib / "libraw.a").is_file()
        )
        if (has_pc or has_headers) and has_lib:
            print(f"Found LibRaw (project-local): {prefix}")
            return [str(inc)], [str(lib)], libraries

    raise RuntimeError(
        "LibRaw が見つかりません（/usr/local や Homebrew は参照しません）。\n"
        "  pixi run build-libraw\n"
        "  pixi run install-libraw-enhanced\n"
        "を実行して third_party/libraw-install を作成し、"
        "pixi shell（activation で LIBRAW_LOCAL_PREFIX が載る）でビルドしてください。"
    )


def get_apple_frameworks():
    """Apple固有フレームワークの取得"""
    if not is_apple_platform():
        return [], []
    
    frameworks = []
    framework_dirs = []
    
    # 必須フレームワーク
    required_frameworks = ['Foundation']
    
    # Metal関連フレームワーク（Apple Siliconで利用可能）
    metal_frameworks = ['Metal', 'MetalPerformanceShaders', 'Accelerate', 'QuartzCore']
    
    # フレームワークの存在確認
    for framework in required_frameworks + metal_frameworks:
        framework_path = f"/System/Library/Frameworks/{framework}.framework"
        if os.path.exists(framework_path):
            frameworks.append(framework)
            if framework in metal_frameworks:
                print(f"Found Metal framework: {framework}")
        else:
            if framework in required_frameworks:
                warnings.warn(f"Required framework not found: {framework}")
            else:
                print(f"Optional framework not available: {framework}")
    
    return frameworks, framework_dirs


class CustomBuildExt(build_ext):
    """カスタムビルド拡張クラス（Objective-C++サポート付き）"""
    
    def initialize_options(self):
        """初期化オプション - Objective-C++サポートを追加"""
        
        # .mm ファイルサポートを追加
        if hasattr(UnixCCompiler, 'src_extensions') and '.mm' not in UnixCCompiler.src_extensions:
            UnixCCompiler.src_extensions.append('.mm')
        
        if hasattr(UnixCCompiler, 'language_map'):
            UnixCCompiler.language_map['.mm'] = 'objc'
        
        # 親クラスの初期化
        super().initialize_options()
        
        # Apple プラットフォームでObjective-C++コンパイラのパッチ適用
        if is_apple_platform():
            unpatched_compile = UnixCCompiler._compile
            
            def patched_compile(compiler_self, obj, src, ext, cc_args, extra_postargs, pp_opts):
                """Objective-C++ファイル用のコンパイルフラグを追加"""
                if ext == ".cpp":
                    patched_postargs = extra_postargs + ["-std=c++17"]
                elif ext == ".mm":
                    patched_postargs = extra_postargs + [
                        "-x", "objective-c++",  # Objective-C++として強制
                        "-fobjc-arc",          # ARC有効
                        "-fobjc-weak",         # weak参照サポート
                        "-std=c++17",          # C++17標準
                        "-D__OBJC__=1",        # __OBJC__マクロ定義
                    ]
                else:
                    patched_postargs = extra_postargs
                    
                unpatched_compile(compiler_self, obj, src, ext, cc_args, patched_postargs, pp_opts)
            
            UnixCCompiler._compile = patched_compile
    
    def build_extensions(self):
        """拡張モジュールのビルド"""
        
        # Apple プラットフォーム用の最適化
        if is_apple_platform():
            for ext in self.extensions:
                # Apple Silicon最適化フラグ
                if is_apple_silicon():
                    # Universal Binaryやめてapple silicon専用に
                    ext.extra_compile_args = [arg for arg in ext.extra_compile_args if not (arg == '-arch' or arg == 'x86_64')]
                    ext.extra_compile_args.extend([
                        '-arch', 'arm64',      # Apple Silicon専用
                        '-mcpu=apple-a14',     # Apple Silicon最適化
                        '-O3',                 # 最高レベル最適化
                        '-DAPPLE_SILICON_OPTIMIZED',
                        '-D__ARM_NEON=1',      # NEON明示的有効化
                        '-D__ARM_FEATURE_FMA=1',  # FMA (Fused Multiply-Add) サポート
                        # 削除: '-D__ARM_FP=0xE' - Metal環境では不要・有害
                        # 削除: '-ffp-contract=fast' - NEON最適化を阻害する可能性
                        '-fvectorize',         # ベクトル化強制
                        '-mllvm', '-force-vector-width=4',  # NEON幅指定
                        # OpenMP は create_extension / get_libomp_paths で設定（brew 文字列は使わない）

                    ])
                
                # Metal定義
                ext.define_macros.append(('__arm64__', '1'))
                
                # Apple プラットフォーム固有のフラグ
                ext.extra_compile_args.extend([
                    '-std=c++17',
                    '-DAPPLE_PLATFORM',
                ])
                
                print("Applied Apple platform optimizations with Objective-C++ support")
        
        super().build_extensions()
    
    def get_ext_filename(self, ext_name):
        """拡張モジュールファイル名の取得"""
        filename = super().get_ext_filename(ext_name)
        
        # Apple Silicon Universalバイナリ対応
        if is_apple_platform() and is_apple_silicon():
            # ファイル名にarm64マーカーを追加（必要に応じて）
            pass
        
        return filename
    
    def compiler_type_is_mmcompiler(self, filename):
        """Objective-C++ (.mm) ファイルの判定"""
        return filename.endswith('.mm')
    
    def build_extension(self, ext):
        """個別拡張モジュールのビルド"""
        super().build_extension(ext)


def create_extension():
    """拡張モジュールの作成"""
    
    # LibRaw検出
    libraw_includes, libraw_lib_dirs, libraw_libs = find_libraw()
    
    # Apple フレームワーク検出
    apple_frameworks, apple_framework_dirs = get_apple_frameworks()
    
    # libompのパス取得
    libomp_include, libomp_lib = get_libomp_paths()
    
    # ソースファイル
    sources = [
        'core/python_bindings.cpp',
        'core/libraw_wrapper.cpp',
        'core/camera_matrices.cpp',
        'external/LibRaw-master/src/metadata/identify.cpp',
    ]
    
    # Apple プラットフォームでのみMetal統合ファイルを追加
    if is_apple_platform():
        sources.append('core/accelerator.cpp')    # Unified GPU/CPU accelerator interface
        sources.append('core/cpu_accelerator.cpp')      # CPU Accelerate framework
        sources.append('core/gpu_accelerator.mm')       # True GPU Metal acceleration (Objective-C++)
        print("Added Metal acceleration support: CPU + GPU unified with Metal shaders")
    else:
        print("Skipping Metal accelerator on non-Apple platform")
    
    # インクルードディレクトリ (libraw_includes MUST be before bundled external/LibRaw-master to avoid ABI memory corruption!)
    include_dirs = [
        'core',
        *libraw_includes,
        'external/LibRaw-master',        # LibRaw internal headers (fallback)
        'external/LibRaw-master/internal', # LibRaw internal defs (fallback)
    ]
    
    # libompのインクルードパスを追加
    if libomp_include:
        include_dirs.append(libomp_include)
    
    # ライブラリディレクトリ
    library_dirs = libraw_lib_dirs + apple_framework_dirs
    
    # libompのライブラリパスを追加
    if libomp_lib:
        library_dirs.append(libomp_lib)
    
    # ライブラリ
    libraries = libraw_libs.copy()
    
    # コンパイルフラグ
    extra_compile_args = [
        '-std=c++17',
        '-O3',
        '-DVERSION_INFO="0.7.2"',  # エスケープ修正
    ]
    
    # リンクフラグ
    extra_link_args = []
    
    # Apple固有設定
    if is_apple_platform():
        # フレームワークリンク
        for framework in apple_frameworks:
            extra_link_args.extend(['-framework', framework])
        
        # Objective-C++ compilation flags (handled by CustomBuildExt for .mm files)
        extra_compile_args.extend(['-fmodules'])
        
        # OpenMPサポートを追加
        if libomp_include and libomp_lib:
            extra_compile_args.extend(['-Xpreprocessor', '-fopenmp'])
            extra_link_args.extend(['-lomp'])
        # ローカルビルド LibRaw / libomp を実行時に解決
        for _ld in libraw_lib_dirs + ([libomp_lib] if libomp_lib else []):
            if _ld:
                extra_link_args.extend(["-Wl,-rpath," + _ld])
        
        # デバッグ情報（デバッグビルド時）- O3は維持
        if os.environ.get('DEBUG'):
            extra_compile_args.extend(['-g'])
            # O3は性能維持のため削除しない
        else:
            # Release mode での追加最適化
            extra_compile_args.extend([
                '-Ofast',           # O3より積極的な最適化
                '-flto',            # Link Time Optimization
                '-fno-math-errno',  # 数学関数の高速化
            ])
    else:
        # Linux 等: ローカル LibRaw / libomp を rpath で解決
        if libomp_include and libomp_lib:
            extra_compile_args.extend(["-fopenmp"])
            extra_link_args.extend(["-fopenmp"])
        for _ld in libraw_lib_dirs + ([libomp_lib] if libomp_lib else []):
            if _ld:
                extra_link_args.extend(["-Wl,-rpath," + _ld])
    
    # 警告レベル
    if sys.platform != 'win32':
        extra_compile_args.extend([
            '-Wall',
            '-Wextra',
            '-Wno-unused-parameter',
            '-Wno-missing-field-initializers'
        ])
    
    # プリプロセッサ定義
    define_macros = [
        ('VERSION_INFO', '"0.7.2"'),  # 引用符修正
    ]
    
    if is_apple_platform():
        define_macros.append(('APPLE_PLATFORM', '1'))
        define_macros.append(('__arm64__', '1'))
        define_macros.append(('GPU_METAL_ENABLED', '1'))  # GPU Metal実装を有効化
        if is_apple_silicon():
            define_macros.append(('__arm64__', '1'))
    
    # pybind11拡張の作成
    ext = Pybind11Extension(
        "libraw_enhanced._core",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
        cxx_std=17,
    )
    
    return ext

if __name__ == "__main__":
    print("=" * 60)
    print("LibRaw Enhanced Build Configuration")
    print("=" * 60)
    print(f"Platform: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Apple Platform: {is_apple_platform()}")
    print(f"Apple Silicon: {is_apple_silicon()}")
    print("=" * 60)


# セットアップ実行
try:
    ext_module = create_extension()
except Exception as e:
    print(f"Error creating extension: {e}")
    sys.exit(1)

setup(
    ext_modules=[ext_module],
    cmdclass={'build_ext': CustomBuildExt},
    zip_safe=False,
)