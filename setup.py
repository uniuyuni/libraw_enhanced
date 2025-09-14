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
import warnings


def is_apple_platform():
    """Apple プラットフォームの検出"""
    return platform.system() == "Darwin"

def is_apple_silicon():
    """Apple Silicon の検出"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_brew_libomp_paths():
    """Homebrewでインストールしたlibompのパスを取得"""
    try:
        # brew --prefix libomp でパスを取得
        brew_prefix = subprocess.check_output(
            ['brew', '--prefix', 'libomp'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        
        include_path = os.path.join(brew_prefix, 'include')
        lib_path = os.path.join(brew_prefix, 'lib')
        
        if os.path.exists(include_path) and os.path.exists(lib_path):
            print(f"Found libomp via Homebrew:")
            print(f"  Include path: {include_path}")
            print(f"  Library path: {lib_path}")
            return include_path, lib_path
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    warnings.warn("libomp not found via Homebrew. OpenMP support may be limited.")
    return None, None

def find_libraw():
    """LibRaw ライブラリの検出"""
    include_dirs = []
    library_dirs = []
    libraries = ['raw']
    
    # pkg-config による検出
    try:
        pkg_config = shutil.which('pkg-config')
        if pkg_config:
            # インクルードディレクトリ
            include_output = subprocess.check_output(
                [pkg_config, '--cflags-only-I', 'libraw'], 
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            if include_output:
                include_dirs.extend([
                    path[2:] for path in include_output.split() 
                    if path.startswith('-I')
                ])
            
            # ライブラリディレクトリ
            libs_output = subprocess.check_output(
                [pkg_config, '--libs-only-L', 'libraw'], 
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            if libs_output:
                library_dirs.extend([
                    path[2:] for path in libs_output.split() 
                    if path.startswith('-L')
                ])
            
            print(f"Found LibRaw via pkg-config:")
            print(f"  Include dirs: {include_dirs}")
            print(f"  Library dirs: {library_dirs}")
            return include_dirs, library_dirs, libraries
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Homebrew による検出
    if is_apple_platform():
        if is_apple_silicon():
            homebrew_prefix = "/opt/homebrew"
        else:
            homebrew_prefix = "/usr/local"
        
        libraw_include = os.path.join(homebrew_prefix, "include")
        libraw_lib = os.path.join(homebrew_prefix, "lib")
        
        if (os.path.exists(os.path.join(libraw_include, "libraw", "libraw.h")) and
            os.path.exists(os.path.join(libraw_lib, "libraw.dylib")) or 
            os.path.exists(os.path.join(libraw_lib, "libraw.a"))):
            include_dirs = [libraw_include]
            library_dirs = [libraw_lib]
            
            print(f"Found LibRaw via Homebrew:")
            print(f"  Include dirs: {include_dirs}")
            print(f"  Library dirs: {library_dirs}")
            return include_dirs, library_dirs, libraries
    
    # システムディレクトリでの検索
    system_include_paths = ['/usr/include', '/usr/local/include']
    system_lib_paths = ['/usr/lib', '/usr/local/lib']
    
    for include_path in system_include_paths:
        if os.path.exists(os.path.join(include_path, "libraw", "libraw.h")):
            include_dirs = [include_path]
            break
    
    for lib_path in system_lib_paths:
        if (os.path.exists(os.path.join(lib_path, "libraw.so")) or
            os.path.exists(os.path.join(lib_path, "libraw.a")) or
            os.path.exists(os.path.join(lib_path, "libraw.dylib"))):
            library_dirs = [lib_path]
            break
    
    if include_dirs:
        print(f"Found LibRaw in system directories:")
        print(f"  Include dirs: {include_dirs}")
        print(f"  Library dirs: {library_dirs}")
        return include_dirs, library_dirs, libraries
    
    raise RuntimeError(
        "LibRaw not found. Please install LibRaw:\n"
        "  macOS: brew install libraw\n"
        "  Ubuntu/Debian: sudo apt-get install libraw-dev\n"
        "  CentOS/RHEL: sudo yum install LibRaw-devel"
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
                        '-Xpreprocessor', '-fopenmp', '-lomp',  # OpenMPサポート
                        '-I"$(brew --prefix libomp)/include"', '-L"$(brew --prefix libomp)/lib"',

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
    libomp_include, libomp_lib = get_brew_libomp_paths()
    
    # ソースファイル
    sources = [
        'core/python_bindings.cpp',
        'core/libraw_wrapper.cpp',
        'core/camera_matrices.cpp',
        'external/LibRaw-0.21.4/src/metadata/identify.cpp',
    ]
    
    # Apple プラットフォームでのみMetal統合ファイルを追加
    if is_apple_platform():
        sources.append('core/accelerator.cpp')    # Unified GPU/CPU accelerator interface
        sources.append('core/cpu_accelerator.cpp')      # CPU Accelerate framework
        sources.append('core/gpu_accelerator.mm')       # True GPU Metal acceleration (Objective-C++)
        print("Added Metal acceleration support: CPU + GPU unified with Metal shaders")
    else:
        print("Skipping Metal accelerator on non-Apple platform")
    
    # インクルードディレクトリ
    include_dirs = [
        'core',
        'external/LibRaw-0.21.4',        # LibRaw internal headers
        'external/LibRaw-0.21.4/internal', # LibRaw internal defs
        *libraw_includes,
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
        '-DVERSION_INFO="dev"',  # エスケープ修正
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
        ('VERSION_INFO', '"dev"'),  # 引用符修正
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