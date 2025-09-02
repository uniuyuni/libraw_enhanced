cmake_minimum_required(VERSION 3.16)
project(libraw_enhanced VERSION 1.0.0)

# デフォルトビルドタイプをReleaseに設定（パフォーマンス優先）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Build type not specified, defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# プラットフォーム検出
if(APPLE)
    set(APPLE_PLATFORM TRUE)
    
    # Apple Siliconかどうかを検出
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE MACHINE_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(MACHINE_ARCH STREQUAL "arm64")
        set(APPLE_SILICON TRUE)
        message(STATUS "Building for Apple Silicon (arm64)")
    else()
        set(APPLE_SILICON FALSE)
        message(STATUS "Building for Intel Mac (x86_64)")
    endif()
else()
    set(APPLE_PLATFORM FALSE)
    set(APPLE_SILICON FALSE)
    message(STATUS "Building for non-Apple platform")
endif()

# 最適化フラグ設定
if(APPLE_SILICON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-a14")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
    add_definitions(-DAPPLE_SILICON_OPTIMIZED)
elseif(APPLE_PLATFORM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif()

# Apple固有の設定
if(APPLE_PLATFORM)
    # Objective-C++サポートを有効化
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    
    # Apple フレームワーク検索
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    
    message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
    message(STATUS "Found MetalPerformanceShaders framework: ${METALPERFORMANCESHADERS_FRAMEWORK}")
    message(STATUS "Found Accelerate framework: ${ACCELERATE_FRAMEWORK}")
    
    set(APPLE_FRAMEWORKS
        ${METAL_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
endif()

# LibRaw検索
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBRAW libraw)
endif()

if(NOT LIBRAW_FOUND)
    # Homebrewパスを確認
    if(APPLE_PLATFORM)
        if(APPLE_SILICON)
            set(HOMEBREW_PREFIX "/opt/homebrew")
        else()
            set(HOMEBREW_PREFIX "/usr/local")
        endif()
        
        find_path(LIBRAW_INCLUDE_DIR libraw/libraw.h
            PATHS ${HOMEBREW_PREFIX}/include
        )
        
        find_library(LIBRAW_LIBRARIES raw
            PATHS ${HOMEBREW_PREFIX}/lib
        )
        
        if(LIBRAW_INCLUDE_DIR AND LIBRAW_LIBRARIES)
            set(LIBRAW_FOUND TRUE)
            message(STATUS "Found LibRaw via Homebrew: ${LIBRAW_LIBRARIES}")
        endif()
    endif()
endif()

if(NOT LIBRAW_FOUND)
    message(FATAL_ERROR "LibRaw not found. Please install via: brew install libraw")
endif()

# Python関連の依存関係
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# ソースファイル定義
set(CORE_SOURCES
    core/libraw_wrapper.cpp
    core/python_bindings.cpp
    core/camera_matrices.cpp
)

if(APPLE_PLATFORM)
    list(APPEND CORE_SOURCES core/accelerator.cpp core/cpu_accelerator.cpp core/gpu_accelerator.mm)
    
    # Objective-C++ファイルのコンパイルフラグ
    set_source_files_properties(
        core/gpu_accelerator.mm
        PROPERTIES
        COMPILE_FLAGS "-fobjc-arc -fmodules"
    )
endif()

# メインライブラリターゲット
pybind11_add_module(libraw_enhanced_core ${CORE_SOURCES})

# インクルードディレクトリ
target_include_directories(libraw_enhanced_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${LIBRAW_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

# ライブラリリンク
target_link_libraries(libraw_enhanced_core PRIVATE
    ${LIBRAW_LIBRARIES}
    ${Python_LIBRARIES}
)

if(APPLE_PLATFORM)
    target_link_libraries(libraw_enhanced_core PRIVATE
        ${APPLE_FRAMEWORKS}
    )
    
    # Metal関連の定義を追加
    target_compile_definitions(libraw_enhanced_core PRIVATE
        METAL_ACCELERATION_AVAILABLE
    )
endif()

# コンパイル設定
target_compile_definitions(libraw_enhanced_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(libraw_enhanced_core PRIVATE DEBUG_MODE)
    if(APPLE_PLATFORM)
        set_source_files_properties(
            core/gpu_accelerator.mm
            PROPERTIES
            COMPILE_FLAGS "-fobjc-arc -fmodules -DDEBUG"
        )
    endif()
endif()

# 警告レベル設定
if(MSVC)
    target_compile_options(libraw_enhanced_core PRIVATE /W4)
else()
    target_compile_options(libraw_enhanced_core PRIVATE -Wall -Wextra -Wpedantic)
    if(APPLE_PLATFORM)
        # Objective-C++特有の警告を抑制
        target_compile_options(libraw_enhanced_core PRIVATE 
            -Wno-unused-parameter 
            -Wno-missing-field-initializers
        )
    endif()
endif()

# インストール設定
install(TARGETS libraw_enhanced_core
    LIBRARY DESTINATION .
    ARCHIVE DESTINATION .
)

# テスト設定
enable_testing()

# Python用テストの検出
find_program(PYTEST_EXECUTABLE pytest)
if(PYTEST_EXECUTABLE)
    add_test(
        NAME python_tests
        COMMAND ${PYTEST_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    set_tests_properties(python_tests PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}"
    )
else()
    message(WARNING "pytest not found, Python tests will not be available")
endif()

# パフォーマンステスト用の特別ターゲット
if(APPLE_PLATFORM)
    add_test(
        NAME metal_performance_test
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_metal_performance.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    set_tests_properties(metal_performance_test PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}"
        TIMEOUT 300
    )
endif()

# デバッグ情報表示
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Apple Platform: ${APPLE_PLATFORM}")
message(STATUS "Apple Silicon: ${APPLE_SILICON}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(APPLE_PLATFORM)
    message(STATUS "Objective-C++ Standard: ${CMAKE_OBJCXX_STANDARD}")
    message(STATUS "Metal Framework: ${METAL_FRAMEWORK}")
endif()
message(STATUS "LibRaw Include: ${LIBRAW_INCLUDE_DIR}")
message(STATUS "LibRaw Library: ${LIBRAW_LIBRARIES}")
message(STATUS "Python Version: ${Python_VERSION}")
message(STATUS "Python Include: ${Python_INCLUDE_DIRS}")
message(STATUS "======================================")

# 追加の開発者向けターゲット
if(APPLE_PLATFORM)
    # Metal Shader検証ターゲット
    add_custom_target(validate_metal_shaders
        COMMAND xcrun -sdk macosx metal -c -o /dev/null -Wno-unused-variable
                ${CMAKE_CURRENT_SOURCE_DIR}/core/shaders/*.metal || true
        COMMENT "Validating Metal shaders"
        VERBATIM
    )
    
    # Apple Silicon固有の最適化レポート
    if(APPLE_SILICON)
        add_custom_target(optimization_report
            COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} -S -mllvm -print-after-all
                    ${CMAKE_CURRENT_SOURCE_DIR}/core/libraw_wrapper.cpp -o /dev/null 2>&1 | 
                    grep -A5 -B5 "optimization" || true
            COMMENT "Generating optimization report"
            VERBATIM
        )
    endif()
endif()

# CPack設定（パッケージング用）
set(CPACK_PACKAGE_NAME "libraw-enhanced")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance LibRaw Python wrapper with Metal acceleration")
set(CPACK_PACKAGE_VENDOR "LibRaw Enhanced Project")

if(APPLE_PLATFORM)
    set(CPACK_GENERATOR "productbuild")
    set(CPACK_PRODUCTBUILD_IDENTITY_NAME "Developer ID Installer")
endif()

include(CPack)