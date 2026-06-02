import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW_FIXTURE = ROOT / "tests" / "fixtures" / "temp" / "GR IIIx Plant.DNG"


def test_plain_import_makes_core_available():
    code = (
        "import libraw_enhanced as lre; "
        "info = lre.get_platform_info(); "
        "assert info['core_available'], info"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_cmake_configures_from_source_tree(tmp_path):
    result = subprocess.run(
        ["cmake", "-S", str(ROOT), "-B", str(tmp_path / "cmake-build")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_constants_all_exports_exist():
    import libraw_enhanced.constants as constants

    missing = [name for name in constants.__all__ if not hasattr(constants, name)]
    assert not missing


def test_highlight_mode_enhanced_names_match_pipeline_thresholds():
    import libraw_enhanced as lre

    assert int(lre.HighlightMode.RebuildAndMicroContrast) == 4
    assert int(lre.HighlightMode.RebuildAndDetailToneMap) == 5
    assert int(lre.HighlightMode.RebuildAndToneMap) == 6


def test_optimization_info_property_is_usable():
    from libraw_enhanced.high_level_api import RawImage

    raw = object.__new__(RawImage)
    info = raw.optimization_info

    assert set(info) == {"accelerate_available", "apple_silicon", "available"}


def test_imread_buffer_loads_valid_bytes():
    import libraw_enhanced as lre

    if not RAW_FIXTURE.exists():
        pytest.skip(f"fixture missing: {RAW_FIXTURE}")

    with lre.imread_buffer(RAW_FIXTURE.read_bytes()) as raw:
        assert raw.sizes.raw_width > 0
        assert raw.sizes.raw_height > 0
        assert raw.camera_info["filepath"] == "<buffer>"


def test_imread_buffer_accepts_memoryview_without_list_conversion():
    import libraw_enhanced as lre

    if not RAW_FIXTURE.exists():
        pytest.skip(f"fixture missing: {RAW_FIXTURE}")

    data = memoryview(bytearray(RAW_FIXTURE.read_bytes()))
    with lre.imread_buffer(data) as raw:
        assert raw.sizes.raw_width > 0


def test_imread_buffer_rejects_invalid_bytes():
    import libraw_enhanced as lre

    with pytest.raises(RuntimeError):
        lre.imread_buffer(b"not a raw file")


def test_raw_output_color_does_not_introduce_negative_matrix_values():
    import libraw_enhanced as lre

    if not RAW_FIXTURE.exists():
        pytest.skip(f"fixture missing: {RAW_FIXTURE}")

    with lre.imread(str(RAW_FIXTURE)) as raw:
        image = raw.postprocess(
            half_size=True,
            output_bps=32,
            demosaic_algorithm=lre.DemosaicAlgorithm.Linear,
            output_color=lre.ColorSpace.Raw,
            gamma=(1.0, 1.0),
            no_auto_scale=True,
            use_camera_wb=False,
        )

    assert image.dtype == np.float32
    assert float(np.nanmin(image)) >= -1e-6


def test_repeated_postprocess_rebuilds_from_original_raw_data():
    import libraw_enhanced as lre

    if not RAW_FIXTURE.exists():
        pytest.skip(f"fixture missing: {RAW_FIXTURE}")

    kwargs = dict(
        half_size=True,
        output_bps=32,
        demosaic_algorithm=lre.DemosaicAlgorithm.Linear,
        output_color=lre.ColorSpace.Raw,
        gamma=(1.0, 1.0),
        no_auto_scale=True,
        use_camera_wb=False,
    )

    with lre.imread(str(RAW_FIXTURE)) as raw:
        first = raw.postprocess(**kwargs)
        second = raw.postprocess(**kwargs)

    np.testing.assert_allclose(second, first, rtol=0, atol=0)


def test_micro_contrast_noops_when_local_contrast_is_zero():
    import libraw_enhanced as lre

    wrapper = lre._core.LibRawWrapper()
    image = np.ones((16, 16, 3), dtype=np.float32) * 1.5

    try:
        output = wrapper.enhance_micro_contrast(
            image, threshold=0.0, strength=8.0, target_contrast=0.04
        )
    finally:
        wrapper.close()

    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output, image)


def test_micro_contrast_noops_when_target_contrast_is_zero():
    import libraw_enhanced as lre

    wrapper = lre._core.LibRawWrapper()
    image = np.linspace(0.0, 2.0, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)

    try:
        output = wrapper.enhance_micro_contrast(
            image, threshold=0.0, strength=8.0, target_contrast=0.0
        )
    finally:
        wrapper.close()

    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output, image)


def test_micro_contrast_preserves_superwhite_float_values():
    import libraw_enhanced as lre

    wrapper = lre._core.LibRawWrapper()
    image = np.ones((64, 64, 3), dtype=np.float32)
    image[0:8, 0:8, :] = 0.0
    image[0:8, 8:16, :] = 2.0
    image[32, 32, :] = 1.01

    try:
        output = wrapper.enhance_micro_contrast(
            image, threshold=0.0, strength=8.0, target_contrast=0.04
        )
    finally:
        wrapper.close()

    assert np.isfinite(output).all()
    assert float(output[32, 32].max()) > 1.0


def test_cpu_color_space_conversion_math_is_in_place_safe():
    pixel = np.array([0.25, 0.50, 0.75], dtype=np.float32)
    transform = np.array(
        [
            [1.0, 2.0, 3.0, 0.10],
            [4.0, 5.0, 6.0, 0.20],
            [7.0, 8.0, 9.0, 0.30],
        ],
        dtype=np.float32,
    )

    expected = transform[:, :3] @ pixel + transform[:, 3]

    aliased = pixel.copy()
    in_ref = aliased.copy()
    out_ref = aliased
    out_ref[0] = transform[0, 0] * in_ref[0] + transform[0, 1] * in_ref[1] + transform[0, 2] * in_ref[2] + transform[0, 3]
    out_ref[1] = transform[1, 0] * in_ref[0] + transform[1, 1] * in_ref[1] + transform[1, 2] * in_ref[2] + transform[1, 3]
    out_ref[2] = transform[2, 0] * in_ref[0] + transform[2, 1] * in_ref[1] + transform[2, 2] * in_ref[2] + transform[2, 3]

    np.testing.assert_allclose(aliased, expected, rtol=1e-6, atol=1e-6)
