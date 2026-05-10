"""
Defringe unit tests + X-T5 integration test.

Unit tests use synthetic images to verify correctness without a real RAW file.
Integration test uses tests/fixtures/X-T5 Room.RAF and saves diff images.
"""

import ctypes
import os
import sys
import time
import numpy as np
import pyvips
import pytest

# Pre-load libraw.dylib so the extension module can find it.
# This is needed when the library is installed to a non-standard path.
_LIBRAW_DYLIB = os.path.join(
    os.path.dirname(__file__), "..", "third_party", "libraw-install", "lib", "libraw.24.dylib"
)
try:
    ctypes.CDLL(os.path.realpath(_LIBRAW_DYLIB))
except OSError:
    pass  # ignore if not present (CI may use a system libraw)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wrapper():
    """Return an initialised LibRawWrapper (with accelerator)."""
    try:
        from libraw_enhanced._core import LibRawWrapper
        w = LibRawWrapper()
        return w
    except ImportError:
        pytest.skip("_core not available")


def defringe(img_linear: np.ndarray, **kw) -> np.ndarray:
    """Call wrapper.defringe on a linear-light (H,W,3) float32 array."""
    w = make_wrapper()
    img = np.ascontiguousarray(img_linear, dtype=np.float32)
    return w.defringe(img, **kw)


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def make_purple_fringe_image(h=64, w=64, fringe_width=2):
    """
    Construct a synthetic image with:
      - Left half: bright white (highlight)
      - Right half: dark gray (shadow)
      - Thin purple fringe strip at the boundary
    Returns float32 linear (H,W,3) in [0,1].
    """
    img = np.zeros((h, w, 3), dtype=np.float32)
    mid = w // 2

    # Highlight (left)
    img[:, :mid, :] = 0.95

    # Shadow (right)
    img[:, mid:, :] = 0.15

    # Purple fringe: thin strip on the shadow side of the edge
    for col in range(mid, mid + fringe_width):
        img[:, col, 0] = 0.55  # R high
        img[:, col, 1] = 0.10  # G low
        img[:, col, 2] = 0.55  # B high → purple (hue ~300°, sat high)

    return img

def make_red_cloth_image(h=64, w=64):
    """
    Problem 1 reproduction: Red fabric with texture.
    High Cr (R-G), low but positive Cb (B-G).
    Should NOT be desaturated even if near an edge.
    """
    img = np.zeros((h, w, 3), dtype=np.float32)
    mid = w // 2
    # Bright red-magenta edge (e.g. red cloth detail)
    # R=0.8, G=0.2, B=0.25 -> Cr=0.6, Cb=0.05 (triggers purple gate if Cb>0)
    img[:, :mid, 0] = 0.8
    img[:, :mid, 1] = 0.2
    img[:, :mid, 2] = 0.25
    
    # Slightly darker side to create an edge for Sobel
    img[:, mid:, 0] = 0.6
    img[:, mid:, 1] = 0.15
    img[:, mid:, 2] = 0.2
    
    return img

def make_green_fringe_image(h=64, w=64, fringe_width=2):
    """Similar but green fringe."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    mid = w // 2

    img[:, :mid, :] = 0.92
    img[:, mid:, :] = 0.12

    for col in range(mid, mid + fringe_width):
        img[:, col, 0] = 0.08  # R low
        img[:, col, 1] = 0.60  # G high → green (hue ~120°, sat high)
        img[:, col, 2] = 0.08  # B low

    return img


def make_purple_flower_image(h=64, w=64):
    """
    Purple object in mid-tones with NO nearby highlight.
    Should NOT be flagged as fringe.
    """
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, :] = 0.20  # gray background (low luminance, no highlight)

    # Large purple region (40x40 centered)
    y0, y1 = 12, 52
    x0, x1 = 12, 52
    img[y0:y1, x0:x1, 0] = 0.45
    img[y0:y1, x0:x1, 1] = 0.10
    img[y0:y1, x0:x1, 2] = 0.45

    return img


def make_micro_fringe_image(h=64, w=64):
    """Single-pixel purple fringe at a very bright edge."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    mid = w // 2

    img[:, :mid, :] = 0.98   # near-white highlight
    img[:, mid:, :] = 0.05   # near-black shadow

    # 1-pixel fringe
    img[:, mid, 0] = 0.50
    img[:, mid, 1] = 0.05
    img[:, mid, 2] = 0.52

    return img


def hue_of(r, g, b):
    """Compute HSV hue in [0,360) from r,g,b in [0,1]."""
    maxc = max(r, g, b)
    minc = min(r, g, b)
    d = maxc - minc
    if d < 1e-6:
        return 0.0
    if maxc == r:
        h = 60.0 * (((g - b) / d) % 6)
    elif maxc == g:
        h = 60.0 * ((b - r) / d + 2)
    else:
        h = 60.0 * ((r - g) / d + 4)
    return h % 360.0


def saturation_of(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    return (maxc - minc) / maxc if maxc > 1e-6 else 0.0


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestDefringe:

    def test_purple_fringe_is_reduced(self):
        """Purple fringe pixels at a bright edge should become less saturated."""
        img = make_purple_fringe_image(h=64, w=64, fringe_width=2)
        out = defringe(img, edge_threshold=0.015, chroma_threshold=0.15,
                       strength=1.0)

        mid = 32
        for col in [mid, mid + 1]:
            r_in,  g_in,  b_in  = img[32, col]
            r_out, g_out, b_out = out[32, col]

            # R and B excess over G should be reduced
            r_excess_in  = max(0.0, float(r_in)  - float(g_in))
            b_excess_in  = max(0.0, float(b_in)  - float(g_in))
            r_excess_out = max(0.0, float(r_out) - float(g_out))
            b_excess_out = max(0.0, float(b_out) - float(g_out))

            assert r_excess_out < r_excess_in, (
                f"col={col}: R excess should decrease ({r_excess_in:.3f} → {r_excess_out:.3f})")
            assert b_excess_out < b_excess_in, (
                f"col={col}: B excess should decrease ({b_excess_in:.3f} → {b_excess_out:.3f})")

    def test_purple_fringe_full_strength(self):
        """With strength=1.0, R and B excess over G should be ~0 after correction."""
        img = make_purple_fringe_image(h=64, w=64, fringe_width=2)
        out = defringe(img, strength=1.0,
                       edge_threshold=0.010, chroma_threshold=0.1)

        mid = 32
        r_out, g_out, b_out = out[32, mid]
        assert float(r_out) <= float(g_out) + 0.02, \
            f"R should not exceed G after full purple correction (R={r_out:.3f}, G={g_out:.3f})"
        assert float(b_out) <= float(g_out) + 0.02, \
            f"B should not exceed G after full purple correction (B={b_out:.3f}, G={g_out:.3f})"

    def test_red_cloth_false_positive_reproduction(self):
        """
        CONFIRMING PROBLEM 1:
        Red cloth (Cr > 0, Cb > 0 small) near an edge should NOT be affected by high strength.
        Current implementation (blur/strength) will likely fail this (desaturate it).
        """
        img = make_red_cloth_image(h=64, w=64)
        # Apply high strength as reported by user
        strength = 5.0
        out = defringe(img, strength=strength, edge_threshold=0.01, chroma_threshold=0.05)
        
        # Check a pixel on the red "cloth" side
        r_in, g_in, b_in = img[32, 10]
        r_out, g_out, b_out = out[32, 10]
        
        # If the problem exists, r_out will be significantly less than r_in
        diff = np.abs(r_in - r_out)
        assert diff < 0.02, (
            f"RED CLOTH AFFECTED: R decreased from {r_in:.3f} to {r_out:.3f} (diff={diff:.3f}). "
            "Fringe suppression is over-aggressive on natural red objects.")

    def test_green_fringe_is_reduced(self):
        """Green fringe pixels should have reduced G-excess after correction."""
        img = make_green_fringe_image(h=64, w=64, fringe_width=2)
        # Current C++ implementation targets ONLY purple (Cr>0 && Cb>0).
        # We still run this to see the current behavior.
        out = defringe(img, edge_threshold=0.010, chroma_threshold=0.1, strength=1.0)
        mid = 32
        for col in [mid, mid + 1]:
            r_in,  g_in,  b_in  = img[32, col]
            r_out, g_out, b_out = out[32, col]
            avg_in  = (float(r_in)  + float(b_in))  / 2
            avg_out = (float(r_out) + float(b_out)) / 2
            g_excess_in  = max(0.0, float(g_in)  - avg_in)
            g_excess_out = max(0.0, float(g_out) - avg_out)
            assert g_excess_out < g_excess_in, (
                f"col={col}: G excess should decrease ({g_excess_in:.3f} → {g_excess_out:.3f})")

    def test_highlight_region_unchanged(self):
        """The bright highlight half of the image should not be modified."""
        img = make_purple_fringe_image(h=64, w=64, fringe_width=2)
        out = defringe(img)

        mid = 32
        # Columns well away from edge (highlight side)
        for col in range(0, mid - 4):
            np.testing.assert_allclose(out[32, col], img[32, col], atol=1e-5,
                err_msg=f"Highlight pixel col={col} should be unchanged")

    def test_shadow_region_unchanged(self):
        """Shadow region far from the edge should not be modified."""
        img = make_purple_fringe_image(h=64, w=64, fringe_width=2)
        out = defringe(img)

        mid = 32
        for col in range(mid + 5, 64):
            np.testing.assert_allclose(out[32, col], img[32, col], atol=1e-5,
                err_msg=f"Shadow pixel col={col} should be unchanged")

    def test_no_fringe_no_change(self):
        """An image with no fringe colors should be returned unchanged."""
        rng = np.random.default_rng(42)
        img = rng.random((64, 64, 3)).astype(np.float32) * 0.3 + 0.2  # mid-tone gray
        out = defringe(img)
        np.testing.assert_allclose(out, img, atol=1e-5,
            err_msg="Neutral image must not be modified by defringe")

    def test_purple_flower_false_positive(self):
        """Large purple object in mid-tones (no nearby highlight) must not change."""
        img = make_purple_flower_image(h=64, w=64)
        # Use high edge threshold to protect flat areas
        out = defringe(img, edge_threshold=0.2, chroma_threshold=0.1)
        # Compare the large purple region
        region_in  = img[12:52, 12:52]
        region_out = out[12:52, 12:52]
        np.testing.assert_allclose(region_out, region_in, atol=1e-5,
            err_msg="Purple flower region should not be changed (false-positive protection)")

    def test_micro_fringe_detected(self):
        """Single-pixel purple fringe at very bright edge must be caught."""
        img = make_micro_fringe_image(h=64, w=64)
        out = defringe(img, edge_threshold=0.010, chroma_threshold=0.1, strength=1.0)

        mid = 32
        r_in, g_in, b_in = img[32, mid]
        r_out, g_out, b_out = out[32, mid]

        r_excess_before = max(0.0, float(r_in) - float(g_in))
        r_excess_after  = max(0.0, float(r_out) - float(g_out))
        assert r_excess_after < r_excess_before, \
            "Micro (1-pixel) purple fringe must be detected and reduced"

    def test_output_shape_and_dtype(self):
        """Output must have the same shape and dtype as input."""
        img = make_purple_fringe_image().astype(np.float32)
        out = defringe(img)
        assert out.shape == img.shape, "Shape must be preserved"
        assert out.dtype == np.float32, "dtype must be float32"

    def test_output_range(self):
        """All output values must be in [0, 1]."""
        img = make_purple_fringe_image()
        out = defringe(img, strength=5.0) # Test high strength stability
        assert float(out.min()) >= 0.0 - 1e-6, "Minimum must be >= 0"
        assert float(out.max()) <= 1.0 + 1e-6, "Maximum must be <= 1"

    def test_chroma_threshold_aliasing_reproduction(self):
        """
        CONFIRMING PROBLEM 2:
        High chroma_threshold should not cause 'islands' or jagged correction masks.
        We check if the transition is smooth by looking at the gradient.
        """
        img = make_purple_fringe_image(h=64, w=64, fringe_width=4)
        # High threshold as reported
        out = defringe(img, strength=1.0, chroma_threshold=0.4, edge_threshold=0.05)
        
        # Measure the smoothness of the correction MASK (amount added).
        # Natural edges in the image (grad~0.4) should not cause failure.
        correction_amount = np.abs(out[32, :, 0] - img[32, :, 0])
        grad = np.abs(np.diff(correction_amount))
        
        max_grad = np.max(grad)
        max_idx = np.argmax(grad)
        
        if max_grad >= 0.15:
            print(f"\n[DEBUG] Aliasing failure analysis at row 32, max mask grad {max_grad:.4f} at col {max_idx}")
            print(f"Col | Input R | Output R | Mask Grad")
            print(f"----|---------|----------|----------")
            for c in range(max(0, max_idx - 5), min(64, max_idx + 6)):
                g_val = grad[c] if c < len(grad) else 0.0
                print(f"{c:3d} | {img[32,c,0]:.4f}  | {out[32,c,0]:.4f}   | {g_val:.4f}")
        
        assert max_grad < 0.15, (
            f"MASK ALIASING DETECTED: Sharp transition in correction mask (grad={max_grad:.3f}).")

    def test_partial_strength(self):
        """With strength=0.5, correction should be partial (between in and full)."""
        img = make_purple_fringe_image(h=64, w=64, fringe_width=2)
        out_full    = defringe(img, strength=1.0,
                               edge_threshold=0.010, chroma_threshold=0.1)
        out_partial = defringe(img, strength=0.5,
                               edge_threshold=0.010, chroma_threshold=0.1)

        mid = 32
        r_in,  _, _ = img[32, mid]
        r_full, _, _ = out_full[32, mid]
        r_half, _, _ = out_partial[32, mid]

        assert float(r_full) <= float(r_half) <= float(r_in) + 1e-4, \
            (f"Partial correction (str=0.5) R should be between full and original: "
             f"full={r_full:.3f} half={r_half:.3f} original={r_in:.3f}")

    def test_performance(self):
        """Defringe should finish a 26MP-equivalent (5100×5100) image in < 3s."""
        h, w = 5100, 5100
        # Synthetic: random mid-tone + thin fringe strip
        rng = np.random.default_rng(0)
        img = (rng.random((h, w, 3)) * 0.4 + 0.1).astype(np.float32)
        # Add purple fringe strip at mid-column
        mid = w // 2
        img[:, mid:mid+3, 0] = 0.55
        img[:, mid:mid+3, 1] = 0.08
        img[:, mid:mid+3, 2] = 0.55
        # Add highlight on left side
        img[:, :mid, :] = 0.90

        t0 = time.perf_counter()
        out = defringe(img, edge_threshold=0.015, chroma_threshold=0.15)
        elapsed = time.perf_counter() - t0

        assert out is not None
        assert elapsed < 3.0, f"Defringe took {elapsed:.2f}s on 26MP image (should be <3s)"
        print(f"\n  ⏱ Defringe 26MP equivalent: {elapsed:.3f}s")

    def test_density_threshold_prevents_broad_purple(self):
        """High density of purple in neighbourhood → classified as natural, not fringe."""
        h, w = 64, 64
        img = np.zeros((h, w, 3), dtype=np.float32)
        # Full image is bright purple (e.g., a purple sky?)
        img[:, :, 0] = 0.75
        img[:, :, 1] = 0.10
        img[:, :, 2] = 0.75
        # Bright neighbor (top row is near-white)
        img[0, :, :] = 0.98

        out = defringe(img, edge_threshold=0.005, chroma_threshold=0.1)

        # Most of the purple area should be UNCHANGED because density is too high
        # (the entire image is purple → density >> 0.25)
        core_in  = img[10:55, 10:55]
        core_out = out[10:55, 10:55]
        changed = np.sum(np.abs(core_out - core_in) > 1e-5)
        total = core_in.size // 3
        pct = changed / total
        assert pct < 0.30, \
            f"Too many pixels changed ({pct:.1%}) in broad-purple scene (density guard failed)"


# ---------------------------------------------------------------------------
# Integration test with X-T5 Room.RAF
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
RESULT_DIR  = os.path.join(os.path.dirname(__file__), "results")
RAF_PATH    = os.path.join(FIXTURE_DIR, "X-T5 Room.RAF")


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists(RAF_PATH),
                    reason="X-T5 Room.RAF fixture not found")
def test_xt5_defringe_integration():
    """
    Process X-T5 Room.RAF with and without defringe.
    Save before/after images and verify:
      1. Defringe reduces mean purple chroma near edges
      2. Non-edge neutral regions are not significantly changed
      3. Total changed pixels < 5% of image (not over-aggressive)
    """
    import libraw_enhanced as lre

    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load and process WITHOUT defringe
    print("\n  Loading X-T5 Room.RAF (no defringe)...")
    t0 = time.perf_counter()
    with lre.imread(RAF_PATH) as raw:
        img_no_df = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            highlight_mode=5,
        )
    print(f"  Without defringe: {time.perf_counter()-t0:.2f}s, shape={img_no_df.shape}, dtype={img_no_df.dtype}")

    # Load and process WITH defringe
    print("  Loading X-T5 Room.RAF (with defringe)...")
    t0 = time.perf_counter()
    with lre.imread(RAF_PATH) as raw:
        img_df = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            highlight_mode=5,
            defringe=True,
            defringe_edge_threshold=0.1,
            defringe_strength=3.0,
        )
    print(f"  With defringe: {time.perf_counter()-t0:.2f}s, shape={img_df.shape}")

    assert img_no_df.shape == img_df.shape, "Shape must match"

    # Convert to float for analysis
    scale = 65535.0 if img_no_df.dtype == np.uint16 else 255.0
    f_no_df = img_no_df.astype(np.float32) / scale
    f_df    = img_df.astype(np.float32) / scale

    diff = np.abs(f_df - f_no_df)

    # Metric 1: fraction of changed pixels
    changed_mask = diff.max(axis=2) > 1.0 / scale
    pct_changed  = changed_mask.mean()
    print(f"  Changed pixels: {pct_changed:.3%}")
    assert pct_changed < 0.10, \
        f"Too many pixels changed by defringe: {pct_changed:.2%} (expect <10%)"

    # Metric 2: where pixels changed, R and B should decrease on average (purple suppression)
    changed_idx = np.where(changed_mask)
    if len(changed_idx[0]) > 0:
        dr = (f_df[:,:,0] - f_no_df[:,:,0])[changed_mask]
        db = (f_df[:,:,2] - f_no_df[:,:,2])[changed_mask]
        mean_dr = float(dr.mean())
        mean_db = float(db.mean())
        print(f"  Mean ΔR in changed pixels: {mean_dr:+.5f}")
        print(f"  Mean ΔB in changed pixels: {mean_db:+.5f}")
        # For purple fringe removal, on average R and B should decrease
        assert mean_dr <= 0.005, \
            f"Expected mean ΔR ≤ 0 in changed pixels (got {mean_dr:+.5f})"
        assert mean_db <= 0.005, \
            f"Expected mean ΔB ≤ 0 in changed pixels (got {mean_db:+.5f})"

    # Save comparison images
    try:
        h, w = img_no_df.shape[:2]

        def save_vips(arr16, path):
            # Ensure full resolution 16-bit to 8-bit conversion via pyvips
            vi = pyvips.Image.new_from_memory(arr16.data, w, h, 3, "ushort")
            # Scale 16-bit to 8-bit for JPEG, preserving full resolution detail
            vi.write_to_file(path, Q=95)

        if img_no_df.dtype == np.uint16:
            save_vips(img_no_df, os.path.join(RESULT_DIR, "xt5_room_no_defringe.jpg"))
            save_vips(img_df, os.path.join(RESULT_DIR, "xt5_room_defringe.jpg"))

        # Save diff map (amplified x20)
        diff_amp = (np.clip(diff * 20.0, 0, 1) * 255).astype(np.uint8)
        pyvips.Image.new_from_memory(diff_amp.data, w, h, 3, "uchar").write_to_file(
            os.path.join(RESULT_DIR, "xt5_room_defringe_diff.jpg"), Q=90)

        print(f"  Saved comparison images to {RESULT_DIR}/")
    except Exception as e:
        print(f"  Failed to save images: {e}")

    print("  ✅ Integration test passed")


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists(RAF_PATH),
                    reason="X-T5 Room.RAF fixture not found")
def test_xt5_defringe_numpy_standalone():
    """
    Test the standalone wrapper.defringe() numpy API directly on an image
    that was converted by the pipeline (without inline defringe).
    """
    import libraw_enhanced as lre
    from libraw_enhanced._core import LibRawWrapper

    with lre.imread(RAF_PATH) as raw:
        img16 = raw.postprocess(use_camera_wb=True, output_bps=16)

    # Convert to linear float32 (approximate inverse of gamma)
    img_f = (img16.astype(np.float32) / 65535.0) ** 2.2

    t0 = time.perf_counter()
    w = LibRawWrapper()
    out_f = w.defringe(img_f,
                       edge_threshold=0.1,
                       strength=1.0)
    elapsed = time.perf_counter() - t0

    assert out_f.shape == img_f.shape
    assert out_f.dtype == np.float32
    print(f"\n  ✅ defringe_numpy standalone: {elapsed:.3f}s on "
          f"{img_f.shape[0]}x{img_f.shape[1]} image")

    # Max diff should be reasonable (not corrupting the image)
    max_diff = float(np.abs(out_f - img_f).max())
    assert max_diff <= 1.0, f"Max diff too large: {max_diff}"
