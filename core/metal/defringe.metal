//
// defringe.metal
//
// GPU port of CPUAccelerator::defringe — guide-green chroma-ratio
// fringe suppression.  The Gaussian-blur passes are handled by
// MPSImageGaussianBlur on the orchestration side; this file provides
// the algorithm-specific custom kernels.
//
// Pipeline (in execution order):
//
//   defringe_setup           — split RGB → R, G, B, guide_linear,
//                               guide_signal, log_rg, log_bg
//   defringe_sobel_raw       — un-normalised Sobel magnitude on
//                               guide_signal (used as `edge` and edge_raw)
//   defringe_edge_normalize  — divide edge by max(edge_raw) using a
//                               buffer-supplied scalar (computed via
//                               MPSImageStatisticsMinAndMax on edge_raw)
//   MPS Gaussian:
//     blur_guide  ← guide_linear
//     blur_log_rg ← log_rg
//     blur_log_bg ← log_bg
//     blur_edge   ← edge (normalised)
//     blur_edge_raw ← edge_raw
//   defringe_support_seed    — compute fringe_support per pixel
//   MPS Gaussian: blur_fringe_support ← fringe_support
//   defringe_line_link       — 8-direction line linkage map
//   defringe_solo_seed       — compute solo_r_seed, solo_b_seed
//   MPS Gaussian: solo_r_mask_blur, solo_b_mask_blur
//   defringe_main            — primary mask + R/G/B correction
//   defringe_solo            — high-strength one-sided R/B cleanup
//

#include <metal_stdlib>
#include "shader_types.h"
using namespace metal;

// --------------------------------------------------------------------
// Small smoothstep helper.
inline float ss(float lo, float hi, float x) {
    float t = clamp((x - lo) / max(hi - lo, 1e-12f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// --------------------------------------------------------------------
// Stage 1: setup planes.  Reads the source RGB and writes
//   R, G, B (clamped to >= 0), guide_linear = max(G, 0),
//   guide_signal = sqrt(clamp(G/max_guide, 0, 1)),
//   log_rg, log_bg = log ratios used for chroma analysis.
kernel void defringe_setup(
    texture2d<float, access::read>  src             [[texture(0)]],
    texture2d<float, access::write> R_plane         [[texture(1)]],
    texture2d<float, access::write> G_plane         [[texture(2)]],
    texture2d<float, access::write> B_plane         [[texture(3)]],
    texture2d<float, access::write> guide_linear    [[texture(4)]],
    texture2d<float, access::write> guide_signal    [[texture(5)]],
    texture2d<float, access::write> log_rg          [[texture(6)]],
    texture2d<float, access::write> log_bg          [[texture(7)]],
    constant DefringeParams& p                      [[buffer(0)]],
    uint2 gid                                       [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    float4 c = src.read(gid);
    float r = max(c.r, 0.0f);
    float g = max(c.g, 0.0f);
    float b = max(c.b, 0.0f);

    R_plane.write(r, gid);
    G_plane.write(g, gid);
    B_plane.write(b, gid);
    guide_linear.write(g, gid);
    float gs = sqrt(clamp(g / p.max_guide, 0.0f, 1.0f));
    guide_signal.write(gs, gid);
    float lrg = log((r + p.ratio_eps) / (g + p.ratio_eps));
    float lbg = log((b + p.ratio_eps) / (g + p.ratio_eps));
    log_rg.write(lrg, gid);
    log_bg.write(lbg, gid);
}

// --------------------------------------------------------------------
// Stage 2: Sobel-style un-normalised gradient magnitude on
// guide_signal.  The CPU reference uses the standard 3×3 Sobel kernel
// (NOT the 2-tap diff used by axial_ca), so we replicate it.  Output
// is written to both `edge` (which will later be divided by max) and
// `edge_raw` (un-normalised) — duplicated to avoid an extra copy.
kernel void defringe_sobel(
    texture2d<float, access::read>  signal   [[texture(0)]],
    texture2d<float, access::write> edge_out [[texture(1)]],
    texture2d<float, access::write> raw_out  [[texture(2)]],
    constant uint2& dims                     [[buffer(0)]],
    uint2 gid                                 [[thread_position_in_grid]])
{
    if (gid.x >= dims.x || gid.y >= dims.y) return;
    if (gid.x == 0 || gid.y == 0 || gid.x + 1 >= dims.x || gid.y + 1 >= dims.y) {
        edge_out.write(0.0f, gid);
        raw_out.write(0.0f, gid);
        return;
    }
    const uint x = gid.x;
    const uint y = gid.y;
    const float a = signal.read(uint2(x-1, y-1)).r;
    const float b = signal.read(uint2(x  , y-1)).r;
    const float c = signal.read(uint2(x+1, y-1)).r;
    const float d = signal.read(uint2(x-1, y  )).r;
    const float e = signal.read(uint2(x+1, y  )).r;
    const float f = signal.read(uint2(x-1, y+1)).r;
    const float g = signal.read(uint2(x  , y+1)).r;
    const float h = signal.read(uint2(x+1, y+1)).r;
    const float gx = -a + c - 2.0f * d + 2.0f * e - f + h;
    const float gy = -a - 2.0f * b - c + f + 2.0f * g + h;
    const float mag = sqrt(gx * gx + gy * gy);
    edge_out.write(mag, gid);
    raw_out.write(mag,  gid);
}

// --------------------------------------------------------------------
// Stage 3: normalise edge by max_edge supplied as a scalar buffer.
// max_edge is computed via MPSImageStatisticsMinAndMax on edge_raw.
kernel void defringe_edge_normalize(
    texture2d<float, access::read>  edge_in [[texture(0)]],
    texture2d<float, access::write> edge_out[[texture(1)]],
    constant uint2& dims                    [[buffer(0)]],
    constant float& max_edge                [[buffer(1)]],
    uint2 gid                                [[thread_position_in_grid]])
{
    if (gid.x >= dims.x || gid.y >= dims.y) return;
    float v = edge_in.read(gid).r;
    float inv = (max_edge > 1e-6f) ? (1.0f / max_edge) : 0.0f;
    edge_out.write(v * inv, gid);
}

// --------------------------------------------------------------------
// Stage 4: fringe_support seed per pixel.  Mirrors the CPU code in
// CPUAccelerator::defringe between the "soft neighbourhood support
// map" comment block and the gaussian_blur of fringe_support.
kernel void defringe_support_seed(
    texture2d<float, access::read>  R_plane        [[texture(0)]],
    texture2d<float, access::read>  G_plane        [[texture(1)]],
    texture2d<float, access::read>  B_plane        [[texture(2)]],
    texture2d<float, access::read>  log_rg         [[texture(3)]],
    texture2d<float, access::read>  log_bg         [[texture(4)]],
    texture2d<float, access::read>  blur_guide     [[texture(5)]],
    texture2d<float, access::read>  blur_log_rg    [[texture(6)]],
    texture2d<float, access::read>  blur_log_bg    [[texture(7)]],
    texture2d<float, access::read>  blur_edge      [[texture(8)]],
    texture2d<float, access::read>  blur_edge_raw  [[texture(9)]],
    texture2d<float, access::write> fringe_support [[texture(10)]],
    constant DefringeParams& p                     [[buffer(0)]],
    uint2 gid                                       [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    const float rg = log_rg.read(gid).r;
    const float bg = log_bg.read(gid).r;
    const float dr = rg - blur_log_rg.read(gid).r;
    const float db = bg - blur_log_bg.read(gid).r;
    const float r0 = R_plane.read(gid).r;
    const float g0 = G_plane.read(gid).r;
    const float b0 = B_plane.read(gid).r;

    const float purple_excess = min(max(0.0f, dr), max(0.0f, db));
    const float denom_support = max(blur_guide.read(gid).r, 0.05f);
    const float g_over_min_support =
        max(0.0f, g0 - min(r0, b0)) / denom_support;
    const float green_dom_support = ss(0.010f, 0.060f, g_over_min_support);
    const float green_excess =
        max(max(0.0f, -dr), max(0.0f, -db)) * green_dom_support;
    const bool is_purple_support = purple_excess >= green_excess;
    const float excess_support = is_purple_support ? purple_excess : green_excess;

    const float min_dir_support = is_purple_support
        ? max(0.0f, min(rg, bg))
        : max(0.0f, min(-rg, -bg));
    const float max_dir_support = is_purple_support
        ? max(1e-4f, max(rg, bg))
        : max(1e-4f, max(-rg, -bg));
    const float balance_support = min_dir_support / max_dir_support;

    const float c_edge_support =
        ss(p.edge_threshold * 0.35f * p.inv_sensitivity,
           p.edge_threshold * 1.50f * p.inv_sensitivity,
           blur_edge.read(gid).r) *
        ss(0.55f * p.inv_sensitivity,
           1.35f * p.inv_sensitivity,
           blur_edge_raw.read(gid).r);
    const float c_bright_support =
        ss(0.24f * p.inv_sensitivity,
           0.52f * p.inv_sensitivity,
           blur_guide.read(gid).r);
    const float c_chroma_support =
        ss(p.chroma_threshold * 0.45f * p.inv_sensitivity,
           p.chroma_threshold * 1.60f * p.inv_sensitivity,
           excess_support);

    const float red_strength_support =
        max(0.0f, r0 - max(g0, b0)) / max(r0, 1e-4f);
    const float red_blue_balance_support = b0 / max(r0, 1e-4f);
    const float blue_lift_support =
        max(0.0f, b0 - g0) / max(r0 - g0, 1e-4f);
    const float red_purple_support =
        max(ss(0.55f, 0.78f, red_blue_balance_support),
            ss(0.16f, 0.34f, blue_lift_support));
    const float red_purple_hue_signal =
        ss(0.50f, 0.72f, red_blue_balance_support) *
        ss(0.24f, 0.42f, blue_lift_support);
    const float balanced_hue_support =
        ss(0.06f * p.inv_sensitivity,
           0.22f * p.inv_sensitivity,
           min_dir_support) *
        ss(0.35f, 0.75f, balance_support);
    const float red_purple_hue_support = is_purple_support
        ? min(1.0f, red_purple_hue_signal * 1.50f) *
          ss(0.035f * p.inv_sensitivity,
             0.16f  * p.inv_sensitivity,
             min_dir_support) *
          ss(0.18f, 0.58f, balance_support)
        : 0.0f;
    const float c_hue_support = max(balanced_hue_support, red_purple_hue_support);
    const float natural_red_support_guard =
        1.0f - ss(0.08f, 0.25f, red_strength_support) *
               (1.0f - red_purple_support);
    const float green_support_enable =
        (p.enable_green_defringe != 0u || is_purple_support) ? 1.0f : 0.0f;

    fringe_support.write(c_edge_support * c_bright_support *
                          c_hue_support * c_chroma_support *
                          natural_red_support_guard * green_support_enable,
                          gid);
}

// --------------------------------------------------------------------
// Stage 5: line_fringe_support — 8-direction "thin line" detector.
// Walks up to max_step pixels along each of 8 directions sampling the
// fringe support map (with a 55 % blur-fringe pad), keeps the
// "thin × along" maximum.
// Metal helpers used by defringe_line_link.  Metal does not allow C++
// lambdas inside kernel bodies, so we factor the closures into regular
// inline functions taking textures + bounds as explicit arguments.
inline float _line_sample(texture2d<float, access::read> fringe_support,
                          texture2d<float, access::read> blur_fringe_support,
                          int x, int y, int W, int H) {
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    const float fs  = fringe_support.read(uint2(uint(x), uint(y))).r;
    const float bfs = blur_fringe_support.read(uint2(uint(x), uint(y))).r;
    return max(fs, bfs * 0.55f);
}

inline float _line_sample_fs(texture2d<float, access::read> fringe_support,
                              int x, int y, int W, int H) {
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    return fringe_support.read(uint2(uint(x), uint(y))).r;
}

kernel void defringe_line_link(
    texture2d<float, access::read>  fringe_support      [[texture(0)]],
    texture2d<float, access::read>  blur_fringe_support [[texture(1)]],
    texture2d<float, access::write> line_support_out    [[texture(2)]],
    constant uint2& dims                                [[buffer(0)]],
    uint2 gid                                            [[thread_position_in_grid]])
{
    if (gid.x >= dims.x || gid.y >= dims.y) return;
    const int W = int(dims.x);
    const int H = int(dims.y);

    const int2 dirs[8] = {
        int2(1, 0),  int2(0, 1),  int2(1, 1),   int2(1, -1),
        int2(2, 1),  int2(1, 2),  int2(2, -1),  int2(1, -2)
    };
    float best = 0.0f;
    const int cx = int(gid.x);
    const int cy = int(gid.y);
    for (int d = 0; d < 8; d++) {
        const int tx = dirs[d].x;
        const int ty = dirs[d].y;
        const int nx = -ty;
        const int ny =  tx;
        float along = 0.0f;
        const int max_step = (d < 4) ? 14 : 6;
        for (int s = 1; s <= max_step; s++) {
            along = max(along, _line_sample(fringe_support, blur_fringe_support,
                                              cx + tx * s, cy + ty * s, W, H));
            along = max(along, _line_sample(fringe_support, blur_fringe_support,
                                              cx - tx * s, cy - ty * s, W, H));
        }
        const float across = max(
            _line_sample_fs(fringe_support, cx + nx * 2, cy + ny * 2, W, H),
            _line_sample_fs(fringe_support, cx - nx * 2, cy - ny * 2, W, H));
        const float thin = 1.0f - ss(along * 0.45f, along * 0.95f, across);
        best = max(best, along * thin);
    }
    line_support_out.write(best, gid);
}

// --------------------------------------------------------------------
// Stage 6: solo_r_seed and solo_b_seed.  These feed two more Gaussian
// blurs that produce solo_*_mask_blur consumed by the high-strength
// one-sided cleanup path.
kernel void defringe_solo_seed(
    texture2d<float, access::read>  R_plane             [[texture(0)]],
    texture2d<float, access::read>  G_plane             [[texture(1)]],
    texture2d<float, access::read>  B_plane             [[texture(2)]],
    texture2d<float, access::read>  blur_guide          [[texture(3)]],
    texture2d<float, access::read>  blur_log_rg         [[texture(4)]],
    texture2d<float, access::read>  blur_log_bg         [[texture(5)]],
    texture2d<float, access::read>  blur_edge           [[texture(6)]],
    texture2d<float, access::read>  blur_edge_raw       [[texture(7)]],
    texture2d<float, access::read>  blur_fringe_support [[texture(8)]],
    texture2d<float, access::read>  line_fringe_support [[texture(9)]],
    texture2d<float, access::write> solo_r_seed_out     [[texture(10)]],
    texture2d<float, access::write> solo_b_seed_out     [[texture(11)]],
    constant DefringeParams& p                          [[buffer(0)]],
    uint2 gid                                            [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    const float g       = max(G_plane.read(gid).r, 0.0f);
    const float rg_exp  = exp(blur_log_rg.read(gid).r);
    const float bg_exp  = exp(blur_log_bg.read(gid).r);
    const float target_r = g * rg_exp;
    const float target_b = g * bg_exp;
    const float r0 = max(R_plane.read(gid).r, 0.0f);
    const float g0 = g;
    const float b0 = max(B_plane.read(gid).r, 0.0f);
    const float denom = max(blur_guide.read(gid).r, 0.05f);
    const float solo_r = max(0.0f, r0 - target_r) / denom;
    const float solo_b = max(0.0f, b0 - target_b) / denom;
    const float linked_fringe = ss(0.004f, 0.045f,
        max(blur_fringe_support.read(gid).r,
            line_fringe_support.read(gid).r * 0.90f));
    const float c_edge_solo =
        ss(p.edge_threshold * 0.35f * p.inv_sensitivity,
           p.edge_threshold * 1.50f * p.inv_sensitivity,
           blur_edge.read(gid).r) *
        ss(0.55f * p.inv_sensitivity,
           1.35f * p.inv_sensitivity,
           blur_edge_raw.read(gid).r);
    const float c_bright_solo =
        ss(0.24f * p.inv_sensitivity,
           0.52f * p.inv_sensitivity,
           blur_guide.read(gid).r);
    const float solo_enable = ss(1.25f, 3.0f, p.strength);
    const float red_strength_solo = max(0.0f, r0 - max(g0, b0)) / max(r0, 1e-4f);
    const float solo_red_guard = 1.0f - 0.90f * ss(0.12f, 0.32f, red_strength_solo);

    solo_r_seed_out.write(solo_enable * c_edge_solo * c_bright_solo *
                           linked_fringe * solo_red_guard *
                           ss(p.chroma_threshold * 0.22f * p.inv_sensitivity,
                              p.chroma_threshold * 0.95f * p.inv_sensitivity,
                              solo_r),
                           gid);
    solo_b_seed_out.write(solo_enable * c_edge_solo * c_bright_solo *
                           ss(p.chroma_threshold * 0.22f * p.inv_sensitivity,
                              p.chroma_threshold * 0.95f * p.inv_sensitivity,
                              solo_b),
                           gid);
}

// --------------------------------------------------------------------
// Stage 7: main correction kernel.  Combines the per-pixel mask
// evaluation, the warm/red/blob guard chain and the actual RGB write
// from the CPU "Step 4" loop.  This is the largest kernel; it mirrors
// the CPU code closely.  Writes the corrected RGB into the dst
// RGBA32Float texture (G/A pass-through where applicable).
kernel void defringe_main(
    texture2d<float, access::read>  src                 [[texture(0)]],
    texture2d<float, access::read>  R_plane             [[texture(1)]],
    texture2d<float, access::read>  G_plane             [[texture(2)]],
    texture2d<float, access::read>  B_plane             [[texture(3)]],
    texture2d<float, access::read>  log_rg              [[texture(4)]],
    texture2d<float, access::read>  log_bg              [[texture(5)]],
    texture2d<float, access::read>  blur_guide          [[texture(6)]],
    texture2d<float, access::read>  blur_log_rg         [[texture(7)]],
    texture2d<float, access::read>  blur_log_bg         [[texture(8)]],
    texture2d<float, access::read>  blur_edge           [[texture(9)]],
    texture2d<float, access::read>  blur_edge_raw       [[texture(10)]],
    texture2d<float, access::read>  blur_fringe_support [[texture(11)]],
    texture2d<float, access::read>  line_fringe_support [[texture(12)]],
    texture2d<float, access::write> dst                 [[texture(13)]],
    constant DefringeParams& p                          [[buffer(0)]],
    uint2 gid                                            [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;

    const float rg = log_rg.read(gid).r;
    const float bg = log_bg.read(gid).r;
    const float dr = rg - blur_log_rg.read(gid).r;
    const float db = bg - blur_log_bg.read(gid).r;

    const float r0 = R_plane.read(gid).r;
    const float g0 = G_plane.read(gid).r;
    const float b0 = B_plane.read(gid).r;
    const float rb_avg_pre = 0.5f * (r0 + b0);

    const float purple_r = max(0.0f, dr);
    const float purple_b = max(0.0f, db);
    const float green_r  = max(0.0f, -dr);
    const float green_b  = max(0.0f, -db);

    const float purple_excess = min(purple_r, purple_b);
    const float denom = max(blur_guide.read(gid).r, 0.05f);
    const float g_over_min = max(0.0f, g0 - min(r0, b0)) / denom;
    const float green_dom = ss(0.010f, 0.060f, g_over_min);
    const float green_excess = max(green_r, green_b) * green_dom;
    const bool is_purple = purple_excess >= green_excess;
    const float excess = is_purple ? purple_excess : green_excess;

    const float min_dir = is_purple
        ? max(0.0f, min(rg, bg))
        : max(0.0f, min(-rg, -bg));
    const float max_dir = is_purple
        ? max(1e-4f, max(rg, bg))
        : max(1e-4f, max(-rg, -bg));
    const float balance = min_dir / max_dir;

    const float c_edge_core =
        ss(p.edge_threshold * 0.35f * p.inv_sensitivity,
           p.edge_threshold * 1.50f * p.inv_sensitivity,
           blur_edge.read(gid).r) *
        ss(0.55f * p.inv_sensitivity,
           1.35f * p.inv_sensitivity,
           blur_edge_raw.read(gid).r);
    const float c_edge_low =
        ss(p.edge_threshold * 0.10f * p.inv_sensitivity,
           p.edge_threshold * 0.80f * p.inv_sensitivity,
           blur_edge.read(gid).r) *
        ss(0.20f * p.inv_sensitivity,
           0.90f * p.inv_sensitivity,
           blur_edge_raw.read(gid).r);
    const float linked_fringe = ss(0.004f, 0.045f,
        max(blur_fringe_support.read(gid).r,
            line_fringe_support.read(gid).r * 0.90f));
    const float c_bright = ss(0.24f * p.inv_sensitivity,
                              0.52f * p.inv_sensitivity,
                              blur_guide.read(gid).r);
    const float c_chroma = ss(p.chroma_threshold * 0.45f * p.inv_sensitivity,
                              p.chroma_threshold * 1.60f * p.inv_sensitivity,
                              excess);

    const float red_blue_balance_hue = b0 / max(r0, 1e-4f);
    const float blue_lift_hue = max(0.0f, b0 - g0) / max(r0 - g0, 1e-4f);
    const float red_purple_hue_signal =
        ss(0.50f, 0.72f, red_blue_balance_hue) *
        ss(0.24f, 0.42f, blue_lift_hue);
    const float balanced_hue =
        ss(0.06f * p.inv_sensitivity,
           0.22f * p.inv_sensitivity,
           min_dir) *
        ss(0.35f, 0.75f, balance);
    const float red_purple_hue_gate = is_purple
        ? min(1.0f, red_purple_hue_signal * 1.50f) *
          ss(0.035f * p.inv_sensitivity,
             0.16f  * p.inv_sensitivity,
             min_dir) *
          ss(0.18f, 0.58f, balance)
        : 0.0f;
    const float green_unbalanced_hue =
        (p.enable_green_defringe != 0u && !is_purple)
        ? ss(0.035f * p.inv_sensitivity,
             0.16f  * p.inv_sensitivity,
             max_dir)
        : 0.0f;
    const float c_hue = max(balanced_hue,
                            max(red_purple_hue_gate, green_unbalanced_hue));

    const float green_soft_edge_boost =
        (p.enable_green_defringe != 0u && !is_purple)
        ? c_edge_low * ss(0.05f * p.inv_sensitivity,
                          0.20f * p.inv_sensitivity,
                          max_dir)
        : 0.0f;

    const float rb_avg = rb_avg_pre;
    const float purple_abs =
        min(max(0.0f, r0 - g0), max(0.0f, b0 - g0)) /
        max(blur_guide.read(gid).r, 0.05f);
    const float green_abs_base = max(0.0f, g0 - rb_avg) /
        max(blur_guide.read(gid).r, 0.05f);
    const float green_abs_alt = max(0.0f, g0 - min(r0, b0)) /
        max(blur_guide.read(gid).r, 0.05f);
    const float green_abs = max(green_abs_base, 0.55f * green_abs_alt);
    const bool abs_is_purple = purple_abs >= green_abs;
    const float abs_excess = abs_is_purple ? purple_abs : green_abs;
    const float c_abs = ss(p.chroma_threshold * 0.35f * p.inv_sensitivity,
                            p.chroma_threshold * 1.20f * p.inv_sensitivity,
                            abs_excess);
    const float low_contrast_color =
        ss(p.chroma_threshold * 1.20f * p.inv_sensitivity,
           p.chroma_threshold * 2.40f * p.inv_sensitivity,
           abs_excess);
    const float c_edge = max(c_edge_core,
                              max(max(c_edge_low * linked_fringe * 1.45f,
                                       c_edge_low * low_contrast_color * 0.35f),
                                   green_soft_edge_boost));
    const float ratio_mask  = c_edge * c_bright * c_hue * c_chroma;
    const float direct_mask = c_edge * c_bright * c_hue * c_abs;
    const bool use_direct = direct_mask > ratio_mask;
    float mask = max(ratio_mask, direct_mask) * p.scene_highlight_gate;
    const bool correct_purple = use_direct ? abs_is_purple : is_purple;

    float local_gate = 0.0f;
    if (correct_purple) {
        local_gate = ss(p.chroma_threshold * 0.10f * p.inv_sensitivity,
                         p.chroma_threshold * 0.55f * p.inv_sensitivity,
                         purple_abs);
        mask *= local_gate;
    } else {
        local_gate = ss(p.chroma_threshold * 0.10f * p.inv_sensitivity,
                         p.chroma_threshold * 0.55f * p.inv_sensitivity,
                         green_abs);
        const float green_highlight_protect =
            1.0f - 0.85f * ss(0.62f, 0.88f, blur_guide.read(gid).r);
        const float rb_to_green = rb_avg / max(g0, 1e-4f);
        const float natural_green_highlight =
            ss(0.22f, 0.46f, rb_to_green) *
            ss(0.30f, 0.55f, g0);
        const float natural_green_suppress = 1.0f - 0.99f * natural_green_highlight;
        const float natural_green_suppress_sq =
            natural_green_suppress * natural_green_suppress;

        const float green_halo_strength =
            ss(p.chroma_threshold * 0.55f * p.inv_sensitivity,
               p.chroma_threshold * 1.65f * p.inv_sensitivity,
               green_abs);
        const float green_gain_base  = (p.enable_green_defringe != 0u) ? 0.60f : 0.35f;
        const float green_gain_slope = (p.enable_green_defringe != 0u) ? 1.10f : 0.95f;
        const float green_gain = green_gain_base + green_gain_slope * green_halo_strength;

        const float min_ratio_ref = min(blur_log_rg.read(gid).r,
                                         blur_log_bg.read(gid).r);
        const float green_neighbour =
            ss(0.02f, 0.10f, max(0.0f, -min_ratio_ref));
        const float src_max = max(r0, max(g0, b0));
        const float src_min = min(r0, min(g0, b0));
        const float src_chroma = src_max - src_min;
        const float src_sat = src_chroma / max(src_max, 1e-4f);
        const float whiteish = 1.0f - ss(0.05f, 0.13f, src_sat);
        const float foliage_guard =
            green_neighbour * whiteish *
            ss(0.35f, 0.75f, blur_guide.read(gid).r) *
            (1.0f - green_halo_strength);

        const float strength_release = (p.enable_green_defringe != 0u)
            ? ss(2.0f, 6.0f, p.strength)
            : 0.0f;
        const float green_line_evidence =
            ss(0.004f, 0.030f, line_fringe_support.read(gid).r) *
            ss(0.20f,  0.55f, green_halo_strength) *
            ss(0.05f,  0.15f, src_sat);
        const float guard_release = strength_release * green_line_evidence;

        const float released_highlight_protect =
            green_highlight_protect +
            (1.0f - green_highlight_protect) * guard_release;
        const float released_natural_suppress =
            natural_green_suppress_sq +
            (1.0f - natural_green_suppress_sq) * guard_release;
        const float foliage_term = 1.0f - 0.999f * foliage_guard;
        const float released_foliage_term =
            foliage_term + (1.0f - foliage_term) * guard_release;

        mask *= local_gate * green_gain * released_highlight_protect *
                 released_natural_suppress * released_foliage_term;

        const float line_kill_evidence =
            ss(0.04f, 0.15f, line_fringe_support.read(gid).r);
        const float natural_blob =
            sqrt(max(0.0f, natural_green_highlight)) *
            (1.0f - line_kill_evidence) * green_neighbour;
        mask *= 1.0f - 0.999f * natural_blob;

        if (p.enable_green_defringe == 0u) {
            mask = 0.0f;
        }
    }

    // Default: pass-through.
    float out_r = r0;
    float out_g = g0;
    float out_b = b0;

    if (mask > p.mask_cutoff) {
        const float red_strength = max(0.0f, r0 - max(g0, b0)) / max(r0, 1e-4f);
        const float red_blue_balance = b0 / max(r0, 1e-4f);
        const float blue_lift = max(0.0f, b0 - g0) / max(r0 - g0, 1e-4f);
        const float red_purple_balance = ss(0.55f, 0.78f, red_blue_balance);
        const float red_purple_lift    = ss(0.16f, 0.34f, blue_lift);
        const float red_purple = max(red_purple_balance, red_purple_lift);
        const float red_strength_guard = ss(0.08f, 0.25f, red_strength);

        float red_purple_context = 0.0f;
        float contextual_red_purple = 0.0f;
        float red_magenta_surface = 0.0f;
        if (red_purple > 0.50f) {
            red_purple_context = linked_fringe;
            contextual_red_purple = red_purple * red_purple_context;
            if (red_purple_context < 0.20f &&
                red_strength_guard < 0.50f &&
                red_purple > 0.50f) {
                red_magenta_surface = ss(0.02f, 0.12f, purple_abs) *
                                       (1.0f - red_purple_context);
            }
        }
        float neutral_gray_surface = 0.0f;
        if (p.strength > 3.0f &&
            red_purple_context > 0.80f &&
            line_fringe_support.read(gid).r > 0.012f &&
            red_strength_guard < 0.98f &&
            blue_lift > 1.10f &&
            red_blue_balance > 0.82f &&
            purple_abs > 0.015f) {
            const float neutral_gray_edge =
                ss(1.10f, 1.80f, blue_lift) *
                ss(0.82f, 1.05f, red_blue_balance) *
                (1.0f - red_strength_guard);
            neutral_gray_surface =
                neutral_gray_edge * ss(0.015f, 0.08f, purple_abs) *
                ss(3.0f, 8.0f, p.strength);
        }
        const float red_object_strength =
            max(red_strength_guard, red_magenta_surface);
        const float natural_warm_guard_base =
            1.0f - red_object_strength * (1.0f - contextual_red_purple);
        const float natural_warm_guard_sq =
            natural_warm_guard_base * natural_warm_guard_base *
            natural_warm_guard_base * natural_warm_guard_base;
        const float natural_warm_guard =
            natural_warm_guard_sq * natural_warm_guard_sq;

        const float line_purple_relief =
            ss(0.012f, 0.06f, line_fringe_support.read(gid).r) *
            ss(0.10f,  0.24f, blue_lift);
        const float red_object_guard = ss(0.55f, 0.90f, red_object_strength);
        const float linked_override =
            linked_fringe * 1.05f * (1.0f - 0.85f * red_object_guard);
        const float line_override =
            line_purple_relief * (1.0f - 0.85f * red_object_guard);
        const bool force_warm_guard =
            (red_object_guard > 0.75f) && (linked_fringe < 0.65f) &&
            (line_purple_relief < 0.45f);
        const float warm_protect = correct_purple
            ? (force_warm_guard
                ? natural_warm_guard
                : max(natural_warm_guard,
                       max(linked_override, line_override)))
            : 1.0f;
        const float red_without_support =
            ss(0.22f, 0.45f, red_strength) *
            (1.0f - ss(0.02f, 0.12f, blur_fringe_support.read(gid).r));
        const float protect = correct_purple ? red_without_support : 0.0f;
        const float smooth_mask = ss(p.mask_cutoff,
                                       p.mask_cutoff * 5.0f, mask);
        const float threshold_soften =
            1.0f - 0.86f * ss(0.22f, 0.40f, p.chroma_threshold);
        const float src_max = max(r0, max(g0, b0));
        const float src_min = min(r0, min(g0, b0));
        const float src_chroma = src_max - src_min;
        const float src_sat = src_chroma / max(src_max, 1e-4f);
        const float neutral_guard_base = ss(0.008f, 0.030f, src_sat);
        const float neutral_guard =
            1.0f - (1.0f - neutral_guard_base) *
                    ss(0.35f, 0.80f, red_object_strength);
        float amount = clamp(smooth_mask * p.strength * 1.90f *
                              (1.0f - protect) * warm_protect * neutral_guard *
                              threshold_soften,
                              0.0f, 1.0f);
        const float warm_edge_surface =
            ss(0.006f, 0.035f, purple_abs) *
            ss(0.25f, 0.65f, red_purple) *
            ss(0.15f, 0.65f, red_purple_context) *
            (1.0f - 0.55f * red_strength_guard) *
            ss(3.0f, 8.0f, p.strength);
        const float safe_surface = max(neutral_gray_surface, warm_edge_surface);
        if (correct_purple && safe_surface > 1e-4f) {
            const float safe_amount_cap = 1.0f - 0.99f * safe_surface;
            amount = min(amount, safe_amount_cap);
        }
        if (amount > 1e-4f) {
            const float g = max(g0, 0.0f);
            float target_r = min(g, g * exp(blur_log_rg.read(gid).r));
            float target_b = min(g, g * exp(blur_log_bg.read(gid).r));

            if (correct_purple) {
                float corrected_r = clamp(r0 * (1.0f - amount) + target_r * amount,
                                           0.0f, r0);
                float corrected_g = g0;
                float corrected_b = clamp(b0 * (1.0f - amount) + target_b * amount,
                                           0.0f, b0);
                if (safe_surface > 1e-4f) {
                    const float luma_before =
                        0.2126f * r0 + 0.7152f * g0 + 0.0722f * b0;
                    const float luma_after =
                        0.2126f * corrected_r + 0.7152f * corrected_g +
                        0.0722f * corrected_b;
                    const float max_luma_drop =
                        luma_before * 0.02f * (1.0f - neutral_gray_surface);
                    const float min_luma = max(0.0f, luma_before - max_luma_drop);
                    if (luma_after < min_luma) {
                        const float lift = min_luma - luma_after;
                        const float channel_ceil = max(r0, max(g0, b0));
                        corrected_r = min(r0, corrected_r + lift);
                        corrected_g = min(channel_ceil, corrected_g + lift);
                        corrected_b = min(b0, corrected_b + lift);
                    }
                    const float src_max2 = max(r0, max(g0, b0));
                    const float src_min2 = min(r0, min(g0, b0));
                    const float corrected_max =
                        max(corrected_r, max(corrected_g, corrected_b));
                    const float corrected_min =
                        min(corrected_r, min(corrected_g, corrected_b));
                    const float src_chroma2 = src_max2 - src_min2;
                    const float corrected_chroma = corrected_max - corrected_min;
                    const float min_chroma =
                        src_chroma2 * (1.0f - 0.12f * safe_surface);
                    if (src_chroma2 > 1e-4f && corrected_chroma < min_chroma) {
                        const float denom2 = max(src_chroma2 - corrected_chroma, 1e-4f);
                        const float keep = clamp((src_chroma2 - min_chroma) / denom2,
                                                  0.0f, 1.0f);
                        corrected_r = r0 * (1.0f - keep) + corrected_r * keep;
                        corrected_g = g0 * (1.0f - keep) + corrected_g * keep;
                        corrected_b = b0 * (1.0f - keep) + corrected_b * keep;
                    }
                }
                out_r = corrected_r;
                out_g = corrected_g;
                out_b = corrected_b;
            } else {
                float target_g = g0;
                const float rg_ref = exp(blur_log_rg.read(gid).r);
                const float bg_ref = exp(blur_log_bg.read(gid).r);
                if (rg_ref > 1e-4f && bg_ref > 1e-4f) {
                    const float g_from_r = max(r0, 0.0f) / rg_ref;
                    const float g_from_b = max(b0, 0.0f) / bg_ref;
                    target_g = 0.5f * (g_from_r + g_from_b);
                }
                const float rb_avg_target = 0.5f * (max(r0, 0.0f) + max(b0, 0.0f));
                target_g = max(target_g, rb_avg_target);
                const float green_amount =
                    clamp(amount * p.green_strength_scale, 0.0f, 1.0f);
                out_r = r0;
                out_g = clamp(g0 * (1.0f - green_amount) + target_g * green_amount,
                                0.0f, g0);
                out_b = b0;
            }
        }
    }

    const float4 orig = src.read(gid);
    dst.write(float4(out_r, out_g, out_b, orig.a), gid);
}

// --------------------------------------------------------------------
// Stage 8: solo R/B cleanup pass.  Adjusts R and B (only) where the
// solo masks indicate one-sided lateral CA halos that the main path
// did not cover.  Reads the just-written `dst` colours as input
// (caller-provided ping-pong target).
kernel void defringe_solo(
    texture2d<float, access::read>  src                 [[texture(0)]],
    texture2d<float, access::read>  prev                [[texture(1)]],
    texture2d<float, access::read>  G_plane             [[texture(2)]],
    texture2d<float, access::read>  blur_log_rg         [[texture(3)]],
    texture2d<float, access::read>  blur_log_bg         [[texture(4)]],
    texture2d<float, access::read>  blur_guide          [[texture(5)]],
    texture2d<float, access::read>  solo_r_mask_blur    [[texture(6)]],
    texture2d<float, access::read>  solo_b_mask_blur    [[texture(7)]],
    texture2d<float, access::write> dst                 [[texture(8)]],
    constant DefringeParams& p                          [[buffer(0)]],
    uint2 gid                                            [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    const float solo_enable = ss(1.25f, 3.0f, p.strength);
    const float4 cur = prev.read(gid);
    float r_out = cur.r;
    float b_out = cur.b;

    if (solo_enable > 0.0f) {
        const float g = max(G_plane.read(gid).r, 0.0f);
        const float target_r = g * exp(blur_log_rg.read(gid).r);
        const float target_b = g * exp(blur_log_bg.read(gid).r);
        const float r_src = src.read(gid).r;
        const float b_src = src.read(gid).b;
        const float denom = max(blur_guide.read(gid).r, 0.05f);
        const float solo_r = max(0.0f, r_src - target_r) / denom;
        const float solo_b = max(0.0f, b_src - target_b) / denom;
        const float local_solo_r =
            ss(p.chroma_threshold * 0.12f * p.inv_sensitivity,
               p.chroma_threshold * 0.65f * p.inv_sensitivity,
               solo_r);
        const float local_solo_b =
            ss(p.chroma_threshold * 0.12f * p.inv_sensitivity,
               p.chroma_threshold * 0.65f * p.inv_sensitivity,
               solo_b);
        const float r0 = r_src;
        const float g0 = g;
        const float b0 = b_src;
        const float g_over_min = max(0.0f, g0 - min(r0, b0)) /
                                  max(blur_guide.read(gid).r, 0.05f);
        const float green_dom = ss(0.010f, 0.060f, g_over_min);
        const float denom_solo = max(blur_guide.read(gid).r, 0.05f);
        const float purpleish =
            ss(0.010f, 0.060f,
               max(max(0.0f, r0 - g0), max(0.0f, b0 - g0)) / denom_solo);
        const float solo_allow = 1.0f - green_dom * (1.0f - purpleish);

        const float solo_r_mask = solo_r_mask_blur.read(gid).r * local_solo_r * solo_allow;
        const float solo_b_mask = solo_b_mask_blur.read(gid).r * local_solo_b * solo_allow;
        if (solo_r_mask > p.mask_cutoff) {
            const float amt = clamp(solo_r_mask * p.strength * 0.85f, 0.0f, 0.85f);
            r_out = clamp(r_out * (1.0f - amt) + target_r * amt, 0.0f, r_out);
        }
        if (solo_b_mask > p.mask_cutoff) {
            const float amt = clamp(solo_b_mask * p.strength * 0.95f, 0.0f, 0.90f);
            b_out = clamp(b_out * (1.0f - amt) + target_b * amt, 0.0f, b_out);
        }
    }
    dst.write(float4(r_out, cur.g, b_out, cur.a), gid);
}
