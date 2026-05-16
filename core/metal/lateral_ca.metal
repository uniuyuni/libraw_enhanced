//
// lateral_ca.metal
// Lateral chromatic aberration registration (post-demosaic dense RGB) on
// the GPU.  Mirrors CPUAccelerator::ca_register_lateral 1:1:
//   1) split RGB into R/G/B planes
//   2) build pyramid via 2x box-mean downsample
//   3) per-cell pyramidal Lucas-Kanade (G as reference, R or B as target)
//   4) bilinear-interpolated per-pixel shifts applied to R, B (G untouched)
// Per-level scaling, confidence gating, and 3x3 smoothing on the (small)
// shift maps are handled on the CPU side — those passes touch only
// O(map_w * map_h) cells.
//

#include <metal_stdlib>
#include "shader_types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Manual bilinear sample of an R32Float texture by (x, y) in level-local
// pixel coordinates.  Clamps to image bounds, matching the CPU
// `bilinear_sample_plane` helper exactly.
static inline float lca_bilinear(texture2d<float, access::read> tex,
                                 uint W, uint H,
                                 float x, float y) {
    x = clamp(x, 0.0f, float(W - 1));
    y = clamp(y, 0.0f, float(H - 1));
    const uint x0 = (uint)floor(x);
    const uint y0 = (uint)floor(y);
    const uint x1 = min(x0 + 1u, W - 1u);
    const uint y1 = min(y0 + 1u, H - 1u);
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const float v00 = tex.read(uint2(x0, y0)).r;
    const float v01 = tex.read(uint2(x1, y0)).r;
    const float v10 = tex.read(uint2(x0, y1)).r;
    const float v11 = tex.read(uint2(x1, y1)).r;
    return (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy)
         + (v10 * (1.0f - fx) + v11 * fx) * fy;
}

// ---------------------------------------------------------------------------
// 1) Split RGBA → R, G, B planes (level 0).
// ---------------------------------------------------------------------------
kernel void lateral_ca_split(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> R   [[texture(1)]],
    texture2d<float, access::write> G   [[texture(2)]],
    texture2d<float, access::write> B   [[texture(3)]],
    constant uint2&                 dim [[buffer(0)]],
    uint2 gid                            [[thread_position_in_grid]])
{
    if (gid.x >= dim.x || gid.y >= dim.y) return;
    const float4 c = src.read(gid);
    R.write(c.r, gid);
    G.write(c.g, gid);
    B.write(c.b, gid);
}

// ---------------------------------------------------------------------------
// 2) 2x box-mean downsample of a single-channel plane.  Mirrors
//    downsample_2x_box() exactly.  Output dimensions are floor(W/2),
//    floor(H/2).
// ---------------------------------------------------------------------------
kernel void lateral_ca_downsample(
    texture2d<float, access::read>  in    [[texture(0)]],
    texture2d<float, access::write> outp  [[texture(1)]],
    constant LateralCaDownsampleParams& p [[buffer(0)]],
    uint2 gid                              [[thread_position_in_grid]])
{
    if (gid.x >= p.W_out || gid.y >= p.H_out) return;
    const uint x0 = 2u * gid.x;
    const uint y0 = 2u * gid.y;
    const uint x1 = min(x0 + 1u, p.W_in - 1u);
    const uint y1 = min(y0 + 1u, p.H_in - 1u);
    const float a = in.read(uint2(x0, y0)).r;
    const float b = in.read(uint2(x1, y0)).r;
    const float c = in.read(uint2(x0, y1)).r;
    const float d = in.read(uint2(x1, y1)).r;
    outp.write(0.25f * (a + b + c + d), gid);
}

// ---------------------------------------------------------------------------
// 3) Per-cell Lucas-Kanade estimation at a single pyramid level.
//    One thread = one cell; the thread loops over cell pixels itself to
//    accumulate the 2x2 normal equation, then iterates up to
//    max_iterations times.  dx_map / dy_map are both input (initial
//    hint) and output (refined shift); conf_map receives the smallest
//    eigenvalue of the structure tensor as a coarse confidence proxy.
// ---------------------------------------------------------------------------
kernel void lateral_ca_lk_estimate(
    texture2d<float, access::read> G_tex      [[texture(0)]],
    texture2d<float, access::read> T_tex      [[texture(1)]],
    device float*                  dx_map     [[buffer(0)]],
    device float*                  dy_map     [[buffer(1)]],
    device float*                  conf_map   [[buffer(2)]],
    constant LateralCaLkParams&    p          [[buffer(3)]],
    uint2 gid                                  [[thread_position_in_grid]])
{
    if (gid.x >= p.map_w || gid.y >= p.map_h) return;
    const uint cell_idx = gid.y * p.map_w + gid.x;

    const uint x0 = (uint)(float(gid.x) * p.cell_w_f);
    const uint y0 = (uint)(float(gid.y) * p.cell_h_f);
    uint x1 = (uint)(float(gid.x + 1u) * p.cell_w_f);
    uint y1 = (uint)(float(gid.y + 1u) * p.cell_h_f);
    if (x1 > p.W) x1 = p.W;
    if (y1 > p.H) y1 = p.H;
    if (x1 <= x0 + 4u || y1 <= y0 + 4u) return;

    float accum_dx = dx_map[cell_idx];
    float accum_dy = dy_map[cell_idx];
    float min_eig_final = 0.0f;

    for (uint iter = 0; iter < p.max_iterations; iter++) {
        float A00 = 0.0f, A01 = 0.0f, A11 = 0.0f;
        float bx  = 0.0f, by  = 0.0f;
        uint  valid = 0;

        for (uint y = y0 + 1u; y + 1u < y1; y++) {
            for (uint x = x0 + 1u; x + 1u < x1; x++) {
                const float gx = G_tex.read(uint2(x + 1u, y)).r
                               - G_tex.read(uint2(x - 1u, y)).r;
                const float gy = G_tex.read(uint2(x, y + 1u)).r
                               - G_tex.read(uint2(x, y - 1u)).r;
                const float t_warp = lca_bilinear(T_tex, p.W, p.H,
                                                  float(x) + accum_dx,
                                                  float(y) + accum_dy);
                const float g_ref = G_tex.read(uint2(x, y)).r;
                const float err   = t_warp - g_ref;

                A00 += gx * gx;
                A01 += gx * gy;
                A11 += gy * gy;
                bx  -= gx * err;
                by  -= gy * err;
                valid++;
            }
        }
        if (valid < 32u) break;

        const float det   = A00 * A11 - A01 * A01;
        const float trace = A00 + A11;
        const float disc  = max(0.0f, trace * trace - 4.0f * det);
        const float sq    = sqrt(disc);
        min_eig_final     = 0.5f * (trace - sq);
        if (det <= 1e-12f) break;
        const float inv    = 1.0f / det;
        const float step_x = inv * ( A11 * bx - A01 * by);
        const float step_y = inv * (-A01 * bx + A00 * by);
        accum_dx += step_x;
        accum_dy += step_y;
        accum_dx = clamp(accum_dx, -p.max_shift, p.max_shift);
        accum_dy = clamp(accum_dy, -p.max_shift, p.max_shift);
        if (fabs(step_x) < 0.005f && fabs(step_y) < 0.005f) break;
    }

    dx_map[cell_idx] = accum_dx;
    dy_map[cell_idx] = accum_dy;
    conf_map[cell_idx] = max(0.0f, min_eig_final);
}

// ---------------------------------------------------------------------------
// 4) Per-pixel apply: bilinear-interpolate shift maps to (x, y), clamp,
//    bilinear-sample R0 / B0 at (x + dx, y + dy), pass G through.
//    Mirrors the CPU apply loop and `sample_shift` helper.
// ---------------------------------------------------------------------------

static inline float lca_sample_map(device const float* m,
                                    uint map_w, uint map_h,
                                    int gx0, int gy0,
                                    int gx1, int gy1,
                                    float fx, float fy) {
    const float v00 = m[(uint)gy0 * map_w + (uint)gx0];
    const float v01 = m[(uint)gy0 * map_w + (uint)gx1];
    const float v10 = m[(uint)gy1 * map_w + (uint)gx0];
    const float v11 = m[(uint)gy1 * map_w + (uint)gx1];
    return (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy)
         + (v10 * (1.0f - fx) + v11 * fx) * fy;
}

kernel void lateral_ca_apply(
    texture2d<float, access::read>  R0_tex [[texture(0)]],
    texture2d<float, access::read>  G0_tex [[texture(1)]],
    texture2d<float, access::read>  B0_tex [[texture(2)]],
    texture2d<float, access::write> dst    [[texture(3)]],
    device const float*             dx_r   [[buffer(0)]],
    device const float*             dy_r   [[buffer(1)]],
    device const float*             dx_b   [[buffer(2)]],
    device const float*             dy_b   [[buffer(3)]],
    constant LateralCaApplyParams&  p      [[buffer(4)]],
    uint2 gid                                [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    const float px = float(gid.x);
    const float py = float(gid.y);

    const float gx_f = (px / p.cell_size) - 0.5f;
    const float gy_f = (py / p.cell_size) - 0.5f;
    int gx0 = (int)floor(gx_f);
    int gy0 = (int)floor(gy_f);
    if (gx0 < 0) gx0 = 0;
    if (gy0 < 0) gy0 = 0;
    if (gx0 >= (int)p.map_w - 1) gx0 = (int)p.map_w - 2;
    if (gy0 >= (int)p.map_h - 1) gy0 = (int)p.map_h - 2;
    if (gx0 < 0) gx0 = 0;
    if (gy0 < 0) gy0 = 0;
    const float fx = clamp(gx_f - float(gx0), 0.0f, 1.0f);
    const float fy = clamp(gy_f - float(gy0), 0.0f, 1.0f);
    const int gx1 = min(gx0 + 1, (int)p.map_w - 1);
    const int gy1 = min(gy0 + 1, (int)p.map_h - 1);

    const float drx_raw = lca_sample_map(dx_r, p.map_w, p.map_h,
                                          gx0, gy0, gx1, gy1, fx, fy);
    const float dry_raw = lca_sample_map(dy_r, p.map_w, p.map_h,
                                          gx0, gy0, gx1, gy1, fx, fy);
    const float dbx_raw = lca_sample_map(dx_b, p.map_w, p.map_h,
                                          gx0, gy0, gx1, gy1, fx, fy);
    const float dby_raw = lca_sample_map(dy_b, p.map_w, p.map_h,
                                          gx0, gy0, gx1, gy1, fx, fy);

    const float drx = clamp(drx_raw, -p.clamp_shift, p.clamp_shift);
    const float dry = clamp(dry_raw, -p.clamp_shift, p.clamp_shift);
    const float dbx = clamp(dbx_raw, -p.clamp_shift, p.clamp_shift);
    const float dby = clamp(dby_raw, -p.clamp_shift, p.clamp_shift);

    const float r_new = lca_bilinear(R0_tex, p.width, p.height,
                                      px + drx, py + dry);
    const float b_new = lca_bilinear(B0_tex, p.width, p.height,
                                      px + dbx, py + dby);
    const float g_pass = G0_tex.read(gid).r;

    dst.write(float4(r_new, g_pass, b_new, 0.0f), gid);
}
