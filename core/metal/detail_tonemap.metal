//
// detail_tonemap.metal
// GPU port of LibRawWrapper::apply_detail_preserving_tonemap.
//
// Pipeline (matches the CPU reference 1:1):
//   1) per-pixel: guide = max(R, G, B); work = guide²
//   2) box(r=12) on guide → mean
//      box(r=12) on work  → mean_sq  (MPSImageBox)
//   3) per-pixel: var = max(0, mean_sq − mean²);
//                 a   = var / (var + eps);
//                 b   = mean · (1 − a)
//   4) box(r=12) on a → a_smooth
//      box(r=12) on b → b_smooth   (MPSImageBox)
//   5) per-pixel: base = a_smooth·guide + b_smooth;
//                 gain = ACES(base) / base (clamped to [0,1]);
//                 w    = smoothstep(edge0, edge1, base);
//                 RGB *= 1 + (gain − 1) · w
//

#include <metal_stdlib>
#include "shader_types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Stage 1: split RGB → guide (max channel) and work (guide²).
// ---------------------------------------------------------------------------
kernel void detail_tonemap_setup(
    texture2d<float, access::read>  src   [[texture(0)]],
    texture2d<float, access::write> guide [[texture(1)]],
    texture2d<float, access::write> work  [[texture(2)]],
    constant DetailTonemapParams&   p     [[buffer(0)]],
    uint2 gid                              [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    const float4 c = src.read(gid);
    const float g = max(0.0f, max(c.r, max(c.g, c.b)));
    guide.write(g,     gid);
    work.write(g * g,  gid);
}

// ---------------------------------------------------------------------------
// Stage 3: compute guided-filter regression coefficients per pixel.
//   a = var / (var + eps), b = mean · (1 − a)
// ---------------------------------------------------------------------------
kernel void detail_tonemap_ab(
    texture2d<float, access::read>  mean    [[texture(0)]],
    texture2d<float, access::read>  mean_sq [[texture(1)]],
    texture2d<float, access::write> a_out   [[texture(2)]],
    texture2d<float, access::write> b_out   [[texture(3)]],
    constant DetailTonemapParams&   p       [[buffer(0)]],
    uint2 gid                                [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    const float m  = mean.read(gid).r;
    const float ms = mean_sq.read(gid).r;
    const float var = max(0.0f, ms - m * m);
    const float a   = var / (var + p.eps);
    const float b   = m * (1.0f - a);
    a_out.write(a, gid);
    b_out.write(b, gid);
}

// ---------------------------------------------------------------------------
// Stage 5: per-pixel tone-mapped gain application.
// ---------------------------------------------------------------------------
static inline float aces_tone_map(float x) {
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;
    return (x * (A * x + B)) / (x * (C * x + D) + E);
}

kernel void detail_tonemap_apply(
    texture2d<float, access::read>  src       [[texture(0)]],
    texture2d<float, access::read>  guide     [[texture(1)]],
    texture2d<float, access::read>  a_smooth  [[texture(2)]],
    texture2d<float, access::read>  b_smooth  [[texture(3)]],
    texture2d<float, access::write> dst       [[texture(4)]],
    constant DetailTonemapParams&   p         [[buffer(0)]],
    uint2 gid                                  [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    const float4 c  = src.read(gid);
    const float g   = guide.read(gid).r;
    const float a   = a_smooth.read(gid).r;
    const float b   = b_smooth.read(gid).r;

    const float base = max(1e-6f, a * g + b);
    const float gain_raw = aces_tone_map(base) / base;
    const float gain = clamp(gain_raw, 0.0f, 1.0f);

    float w = (base - p.edge0) / (p.edge1 - p.edge0);
    w = clamp(w, 0.0f, 1.0f);
    w = w * w * (3.0f - 2.0f * w);

    const float final_gain = 1.0f + (gain - 1.0f) * w;
    dst.write(float4(c.r * final_gain,
                     c.g * final_gain,
                     c.b * final_gain,
                     c.a),
              gid);
}
