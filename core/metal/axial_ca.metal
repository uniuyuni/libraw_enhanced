//
// axial_ca.metal
// Axial-CA cleanup support kernels.  MPS provides the heavy lifting
// (MPSImageBox for the box-filter passes, MPSImageSobel for the edge
// magnitude); these small shaders fill in the algorithm-specific
// element-wise steps.
//

#include <metal_stdlib>
#include "shader_types.h"
using namespace metal;

// Pass 1: normalise the input RGB by inv_norm = 1/max(G) and split into
// single-channel R, G, B planes, plus the precomputed products needed
// by the guided-filter regression (G², R·G, B·G).
//
// Doing all six writes in one kernel avoids re-reading the source
// texture six times.
kernel void axial_ca_prepare(
    texture2d<float, access::read>  src   [[texture(0)]],
    texture2d<float, access::write> R     [[texture(1)]],
    texture2d<float, access::write> G     [[texture(2)]],
    texture2d<float, access::write> B     [[texture(3)]],
    texture2d<float, access::write> GG    [[texture(4)]],
    texture2d<float, access::write> RG    [[texture(5)]],
    texture2d<float, access::write> BG    [[texture(6)]],
    constant AxialCaPrepareParams& params [[buffer(0)]],
    uint2 gid                              [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;
    const float4 c = src.read(gid);
    const float r = c.r * params.inv_norm;
    const float g = c.g * params.inv_norm;
    const float b = c.b * params.inv_norm;
    R.write(r,       gid);
    G.write(g,       gid);
    B.write(b,       gid);
    GG.write(g * g,  gid);
    RG.write(r * g,  gid);
    BG.write(b * g,  gid);
}

// Compute the guided-filter (a, b) coefficients per pixel from the four
// box-mean inputs.  Used once for the R-vs-G regression and once for B.
//
//   var_I  = mean_II − mean_I²
//   cov_Ip = mean_Ip − mean_I · mean_p
//   a      = cov_Ip / (var_I + ε)
//   b      = mean_p − a · mean_I
kernel void axial_ca_compute_ab(
    texture2d<float, access::read>  mean_I  [[texture(0)]],
    texture2d<float, access::read>  mean_II [[texture(1)]],
    texture2d<float, access::read>  mean_p  [[texture(2)]],
    texture2d<float, access::read>  mean_Ip [[texture(3)]],
    texture2d<float, access::write> a_out   [[texture(4)]],
    texture2d<float, access::write> b_out   [[texture(5)]],
    constant AxialCaAbParams& params        [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;
    const float mI  = mean_I.read(gid).r;
    const float mII = mean_II.read(gid).r;
    const float mp  = mean_p.read(gid).r;
    const float mIp = mean_Ip.read(gid).r;
    const float var_I  = max(0.0f, mII - mI * mI);
    const float cov_Ip = mIp - mI * mp;
    const float a = cov_Ip / (var_I + params.epsilon);
    const float b = mp - a * mI;
    a_out.write(a, gid);
    b_out.write(b, gid);
}

// Apply the (smoothed) (a, b) coefficients to the guide:
//   q(p) = mean_a(p) · I(p) + mean_b(p)
//
// Used once for each of R and B.
kernel void axial_ca_apply(
    texture2d<float, access::read>  I       [[texture(0)]],
    texture2d<float, access::read>  mean_a  [[texture(1)]],
    texture2d<float, access::read>  mean_b  [[texture(2)]],
    texture2d<float, access::write> q_out   [[texture(3)]],
    constant uint2& dims                    [[buffer(0)]],
    uint2 gid                                [[thread_position_in_grid]])
{
    if (gid.x >= dims.x || gid.y >= dims.y) return;
    const float i  = I.read(gid).r;
    const float ma = mean_a.read(gid).r;
    const float mb = mean_b.read(gid).r;
    q_out.write(ma * i + mb, gid);
}

// Per-pixel gradient magnitude on a single-channel R32Float plane using
// the same 2-tap finite difference the CPU reference implementation
// uses (so MPS's 3x3 Sobel doesn't subtly shift edge weights).
kernel void axial_ca_grad_g(
    texture2d<float, access::read>  G       [[texture(0)]],
    texture2d<float, access::write> grad    [[texture(1)]],
    constant uint2& dims                    [[buffer(0)]],
    uint2 gid                                [[thread_position_in_grid]])
{
    if (gid.x >= dims.x || gid.y >= dims.y) return;
    if (gid.x == 0 || gid.y == 0 || gid.x + 1 >= dims.x || gid.y + 1 >= dims.y) {
        grad.write(0.0f, gid);
        return;
    }
    const float gx = G.read(uint2(gid.x + 1, gid.y    )).r
                   - G.read(uint2(gid.x - 1, gid.y    )).r;
    const float gy = G.read(uint2(gid.x    , gid.y + 1)).r
                   - G.read(uint2(gid.x    , gid.y - 1)).r;
    grad.write(sqrt(gx * gx + gy * gy), gid);
}

// Edge-gated blend.  The smoothed-gradient texture is interpreted via a
// smoothstep into a per-pixel weight; the output replaces R and B with a
// blend toward their filtered versions, scaled by the user-supplied
// `strength` and the edge weight.  G is passed through unchanged.
// Finally we de-normalise back to the original linear-RGB scale.
kernel void axial_ca_blend(
    texture2d<float, access::read>  src         [[texture(0)]],
    texture2d<float, access::read>  R_norm      [[texture(1)]],
    texture2d<float, access::read>  B_norm      [[texture(2)]],
    texture2d<float, access::read>  R_filt      [[texture(3)]],
    texture2d<float, access::read>  B_filt      [[texture(4)]],
    texture2d<float, access::read>  edge_smooth [[texture(5)]],
    texture2d<float, access::write> dst         [[texture(6)]],
    constant AxialCaBlendParams& params         [[buffer(0)]],
    uint2 gid                                    [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;
    const float e = edge_smooth.read(gid).r;
    const float t = clamp((e - params.edge_lo) / (params.edge_hi - params.edge_lo),
                          0.0f, 1.0f);
    const float w = t * t * (3.0f - 2.0f * t);
    const float m = params.strength * w;

    const float r  = R_norm.read(gid).r;
    const float b  = B_norm.read(gid).r;
    const float rf = R_filt.read(gid).r;
    const float bf = B_filt.read(gid).r;
    const float4 orig = src.read(gid);

    const float r_out = (r * (1.0f - m) + rf * m) * params.norm_ref;
    const float b_out = (b * (1.0f - m) + bf * m) * params.norm_ref;
    // G stays exactly as the source had it (no float drift via norm/de-norm).
    dst.write(float4(r_out, orig.g, b_out, 0.0f), gid);
}
