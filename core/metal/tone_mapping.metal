
#include "shader_types.h"
#include "shader_common.h"

inline float3 aces_tone_map(float3 x) {
    constexpr float3 a = float3(2.51f);
    constexpr float3 b = float3(0.03f);
    constexpr float3 c = float3(2.43f);
    constexpr float3 d = float3(0.59f);
    constexpr float3 e = float3(0.14f);
    
    return (x * (a * x + b)) / (x * (c * x + d) + e);
};

kernel void tone_mapping(
    const device float* rgb_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant ToneMappingParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint pixel_idx = (gid.y * params.width + gid.x) * 3;
    
    float3 rgb_in = float3(
        rgb_input[pixel_idx + 0],
        rgb_input[pixel_idx + 1],
        rgb_input[pixel_idx + 2]
    );
    
    float3 rgb_out = aces_tone_map(rgb_in) * params.after_scale;

    // 結果を書き込み
    //rgb_out = clamp(rgb_out, 0.0f, 1.0f);
    rgb_output[pixel_idx + 0] = rgb_out.r;
    rgb_output[pixel_idx + 1] = rgb_out.g;
    rgb_output[pixel_idx + 2] = rgb_out.b;
}
