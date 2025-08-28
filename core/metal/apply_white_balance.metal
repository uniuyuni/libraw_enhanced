//
// apply_white_balance.metal
// LibRaw Enhanced - White Balance Application
// GPU port of cpu_accelerator.cpp apply_white_balance
//
#include "shader_types.h"
#include "shader_common.h"

kernel void apply_white_balance(
    const device float* rgb_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant WhiteBalanceParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint pixel_count = params.width * params.height;
    
    if (gid >= pixel_count) return;
    
    const uint pixel_idx = gid * 3;
    
    // Apply white balance multipliers (exact CPU algorithm)
    float r = rgb_input[pixel_idx + 0] * params.multipliers[0];
    float g = rgb_input[pixel_idx + 1] * params.multipliers[1];
    float b = rgb_input[pixel_idx + 2] * params.multipliers[2];
    
    // Clamp to [0.0, 1.0] range
    rgb_output[pixel_idx + 0] = clamp(r, 0.0f, 1.0f);
    rgb_output[pixel_idx + 1] = clamp(g, 0.0f, 1.0f);
    rgb_output[pixel_idx + 2] = clamp(b, 0.0f, 1.0f);
}

kernel void apply_white_balance_uint16(
    const device uint16_t* rgb_input [[buffer(0)]],
    device uint16_t* rgb_output [[buffer(1)]],
    constant WhiteBalanceParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint pixel_count = params.width * params.height;
    
    if (gid >= pixel_count) return;
    
    const uint pixel_idx = gid * 3;
    
    // Apply white balance multipliers
    float r = float(rgb_input[pixel_idx + 0]) * params.multipliers[0];
    float g = float(rgb_input[pixel_idx + 1]) * params.multipliers[1];
    float b = float(rgb_input[pixel_idx + 2]) * params.multipliers[2];
    
    // Direct conversion without clamping (overflow handled by uint16_t cast)
    rgb_output[pixel_idx + 0] = uint16_t(r);
    rgb_output[pixel_idx + 1] = uint16_t(g);
    rgb_output[pixel_idx + 2] = uint16_t(b);
}