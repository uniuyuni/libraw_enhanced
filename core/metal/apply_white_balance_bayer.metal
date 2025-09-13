//
// apply_white_balance.metal
// LibRaw Enhanced - White Balance Application
// GPU port of cpu_accelerator.cpp apply_white_balance
//
#include "shader_types.h"
#include "shader_common.h"

kernel void apply_white_balance_bayer(
    const device ushort* raw_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant ApplyWhiteBalanceParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint input_idx = (gid.y * params.width + gid.x) * 4;
    const uint output_idx = (gid.y * params.width + gid.x) * 3;
    const uint row = gid.y;
    const uint col = gid.x;
    
    // Get color channel for this pixel position using LibRaw's fcol logic
    //int color_channel = (params.filters >> ((((row) << 1 & 14) | ((col) & 1)) << 1)) & 3;
    const int color_channel = fcol_bayer(row, col, params.filters);
    
    // Apply white balance multiplier to the native color channel
    float adjusted_value = raw_input[input_idx + color_channel] * params.multipliers[color_channel];

    rgb_output[output_idx + color_channel] = adjusted_value;
}
