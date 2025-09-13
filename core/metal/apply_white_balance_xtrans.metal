//
// apply_white_balance.metal
// LibRaw Enhanced - White Balance Application
// GPU port of cpu_accelerator.cpp apply_white_balance
//
#include "shader_types.h"
#include "shader_common.h"

kernel void apply_white_balance_xtrans(
    const device ushort* raw_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant ApplyWhiteBalanceParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint input_idx = (gid.y * params.width + gid.x) * 4;
    const uint output_idx = (gid.y * params.width + gid.x) * 3;
    const uint32_t row = gid.y;
    const uint32_t col = gid.x;
    
    // Get color channel using X-Trans pattern
    const int color_channel = fcol_xtrans(row, col, params.xtrans);
    
    // Apply white balance multiplier to the native color channel
    float adjusted_value = raw_input[input_idx + color_channel] * params.multipliers[color_channel];
    
    rgb_output[output_idx + color_channel] = adjusted_value;
}
