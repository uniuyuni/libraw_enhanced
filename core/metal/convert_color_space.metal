//
// color_space_convert.metal
// LibRaw Enhanced - Color Space Conversion
// GPU port of cpu_accelerator.cpp convert_color_space
//
#include "shader_types.h"
#include "shader_common.h"

kernel void convert_color_space(
    const device float* rgb_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant ColorSpaceParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint pixel_count = params.width * params.height;
    
    if (gid >= pixel_count) return;
    
    const uint pixel_idx = gid * 3;
    
    // Get input RGB values
    float3 rgb_in = {
        rgb_input[pixel_idx + 0],
        rgb_input[pixel_idx + 1], 
        rgb_input[pixel_idx + 2]
    };
    
    // Apply 3x4 transformation matrix (exact CPU algorithm)
    float3 rgb_out = {
        params.transform[0][0] * rgb_in.r + params.transform[0][1] * rgb_in.g + params.transform[0][2] * rgb_in.b + params.transform[0][3],
        params.transform[1][0] * rgb_in.r + params.transform[1][1] * rgb_in.g + params.transform[1][2] * rgb_in.b + params.transform[1][3],
        params.transform[2][0] * rgb_in.r + params.transform[2][1] * rgb_in.g + params.transform[2][2] * rgb_in.b + params.transform[2][3]
    };
    
    // Clamp to [0.0, 1.0] range (exact CPU clamping)
    rgb_output[pixel_idx + 0] = clamp(rgb_out.r, 0.0f, 1.0f);
    rgb_output[pixel_idx + 1] = clamp(rgb_out.g, 0.0f, 1.0f);
    rgb_output[pixel_idx + 2] = clamp(rgb_out.b, 0.0f, 1.0f);
}

kernel void convert_color_space_3x3(
    const device float* rgb_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant ColorSpaceParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint pixel_count = params.width * params.height;
    
    if (gid >= pixel_count) return;
    
    const uint pixel_idx = gid * 3;
    
    // Get input RGB values
    float3 rgb_in = {
        rgb_input[pixel_idx + 0],
        rgb_input[pixel_idx + 1], 
        rgb_input[pixel_idx + 2]
    };
    
    // Apply 3x3 transformation matrix
    float3 rgb_out = {
        params.transform[0][0] * rgb_in.r + params.transform[0][1] * rgb_in.g + params.transform[0][2] * rgb_in.b,
        params.transform[1][0] * rgb_in.r + params.transform[1][1] * rgb_in.g + params.transform[1][2] * rgb_in.b,
        params.transform[2][0] * rgb_in.r + params.transform[2][1] * rgb_in.g + params.transform[2][2] * rgb_in.b
    };
    
    // Clamp to [0.0, 1.0] range
    rgb_output[pixel_idx + 0] = clamp(rgb_out.r, 0.0f, 1.0f);
    rgb_output[pixel_idx + 1] = clamp(rgb_out.g, 0.0f, 1.0f);
    rgb_output[pixel_idx + 2] = clamp(rgb_out.b, 0.0f, 1.0f);
}