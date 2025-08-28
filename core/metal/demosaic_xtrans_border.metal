//
// demosaic_xtrans_border.metal
// X-Trans border interpolation - CPU identical implementation
//

#include <metal_stdlib>
#include "shader_types.h"
#include "shader_common.h"

using namespace metal;

kernel void demosaic_xtrans_border(
    device float* rgb_data [[buffer(0)]],
    const device ushort* raw_data [[buffer(1)]],
    constant XTransParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = params.width;
    uint height = params.height;
    uint border = params.border_width;
    uint row = gid.y;
    uint col = gid.x;

    if (row >= height || col >= width) return;

    // Check if this pixel is in border region
    bool is_border = (row < border) || (row >= height - border) || 
                    (col < border) || (col >= width - border);
    if (!is_border) return;

    // LibRaw-style weight matrix for X-Trans border interpolation
    const float weight[3][3] = {
        {0.25f, 0.5f, 0.25f},
        {0.5f, 0.0f, 0.5f},
        {0.25f, 0.5f, 0.25f}
    };

    // Sum arrays: [R, G, B, R_weight, G_weight, B_weight]
    float sum[6] = {0.0f};

    // 3x3 neighborhood scan (CPU-equivalent logic)
    for (int y_offset = -1; y_offset <= 1; ++y_offset) {
        for (int x_offset = -1; x_offset <= 1; ++x_offset) {
            
            // Clamp coordinates to stay within image bounds, matching CPU behavior
            int y = clamp(int(row) + y_offset, 0, int(height) - 1);
            int x = clamp(int(col) + x_offset, 0, int(width) - 1);
            
            uint f = fcol_xtrans(y, x, params.xtrans);
            uint raw_idx = y * width + x;
            float raw_val = float(raw_data[raw_idx]) / (float)params.maximum_value;
            float w = weight[y_offset + 1][x_offset + 1];
            
            sum[f] += raw_val * w;
            sum[f + 3] += w;
        }
    }
    
    uint out_idx = (row * width + col) * 3;
    uint pixel_color = fcol_xtrans(row, col, params.xtrans);
    
    // LibRaw's exact interpolation logic for border pixels
    switch(pixel_color) {
        case 0: // Red pixel
            // Red channel gets the raw value.
            rgb_data[out_idx + 0] = float(raw_data[row * width + col]) / (float)params.maximum_value;
            // Green and Blue are interpolated using the summed raw data.
            rgb_data[out_idx + 1] = (sum[4] > 0.0f) ? (sum[1] / sum[4]) : 0.0f;
            rgb_data[out_idx + 2] = (sum[5] > 0.0f) ? (sum[2] / sum[5]) : 0.0f;
            break;
            
        case 1: // Green pixel
            if (sum[3] == 0.0f) {
                // This is a special case for pixels on the edge of the mosaic pattern.
                float green_val = float(raw_data[row * width + col]) / (float)params.maximum_value;
                rgb_data[out_idx + 0] = green_val;
                rgb_data[out_idx + 1] = green_val;
                rgb_data[out_idx + 2] = green_val;
            } else {
                // Red and Blue are interpolated using the summed raw data.
                rgb_data[out_idx + 0] = (sum[0] / sum[3]);
                // Green channel gets the raw value.
                rgb_data[out_idx + 1] = float(raw_data[row * width + col]) / (float)params.maximum_value;
                rgb_data[out_idx + 2] = (sum[2] / sum[5]);
            }
            break;
            
        case 2: // Blue pixel
            // Red and Green are interpolated using the summed raw data.
            rgb_data[out_idx + 0] = (sum[3] > 0.0f) ? (sum[0] / sum[3]) : 0.0f;
            rgb_data[out_idx + 1] = (sum[4] > 0.0f) ? (sum[1] / sum[4]) : 0.0f;
            // Blue channel gets the raw value.
            rgb_data[out_idx + 2] = float(raw_data[row * width + col]) / (float)params.maximum_value;
            break;
    }
}