//
// demosaic_bayer_border.metal
// LibRaw Enhanced - Bayer Border Interpolation (CPU identical implementation)
//

#include "shader_types.h"
#include "shader_common.h"

// Bayer border interpolation
kernel void demosaic_bayer_border(
    device float* rgb_data [[buffer(0)]],
    constant DemosaicBayerParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = params.width;
    uint height = params.height;
    uint filters = params.filters;
    uint border = params.border_width;
    uint row = gid.y;
    uint col = gid.x;

    if (row >= height || col >= width) return;

    // Check if this pixel is in border region (LibRaw logic)
    bool is_border = (row < border) || (row >= height - border) || 
                    (col < border) || (col >= width - border);
    if (!is_border) return;

    uint rgb_idx = (row * width + col) * 3;
    
    // LibRaw-style sum array: [R, G, B, R_count, G_count, B_count]
    float sum[6] = {0.0f};
    
    // 3x3 neighborhood scan (LibRaw method)
    for (int y_offset = -1; y_offset <= 1; ++y_offset) {
        for (int x_offset = -1; x_offset <= 1; ++x_offset) {
            int y = int(row) + y_offset;
            int x = int(col) + x_offset;

            if (y >= 0 && y < int(height) && x >= 0 && x < int(width)) {
                uint f = fcol_bayer(y, x, filters);
                uint neighbor_idx = (y * width + x) * 3;
                
                // Accumulate the native color pixel value
                sum[f] += rgb_data[neighbor_idx + f];
                sum[f + 3] += 1.0f;  // Count
            }
        }
    }
    
    // Current position's native color
    uint f_center = fcol_bayer(row, col, filters);
    
    // Interpolate only non-native colors (LibRaw method)
    for (uint c = 0; c < 3; ++c) {
        if (c != f_center && sum[c + 3] > 0) {
            rgb_data[rgb_idx + c] = sum[c] / sum[c + 3];
        }
    }
}