//
// demosaic_bayer_linear.metal
// LibRaw Enhanced - Simple Bilinear Interpolation (CPU-exact port)
//
#include "shader_types.h"
#include "shader_common.h"

kernel void demosaic_bayer_linear(
    const device float* raw_data [[buffer(0)]],
    device float* rgb_data [[buffer(1)]],
    constant DemosaicBayerParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = params.width;
    const uint height = params.height;
    const uint row = gid.y;
    const uint col = gid.x;
    const uint filters = params.filters;
    
    if (row >= height || col >= width) return;
    
    const uint raw_idx = row * width + col;
    const uint rgb_idx = raw_idx * 3;
    const uint pixel_color = fcol_bayer(row, col, filters);
    
    const float raw_val = raw_data[raw_idx] / params.maximum_value;
    
    rgb_data[rgb_idx + 0] = 0.0f;
    rgb_data[rgb_idx + 1] = 0.0f;
    rgb_data[rgb_idx + 2] = 0.0f;
    rgb_data[rgb_idx + pixel_color] = raw_val;
    
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1) {
        for (uint c = 0; c < 3; c++) {
            if (c == pixel_color) continue;
            
            float sum = 0.0f;
            int count = 0;
            
            // Try diagonal neighbors first
            int dy_diag[4] = {-1, -1, 1, 1};
            int dx_diag[4] = {-1, 1, -1, 1};
            for (int i = 0; i < 4; i++) {
                uint ny = row + dy_diag[i];
                uint nx = col + dx_diag[i];
                if (fcol_bayer(ny, nx, filters) == c) {
                    sum += raw_data[ny * width + nx] / params.maximum_value;
                    count++;
                }
            }
            
            // If none found, check all 8 neighbors (CPU-exact logic)
            if (count == 0) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dy == 0 && dx == 0) continue;
                        uint ny = row + dy;
                        uint nx = col + dx;
                        if (fcol_bayer(ny, nx, filters) == c) {
                            sum += raw_data[ny * width + nx] / params.maximum_value;
                            count++;
                        }
                    }
                }
            }
            
            if (count > 0) {
                rgb_data[rgb_idx + c] = sum / float(count);
            }
        }
    }
}