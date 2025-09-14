//
// demosaic_xtrans_1pass.metal
// X-Trans 1-pass demosaic - CPU identical implementation
//

#include "shader_types.h"
#include "shader_common.h"

kernel void demosaic_xtrans_1pass(
    const device float* raw_data [[buffer(0)]],
    device float* rgb_data [[buffer(1)]],
    constant DemosaicXTransParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = params.width;
    uint height = params.height;
    uint row = gid.y;
    uint col = gid.x;
    
    if (row < 1 || row >= height - 1 || col < 1 || col >= width - 1) return;
    
    uint idx = row * width + col;
    uint pixel_color = fcol_xtrans(row, col, params.xtrans);
    uint out_idx = idx * 3;
    
    // RawTherapee's exact weight matrix for 1-pass
    const float weight[3][3] = {
        {0.25f, 0.5f, 0.25f},
        {0.5f, 0.0f, 0.5f},
        {0.25f, 0.5f, 0.25f}
    };
    
    float sum[3] = {0.0f, 0.0f, 0.0f};
    
    // Calculate weighted sum for each color channel
    for (int v = -1; v <= 1; v++) {
        for (int h = -1; h <= 1; h++) {
            int y = row + v;
            int x = col + h;
            int src_idx = y * width + x;
            int src_color = fcol_xtrans(y, x, params.xtrans);
            
            float raw_val = raw_data[src_idx] / params.maximum_value;
            sum[src_color] += raw_val * weight[v + 1][h + 1];
        }
    }
    
    // RawTherapee's exact color interpolation logic
    switch(pixel_color) {
        case 0: // Red pixel
            rgb_data[out_idx + 0] = raw_data[idx] / params.maximum_value;
            rgb_data[out_idx + 1] = sum[1] * 0.5f;
            rgb_data[out_idx + 2] = sum[2];
            break;
            
        case 1: // Green pixel
            {
                rgb_data[out_idx + 1] = raw_data[idx] / params.maximum_value;
                
                // Check if this is a solitary green pixel
                int left_color = fcol_xtrans(row, col - 1, params.xtrans);
                int right_color = fcol_xtrans(row, col + 1, params.xtrans);
                
                if (left_color == right_color) {
                    // Solitary green pixel: exactly two direct red and blue neighbors
                    rgb_data[out_idx + 0] = sum[0];
                    rgb_data[out_idx + 2] = sum[2];
                } else {
                    // Non-solitary green: one direct and one diagonal neighbor
                    rgb_data[out_idx + 0] = sum[0] * 1.3333333f;
                    rgb_data[out_idx + 2] = sum[2] * 1.3333333f;
                }
            }
            break;
            
        case 2: // Blue pixel
            rgb_data[out_idx + 0] = sum[0];
            rgb_data[out_idx + 1] = sum[1] * 0.5f;
            rgb_data[out_idx + 2] = raw_data[idx] / params.maximum_value;
            break;
    }
    
    // Clip to valid range
    for (int c = 0; c < 3; c++) {
        //rgb_data[out_idx + c] = clamp(rgb_data[out_idx + c], 0.0f, 1.0f);
        rgb_data[out_idx + c] = rgb_data[out_idx + c];
    }
}