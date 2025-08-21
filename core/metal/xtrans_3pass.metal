//
// xtrans_3pass.metal
// LibRaw Enhanced - X-Trans 3-pass Demosaicing
// Simplified X-Trans implementation for Fujifilm sensors
//

#include "metal_common.h"

kernel void xtrans_3pass_demosaic(
    const device uint16_t* raw_data [[buffer(0)]],
    device uint16_t* rgb_data [[buffer(1)]],
    constant XTransParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = params.width;
    const uint height = params.height;
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= height || col >= width) return;
    
    const uint raw_idx = row * width + col;
    const uint rgb_idx = raw_idx * 3;
    const uint color = xtrans_color(row, col);
    
    // Edge handling
    if (row < 3 || row >= height-3 || col < 3 || col >= width-3) {
        const uint16_t raw_val = raw_data[raw_idx];
        
        uint16_t r_val = (color == 0) ? raw_val : uint16_t(raw_val * 0.6f);
        uint16_t g_val = (color == 1) ? raw_val : uint16_t(raw_val * 0.8f);
        uint16_t b_val = (color == 2) ? raw_val : uint16_t(raw_val * 0.6f);
        
        // Output raw 16-bit demosaiced values only
        // Let LibRaw handle all post-processing
        rgb_data[rgb_idx + 0] = r_val;  // Red
        rgb_data[rgb_idx + 1] = g_val;  // Green
        rgb_data[rgb_idx + 2] = b_val;  // Blue
        return;
    }
    
    uint16_t r_val, g_val, b_val;
    
    // X-Trans has irregular pattern, so we use weighted averaging
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    float r_count = 0.0f, g_count = 0.0f, b_count = 0.0f;
    
    // Collect values from 5x5 neighborhood
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int nr = int(row) + dy;
            int nc = int(col) + dx;
            
            if (nr >= 0 && nr < int(height) && nc >= 0 && nc < int(width)) {
                uint neighbor_color = xtrans_color(uint(nr), uint(nc));
                float neighbor_val = float(raw_data[nr * width + nc]);
                
                // Weight by distance
                float distance = sqrt(float(dx*dx + dy*dy));
                float weight = 1.0f / (distance + 1.0f);
                
                if (neighbor_color == 0) { // Red
                    r_sum += neighbor_val * weight;
                    r_count += weight;
                } else if (neighbor_color == 1) { // Green
                    g_sum += neighbor_val * weight;
                    g_count += weight;
                } else { // Blue
                    b_sum += neighbor_val * weight;
                    b_count += weight;
                }
            }
        }
    }
    
    // Use original value for pixel's own color, interpolated for others
    if (color == 0) {
        r_val = raw_data[raw_idx];
        g_val = uint16_t(g_count > 0 ? g_sum / g_count : raw_data[raw_idx]);
        b_val = uint16_t(b_count > 0 ? b_sum / b_count : raw_data[raw_idx] / 2);
    } else if (color == 1) {
        r_val = uint16_t(r_count > 0 ? r_sum / r_count : raw_data[raw_idx]);
        g_val = raw_data[raw_idx];
        b_val = uint16_t(b_count > 0 ? b_sum / b_count : raw_data[raw_idx]);
    } else {
        r_val = uint16_t(r_count > 0 ? r_sum / r_count : raw_data[raw_idx] / 2);
        g_val = uint16_t(g_count > 0 ? g_sum / g_count : raw_data[raw_idx]);
        b_val = raw_data[raw_idx];
    }
    
    // Output raw 16-bit demosaiced values only
    // Let LibRaw handle all post-processing (brightness, gamma, color conversion)
    rgb_data[rgb_idx + 0] = r_val;  // Red
    rgb_data[rgb_idx + 1] = g_val;  // Green
    rgb_data[rgb_idx + 2] = b_val;  // Blue
}