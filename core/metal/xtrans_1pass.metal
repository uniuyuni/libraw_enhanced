//
// xtrans_1pass.metal
// LibRaw Enhanced - X-Trans 1-pass Demosaicing
// High-performance single-pass X-Trans implementation for Fujifilm sensors
// Optimized for Apple Silicon Metal acceleration
//

#include "metal_common.h"

// Optimized X-Trans demosaicing using directional interpolation
kernel void xtrans_1pass_demosaic(
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
    
    uint16_t current_val = raw_data[raw_idx];
    uint16_t r_val, g_val, b_val;
    
    // Edge handling - use simple neighbor averaging
    if (row < 2 || row >= height-2 || col < 2 || col >= width-2) {
        // Simple edge interpolation based on current pixel color
        if (color == 0) { // Red pixel
            r_val = current_val;
            g_val = current_val;  // Fallback
            b_val = current_val / 2;  // Estimated
        } else if (color == 1) { // Green pixel  
            r_val = current_val;
            g_val = current_val;
            b_val = current_val;
        } else { // Blue pixel
            r_val = current_val / 2;  // Estimated
            g_val = current_val;
            b_val = current_val;
        }
        
        // Output raw 16-bit demosaiced values only
        rgb_data[rgb_idx + 0] = r_val;  // Red
        rgb_data[rgb_idx + 1] = g_val;  // Green
        rgb_data[rgb_idx + 2] = b_val;  // Blue
        return;
    }
    
    // Single-pass X-Trans demosaicing with optimized neighbor weighting
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    float r_weight = 0.0f, g_weight = 0.0f, b_weight = 0.0f;
    
    // Sample 5x5 neighborhood for X-Trans pattern analysis
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int ny = int(row) + dy;
            int nx = int(col) + dx;
            
            if (ny >= 0 && ny < int(height) && nx >= 0 && nx < int(width)) {
                uint neighbor_color = xtrans_color(uint(ny), uint(nx));
                uint16_t neighbor_val = raw_data[ny * width + nx];
                
                // Distance-based weighting for X-Trans interpolation
                float distance = sqrt(float(dx*dx + dy*dy));
                float weight = (distance > 0.0f) ? (1.0f / (1.0f + distance)) : 2.0f;  // Center pixel gets higher weight
                
                // X-Trans specific color weighting
                if (neighbor_color == color) {
                    weight *= 2.0f;  // Same color pixels get extra weight
                }
                
                // Accumulate weighted sums by color channel
                if (neighbor_color == 0) { // Red
                    r_sum += float(neighbor_val) * weight;
                    r_weight += weight;
                } else if (neighbor_color == 1) { // Green
                    g_sum += float(neighbor_val) * weight;
                    g_weight += weight;
                } else { // Blue
                    b_sum += float(neighbor_val) * weight;
                    b_weight += weight;
                }
            }
        }
    }
    
    // Calculate final RGB values based on current pixel color
    if (color == 0) { // Red pixel
        r_val = current_val;  // Keep original red value
        g_val = uint16_t(g_weight > 0.0f ? g_sum / g_weight : current_val);
        b_val = uint16_t(b_weight > 0.0f ? b_sum / b_weight : current_val);
    } else if (color == 1) { // Green pixel
        r_val = uint16_t(r_weight > 0.0f ? r_sum / r_weight : current_val);
        g_val = current_val;  // Keep original green value
        b_val = uint16_t(b_weight > 0.0f ? b_sum / b_weight : current_val);
    } else { // Blue pixel
        r_val = uint16_t(r_weight > 0.0f ? r_sum / r_weight : current_val);
        g_val = uint16_t(g_weight > 0.0f ? g_sum / g_weight : current_val);
        b_val = current_val;  // Keep original blue value
    }
    
    // Output raw 16-bit demosaiced values only
    // Let LibRaw handle all post-processing (brightness, gamma, color conversion)
    rgb_data[rgb_idx + 0] = r_val;  // Red
    rgb_data[rgb_idx + 1] = g_val;  // Green
    rgb_data[rgb_idx + 2] = b_val;  // Blue
}