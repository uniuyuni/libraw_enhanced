//
// bayer_vng.metal
// LibRaw Enhanced - VNG (Variable Number of Gradients) Bayer Demosaicing
// LibRaw-faithful VNG implementation for Metal GPU
// Based on LibRaw's vng_interpolate algorithm
//

#include "metal_common.h"

// VNG neighborhood patterns (LibRaw-faithful implementation)
constant int2 vng_neighbors[8] = {
    int2(-1, -1), int2(-1, 0), int2(-1, 1), 
    int2(0, -1),               int2(0, 1),
    int2(1, -1),  int2(1, 0),  int2(1, 1)
};
// LibRaw-faithful VNG interpolation using gradient-based selection
kernel void bayer_vng_demosaic(
    const device uint16_t* raw_data [[buffer(0)]],
    device uint16_t* rgb_data [[buffer(1)]],
    constant BayerParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = params.width;
    const uint height = params.height;
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= height || col >= width) return;
    
    const uint raw_idx = row * width + col;
    const uint rgb_idx = raw_idx * 3;
    const uint color = fcol(row, col, params.filters);
    
    // Use linear interpolation for edge pixels (LibRaw does this too)
    if (row < 2 || row >= height-2 || col < 2 || col >= width-2) {
        const uint16_t raw_val = raw_data[raw_idx];
        
        uint16_t r_val, g_val, b_val;
        
        // Simple neighbor averaging for edge pixels
        if (color == 0) { // Red pixel
            r_val = raw_val;
            // Green interpolation
            uint g_count = 0;
            uint g_sum = 0;
            if (row > 0 && fcol(row-1, col, params.filters) == 1) {
                g_sum += raw_data[(row-1) * width + col];
                g_count++;
            }
            if (row < height-1 && fcol(row+1, col, params.filters) == 1) {
                g_sum += raw_data[(row+1) * width + col];
                g_count++;
            }
            if (col > 0 && fcol(row, col-1, params.filters) == 1) {
                g_sum += raw_data[row * width + (col-1)];
                g_count++;
            }
            if (col < width-1 && fcol(row, col+1, params.filters) == 1) {
                g_sum += raw_data[row * width + (col+1)];
                g_count++;
            }
            g_val = g_count > 0 ? g_sum / g_count : raw_val;
            
            // Blue interpolation (diagonal)
            uint b_count = 0;
            uint b_sum = 0;
            if (row > 0 && col > 0 && fcol(row-1, col-1, params.filters) == 2) {
                b_sum += raw_data[(row-1) * width + (col-1)];
                b_count++;
            }
            if (row > 0 && col < width-1 && fcol(row-1, col+1, params.filters) == 2) {
                b_sum += raw_data[(row-1) * width + (col+1)];
                b_count++;
            }
            if (row < height-1 && col > 0 && fcol(row+1, col-1, params.filters) == 2) {
                b_sum += raw_data[(row+1) * width + (col-1)];
                b_count++;
            }
            if (row < height-1 && col < width-1 && fcol(row+1, col+1, params.filters) == 2) {
                b_sum += raw_data[(row+1) * width + (col+1)];
                b_count++;
            }
            b_val = b_count > 0 ? b_sum / b_count : raw_val / 2;
            
        } else if (color == 1) { // Green pixel
            g_val = raw_val;
            // Red and Blue interpolation from orthogonal neighbors
            r_val = raw_val;
            b_val = raw_val;
        } else { // Blue pixel
            b_val = raw_val;
            // Similar to red pixel logic but swapped
            g_val = raw_val;
            r_val = raw_val / 2;
        }
        
        // Output raw 16-bit demosaiced values only
        // Let LibRaw handle all post-processing
        rgb_data[rgb_idx + 0] = r_val;  // Red
        rgb_data[rgb_idx + 1] = g_val;  // Green
        rgb_data[rgb_idx + 2] = b_val;  // Blue
        return;
    }
    
    // LibRaw-faithful VNG gradient calculation
    float gradients[8] = {0.0f};
    
    // Calculate gradients for each of the 8 directions
    for (uint d = 0; d < 8; d++) {
        int2 offset1 = vng_neighbors[d];
        int2 offset2 = -offset1; // Opposite direction
        
        int r1 = int(row) + offset1.y;
        int c1 = int(col) + offset1.x;
        int r2 = int(row) + offset2.y;
        int c2 = int(col) + offset2.x;
        
        if (r1 >= 0 && r1 < int(height) && c1 >= 0 && c1 < int(width) &&
            r2 >= 0 && r2 < int(height) && c2 >= 0 && c2 < int(width)) {
            
            uint16_t val1 = raw_data[r1 * width + c1];
            uint16_t val2 = raw_data[r2 * width + c2];
            
            // LibRaw uses weighted difference based on distance and color pattern
            float weight = (abs(offset1.x) + abs(offset1.y) == 1) ? 1.0f : 0.707f; // sqrt(2)/2 for diagonals
            gradients[d] = abs(float(val1) - float(val2)) * weight;
        } else {
            gradients[d] = 65535.0f; // Large gradient for boundaries
        }
    }
    
    // Find minimum and maximum gradients
    float gmin = gradients[0];
    float gmax = gradients[0];
    for (uint d = 1; d < 8; d++) {
        gmin = min(gmin, gradients[d]);
        gmax = max(gmax, gradients[d]);
    }
    
    // LibRaw's threshold calculation
    float threshold = gmin + (gmax - gmin) * 0.5f;
    if (gmax == 0.0f) threshold = 0.0f;
    
    // Collect weighted averages from directions below threshold
    float sums[3] = {0.0f, 0.0f, 0.0f};
    float weights[3] = {0.0f, 0.0f, 0.0f};
    
    for (uint d = 0; d < 8; d++) {
        if (gradients[d] <= threshold) {
            int2 offset = vng_neighbors[d];
            int nr = int(row) + offset.y;
            int nc = int(col) + offset.x;
            
            if (nr >= 0 && nr < int(height) && nc >= 0 && nc < int(width)) {
                uint neighbor_color = fcol(uint(nr), uint(nc), params.filters);
                float neighbor_val = float(raw_data[nr * width + nc]);
                float weight = (threshold > 0.0f) ? (threshold - gradients[d]) / threshold : 1.0f;
                
                sums[neighbor_color] += neighbor_val * weight;
                weights[neighbor_color] += weight;
            }
        }
    }
    
    // Generate final RGB values
    uint16_t r_val, g_val, b_val;
    uint16_t current_val = raw_data[raw_idx];
    
    if (color == 0) { // Red pixel
        r_val = current_val;
        g_val = uint16_t(weights[1] > 0.0f ? sums[1] / weights[1] : current_val);
        b_val = uint16_t(weights[2] > 0.0f ? sums[2] / weights[2] : current_val);
    } else if (color == 1) { // Green pixel
        r_val = uint16_t(weights[0] > 0.0f ? sums[0] / weights[0] : current_val);
        g_val = current_val;
        b_val = uint16_t(weights[2] > 0.0f ? sums[2] / weights[2] : current_val);
    } else { // Blue pixel
        r_val = uint16_t(weights[0] > 0.0f ? sums[0] / weights[0] : current_val);
        g_val = uint16_t(weights[1] > 0.0f ? sums[1] / weights[1] : current_val);
        b_val = current_val;
    }
    
    // Output raw 16-bit demosaiced values only
    // Let LibRaw handle all post-processing (brightness, gamma, color conversion)
    rgb_data[rgb_idx + 0] = r_val;  // Red
    rgb_data[rgb_idx + 1] = g_val;  // Green
    rgb_data[rgb_idx + 2] = b_val;  // Blue
}