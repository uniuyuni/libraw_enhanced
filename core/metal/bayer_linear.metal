//
// bayer_linear.metal  
// LibRaw Enhanced - Linear Bayer Demosaicing Shader
// Exact LibRaw lin_interpolate() implementation
//

#include "metal_common.h"

kernel void bayer_linear_demosaic(
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
    
    uint16_t current_val = raw_data[raw_idx];
    uint16_t interpolated[3];  // R, G, B values after demosaicing
    
    // LibRaw lin_interpolate exact implementation
    // Border pixels: use border_interpolate(1) equivalent
    if (row == 0 || row == height-1 || col == 0 || col == width-1) {
        // LibRaw's border_interpolate fills missing colors by averaging neighbors
        uint sum[3] = {0, 0, 0};
        uint count[3] = {0, 0, 0};
        
        // Check 3x3 neighborhood
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = int(row) + dy;
                int nx = int(col) + dx;
                
                if (ny >= 0 && ny < int(height) && nx >= 0 && nx < int(width)) {
                    uint neighbor_color = fcol(uint(ny), uint(nx), params.filters);
                    uint16_t neighbor_val = raw_data[ny * width + nx];
                    
                    sum[neighbor_color] += neighbor_val;
                    count[neighbor_color]++;
                }
            }
        }
        
        // Fill interpolated values
        for (uint c = 0; c < 3; c++) {
            if (c == color) {
                interpolated[c] = current_val;  // Original color
            } else {
                interpolated[c] = count[c] > 0 ? uint16_t(sum[c] / count[c]) : current_val;
            }
        }
    } else {
        // Inner pixels: LibRaw's exact linear interpolation with shift weights
        int sum[3] = {0, 0, 0};
        int div[3] = {0, 0, 0};  // LibRaw uses 256/count division factors
        
        // Check 8 neighbors, calculate weighted sums and division factors
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue; // Skip center
                
                int ny = int(row) + dy;
                int nx = int(col) + dx;
                
                uint neighbor_color = fcol(uint(ny), uint(nx), params.filters);
                if (neighbor_color == color) continue; // Only interpolate missing colors
                
                uint16_t neighbor_val = raw_data[ny * width + nx];
                int shift = (dy == 0 ? 1 : 0) + (dx == 0 ? 1 : 0); // LibRaw shift weighting
                
                sum[neighbor_color] += int(neighbor_val) << shift;
                div[neighbor_color] += 256 << shift; // LibRaw division factor
            }
        }
        
        // Fill interpolated values with LibRaw division
        for (uint c = 0; c < 3; c++) {
            if (c == color) {
                interpolated[c] = current_val;  // Original color
            } else {
                interpolated[c] = div[c] > 0 ? uint16_t(sum[c] / div[c]) : current_val;
            }
        }
    }
    
    // Output raw 16-bit demosaiced values only (3-channel RGB for custom pipeline)
    // Let LibRaw handle all post-processing (brightness, gamma, color conversion)
    rgb_data[rgb_idx + 0] = interpolated[0];  // Red
    rgb_data[rgb_idx + 1] = interpolated[1];  // Green
    rgb_data[rgb_idx + 2] = interpolated[2];  // Blue
}