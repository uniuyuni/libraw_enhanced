//
// bayer_amaze.metal
// LibRaw Enhanced - AMaZE (Aliasing Minimization and Zipper Elimination) Demosaicing
// Simplified AMaZE implementation for Metal GPU
//

#include "metal_common.h"

kernel void bayer_amaze_demosaic(
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
    
    // Edge pixels use simple interpolation
    if (row < 3 || row >= height-3 || col < 3 || col >= width-3) {
        const uint16_t raw_val = raw_data[raw_idx];
        
        uint16_t r_val = (color == 0) ? raw_val : uint16_t(raw_val * 0.5f);
        uint16_t g_val = (color == 1) ? raw_val : uint16_t(raw_val * 0.8f);
        uint16_t b_val = (color == 2) ? raw_val : uint16_t(raw_val * 0.5f);
        
        // Output raw 16-bit demosaiced values only
        // Let LibRaw handle all post-processing
        rgb_data[rgb_idx + 0] = r_val;  // Red
        rgb_data[rgb_idx + 1] = g_val;  // Green
        rgb_data[rgb_idx + 2] = b_val;  // Blue
        return;
    }
    
    uint16_t r_val, g_val, b_val;
    
    if (color == 1) { // Green pixel - AMaZE uses sophisticated edge detection
        g_val = raw_data[raw_idx];
        
        // Multi-directional analysis for better edge detection
        float directions[8];
        
        // Calculate directional differences
        directions[0] = abs(float(raw_data[(row-2) * width + (col-2)]) - float(raw_data[(row+2) * width + (col+2)])); // NW-SE
        directions[1] = abs(float(raw_data[(row-2) * width + col])      - float(raw_data[(row+2) * width + col]));      // N-S
        directions[2] = abs(float(raw_data[(row-2) * width + (col+2)]) - float(raw_data[(row+2) * width + (col-2)])); // NE-SW
        directions[3] = abs(float(raw_data[row * width + (col-2)])      - float(raw_data[row * width + (col+2)]));      // W-E
        directions[4] = abs(float(raw_data[(row-1) * width + (col-1)]) - float(raw_data[(row+1) * width + (col+1)])); // diag1
        directions[5] = abs(float(raw_data[(row-1) * width + (col+1)]) - float(raw_data[(row+1) * width + (col-1)])); // diag2
        directions[6] = abs(float(raw_data[(row-1) * width + col])      - float(raw_data[(row+1) * width + col]));      // vert
        directions[7] = abs(float(raw_data[row * width + (col-1)])      - float(raw_data[row * width + (col+1)]));      // horiz
        
        // Find minimum direction (most homogeneous)
        float min_dir = directions[0];
        uint min_idx = 0;
        for (uint i = 1; i < 8; i++) {
            if (directions[i] < min_dir) {
                min_dir = directions[i];
                min_idx = i;
            }
        }
        
        // Interpolate R and B based on most homogeneous direction
        switch (min_idx) {
            case 0: // NW-SE diagonal
                if (fcol(row-1, col-1, params.filters) == 0) {
                    r_val = (raw_data[(row-1) * width + (col-1)] + raw_data[(row+1) * width + (col+1)]) / 2;
                    b_val = (raw_data[(row-1) * width + (col+1)] + raw_data[(row+1) * width + (col-1)]) / 2;
                } else {
                    b_val = (raw_data[(row-1) * width + (col-1)] + raw_data[(row+1) * width + (col+1)]) / 2;
                    r_val = (raw_data[(row-1) * width + (col+1)] + raw_data[(row+1) * width + (col-1)]) / 2;
                }
                break;
                
            case 1: // Vertical
                if (fcol(row-1, col, params.filters) == 0) {
                    r_val = (raw_data[(row-1) * width + col] + raw_data[(row+1) * width + col]) / 2;
                    b_val = (raw_data[row * width + (col-1)] + raw_data[row * width + (col+1)]) / 2;
                } else {
                    b_val = (raw_data[(row-1) * width + col] + raw_data[(row+1) * width + col]) / 2;
                    r_val = (raw_data[row * width + (col-1)] + raw_data[row * width + (col+1)]) / 2;
                }
                break;
                
            case 3: // Horizontal
                if (fcol(row, col-1, params.filters) == 0) {
                    r_val = (raw_data[row * width + (col-1)] + raw_data[row * width + (col+1)]) / 2;
                    b_val = (raw_data[(row-1) * width + col] + raw_data[(row+1) * width + col]) / 2;
                } else {
                    b_val = (raw_data[row * width + (col-1)] + raw_data[row * width + (col+1)]) / 2;
                    r_val = (raw_data[(row-1) * width + col] + raw_data[(row+1) * width + col]) / 2;
                }
                break;
                
            default: // Use average of all neighbors
                r_val = (raw_data[(row-1) * width + (col-1)] + raw_data[(row-1) * width + (col+1)] +
                         raw_data[(row+1) * width + (col-1)] + raw_data[(row+1) * width + (col+1)]) / 4;
                b_val = r_val;
                break;
        }
        
    } else if (color == 0) { // Red pixel
        r_val = raw_data[raw_idx];
        
        // AMaZE green interpolation with edge-aware smoothing
        float gN = float(raw_data[(row-1) * width + col]);
        float gS = float(raw_data[(row+1) * width + col]);
        float gW = float(raw_data[row * width + (col-1)]);
        float gE = float(raw_data[row * width + (col+1)]);
        
        // Calculate second derivatives for smoother interpolation
        float d2N = abs(float(raw_data[(row-3) * width + col]) - 2.0f * float(raw_data[(row-1) * width + col]) + float(raw_data[raw_idx]));
        float d2S = abs(float(raw_data[raw_idx]) - 2.0f * float(raw_data[(row+1) * width + col]) + float(raw_data[(row+3) * width + col]));
        float d2W = abs(float(raw_data[row * width + (col-3)]) - 2.0f * float(raw_data[row * width + (col-1)]) + float(raw_data[raw_idx]));
        float d2E = abs(float(raw_data[raw_idx]) - 2.0f * float(raw_data[row * width + (col+1)]) + float(raw_data[row * width + (col+3)]));
        
        // Weight by inverse of second derivatives
        float wN = 1.0f / (d2N + 1.0f);
        float wS = 1.0f / (d2S + 1.0f);
        float wW = 1.0f / (d2W + 1.0f);
        float wE = 1.0f / (d2E + 1.0f);
        
        g_val = uint16_t((gN * wN + gS * wS + gW * wW + gE * wE) / (wN + wS + wW + wE));
        
        // Blue interpolation with anti-aliasing
        float b1 = float(raw_data[(row-1) * width + (col-1)]);
        float b2 = float(raw_data[(row-1) * width + (col+1)]);
        float b3 = float(raw_data[(row+1) * width + (col-1)]);
        float b4 = float(raw_data[(row+1) * width + (col+1)]);
        
        b_val = uint16_t((b1 + b2 + b3 + b4) / 4.0f);
        
    } else { // Blue pixel
        b_val = raw_data[raw_idx];
        
        // Same green interpolation as red pixel
        float gN = float(raw_data[(row-1) * width + col]);
        float gS = float(raw_data[(row+1) * width + col]);
        float gW = float(raw_data[row * width + (col-1)]);
        float gE = float(raw_data[row * width + (col+1)]);
        
        float d2N = abs(float(raw_data[(row-3) * width + col]) - 2.0f * float(raw_data[(row-1) * width + col]) + float(raw_data[raw_idx]));
        float d2S = abs(float(raw_data[raw_idx]) - 2.0f * float(raw_data[(row+1) * width + col]) + float(raw_data[(row+3) * width + col]));
        float d2W = abs(float(raw_data[row * width + (col-3)]) - 2.0f * float(raw_data[row * width + (col-1)]) + float(raw_data[raw_idx]));
        float d2E = abs(float(raw_data[raw_idx]) - 2.0f * float(raw_data[row * width + (col+1)]) + float(raw_data[row * width + (col+3)]));
        
        float wN = 1.0f / (d2N + 1.0f);
        float wS = 1.0f / (d2S + 1.0f);
        float wW = 1.0f / (d2W + 1.0f);
        float wE = 1.0f / (d2E + 1.0f);
        
        g_val = uint16_t((gN * wN + gS * wS + gW * wW + gE * wE) / (wN + wS + wW + wE));
        
        // Red interpolation
        float r1 = float(raw_data[(row-1) * width + (col-1)]);
        float r2 = float(raw_data[(row-1) * width + (col+1)]);
        float r3 = float(raw_data[(row+1) * width + (col-1)]);
        float r4 = float(raw_data[(row+1) * width + (col+1)]);
        
        r_val = uint16_t((r1 + r2 + r3 + r4) / 4.0f);
    }
    
    // Output raw 16-bit demosaiced values only
    // Let LibRaw handle all post-processing (brightness, gamma, color conversion)
    rgb_data[rgb_idx + 0] = r_val;  // Red
    rgb_data[rgb_idx + 1] = g_val;  // Green
    rgb_data[rgb_idx + 2] = b_val;  // Blue
}