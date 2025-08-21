//
// bayer_ahd.metal
// LibRaw Enhanced - AHD (Adaptive Homogeneity-Directed) Demosaicing
// Simplified AHD implementation for Metal GPU
//

#include "metal_common.h"

// Simplified homogeneity calculation
inline float homogeneity_h(const device uint16_t* raw_data, uint width, uint height, uint row, uint col) {
    if (col < 2 || col >= width - 2) return 0.0f;
    
    float h1 = abs(float(raw_data[row * width + (col-2)]) - float(raw_data[row * width + col]));
    float h2 = abs(float(raw_data[row * width + col]) - float(raw_data[row * width + (col+2)]));
    return h1 + h2;
}

inline float homogeneity_v(const device uint16_t* raw_data, uint width, uint height, uint row, uint col) {
    if (row < 2 || row >= height - 2) return 0.0f;
    
    float v1 = abs(float(raw_data[(row-2) * width + col]) - float(raw_data[row * width + col]));
    float v2 = abs(float(raw_data[row * width + col]) - float(raw_data[(row+2) * width + col]));
    return v1 + v2;
}

kernel void bayer_ahd_demosaic(
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
    
    if (color == 1) { // Green pixel - already have green value
        g_val = raw_data[raw_idx];
        
        // Use homogeneity to decide interpolation direction for R and B
        float h_homo = homogeneity_h(raw_data, width, height, row, col);
        float v_homo = homogeneity_v(raw_data, width, height, row, col);
        
        if (h_homo < v_homo) { // Horizontal homogeneity is stronger
            r_val = (raw_data[row * width + (col-1)] + raw_data[row * width + (col+1)]) / 2;
            b_val = (raw_data[row * width + (col-1)] + raw_data[row * width + (col+1)]) / 2;
        } else { // Vertical homogeneity is stronger
            r_val = (raw_data[(row-1) * width + col] + raw_data[(row+1) * width + col]) / 2;
            b_val = (raw_data[(row-1) * width + col] + raw_data[(row+1) * width + col]) / 2;
        }
        
    } else if (color == 0) { // Red pixel
        r_val = raw_data[raw_idx];
        
        // Interpolate green using gradients
        float gN = float(raw_data[(row-1) * width + col]);
        float gS = float(raw_data[(row+1) * width + col]);
        float gW = float(raw_data[row * width + (col-1)]);
        float gE = float(raw_data[row * width + (col+1)]);
        
        float gradN = abs(float(raw_data[(row-2) * width + col]) - float(raw_data[row * width + col]));
        float gradS = abs(float(raw_data[(row+2) * width + col]) - float(raw_data[row * width + col]));
        float gradW = abs(float(raw_data[row * width + (col-2)]) - float(raw_data[row * width + col]));
        float gradE = abs(float(raw_data[row * width + (col+2)]) - float(raw_data[row * width + col]));
        
        float min_grad = min(min(gradN, gradS), min(gradW, gradE));
        float sum = 0.0f;
        float count = 0.0f;
        
        if (gradN == min_grad) { sum += gN; count += 1.0f; }
        if (gradS == min_grad) { sum += gS; count += 1.0f; }
        if (gradW == min_grad) { sum += gW; count += 1.0f; }
        if (gradE == min_grad) { sum += gE; count += 1.0f; }
        
        g_val = uint16_t(count > 0 ? sum / count : (gN + gS + gW + gE) / 4.0f);
        
        // Blue interpolation - diagonal average
        b_val = (raw_data[(row-1) * width + (col-1)] + 
                 raw_data[(row-1) * width + (col+1)] +
                 raw_data[(row+1) * width + (col-1)] + 
                 raw_data[(row+1) * width + (col+1)]) / 4;
                 
    } else { // Blue pixel
        b_val = raw_data[raw_idx];
        
        // Interpolate green using gradients (same as for red)
        float gN = float(raw_data[(row-1) * width + col]);
        float gS = float(raw_data[(row+1) * width + col]);
        float gW = float(raw_data[row * width + (col-1)]);
        float gE = float(raw_data[row * width + (col+1)]);
        
        float gradN = abs(float(raw_data[(row-2) * width + col]) - float(raw_data[row * width + col]));
        float gradS = abs(float(raw_data[(row+2) * width + col]) - float(raw_data[row * width + col]));
        float gradW = abs(float(raw_data[row * width + (col-2)]) - float(raw_data[row * width + col]));
        float gradE = abs(float(raw_data[row * width + (col+2)]) - float(raw_data[row * width + col]));
        
        float min_grad = min(min(gradN, gradS), min(gradW, gradE));
        float sum = 0.0f;
        float count = 0.0f;
        
        if (gradN == min_grad) { sum += gN; count += 1.0f; }
        if (gradS == min_grad) { sum += gS; count += 1.0f; }
        if (gradW == min_grad) { sum += gW; count += 1.0f; }
        if (gradE == min_grad) { sum += gE; count += 1.0f; }
        
        g_val = uint16_t(count > 0 ? sum / count : (gN + gS + gW + gE) / 4.0f);
        
        // Red interpolation - diagonal average
        r_val = (raw_data[(row-1) * width + (col-1)] + 
                 raw_data[(row-1) * width + (col+1)] +
                 raw_data[(row+1) * width + (col-1)] + 
                 raw_data[(row+1) * width + (col+1)]) / 4;
    }
    
    // Output raw 16-bit demosaiced values only
    // Let LibRaw handle all post-processing (brightness, gamma, color conversion)
    rgb_data[rgb_idx + 0] = r_val;  // Red
    rgb_data[rgb_idx + 1] = g_val;  // Green
    rgb_data[rgb_idx + 2] = b_val;  // Blue
}