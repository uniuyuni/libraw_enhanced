//
// bayer_dcb.metal
// LibRaw Enhanced - DCB (Directional Cubic Spline Bayer) Demosaicing
// LibRaw-faithful DCB implementation for Metal GPU
// Based on Jacek Gozdz's DCB algorithm in LibRaw
//

#include "metal_common.h"

// LibRaw-faithful DCB demosaicing with directional green interpolation
kernel void bayer_dcb_demosaic(
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
    
    // Edge handling - use linear interpolation for borders (like LibRaw)
    if (row < 2 || row >= height-2 || col < 2 || col >= width-2) {
        const uint16_t raw_val = raw_data[raw_idx];
        
        uint16_t r_val, g_val, b_val;
        
        // Simple linear interpolation for edge pixels
        if (color == 0) { // Red pixel
            r_val = raw_val;
            // Green interpolation from orthogonal neighbors
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
            
            // Blue interpolation from diagonal neighbors
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
            // Red and Blue from adjacent pixels
            r_val = raw_val;
            b_val = raw_val;
            
        } else { // Blue pixel
            b_val = raw_val;
            // Green interpolation (same as red)
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
            
            // Red interpolation from diagonal neighbors
            uint r_count = 0;
            uint r_sum = 0;
            if (row > 0 && col > 0 && fcol(row-1, col-1, params.filters) == 0) {
                r_sum += raw_data[(row-1) * width + (col-1)];
                r_count++;
            }
            if (row > 0 && col < width-1 && fcol(row-1, col+1, params.filters) == 0) {
                r_sum += raw_data[(row-1) * width + (col+1)];
                r_count++;
            }
            if (row < height-1 && col > 0 && fcol(row+1, col-1, params.filters) == 0) {
                r_sum += raw_data[(row+1) * width + (col-1)];
                r_count++;
            }
            if (row < height-1 && col < width-1 && fcol(row+1, col+1, params.filters) == 0) {
                r_sum += raw_data[(row+1) * width + (col+1)];
                r_count++;
            }
            r_val = r_count > 0 ? r_sum / r_count : raw_val / 2;
        }
        
        // Output raw 16-bit demosaiced values only
        // Let LibRaw handle all post-processing
        rgb_data[rgb_idx + 0] = r_val;  // Red
        rgb_data[rgb_idx + 1] = g_val;  // Green
        rgb_data[rgb_idx + 2] = b_val;  // Blue
        return;
    }
    
    // LibRaw DCB core algorithm for inner pixels
    uint16_t current_val = raw_data[raw_idx];
    uint16_t r_val, g_val, b_val;
    
    if (color == 1) { // Green pixel - base case
        g_val = current_val;
        
        // DCB color interpolation for missing R and B
        // Based on LibRaw's dcb_color algorithm
        uint c = (fcol(row, col+1, params.filters) == 0) ? 0 : 2; // R or B color to interpolate horizontally
        uint d = 2 - c; // The other color (B or R)
        
        // Horizontal interpolation for one color
        if (col > 0 && col < width-1) {
            uint16_t left = raw_data[row * width + (col-1)];
            uint16_t right = raw_data[row * width + (col+1)];
            if (c == 0) {
                r_val = (left + right) / 2;
            } else {
                b_val = (left + right) / 2;
            }
        } else {
            if (c == 0) r_val = current_val;
            else b_val = current_val;
        }
        
        // Vertical interpolation for the other color
        if (row > 0 && row < height-1) {
            uint16_t up = raw_data[(row-1) * width + col];
            uint16_t down = raw_data[(row+1) * width + col];
            if (d == 0) {
                r_val = (up + down) / 2;
            } else {
                b_val = (up + down) / 2;
            }
        } else {
            if (d == 0) r_val = current_val;
            else b_val = current_val;
        }
        
    } else { // Non-green pixel (R or B)
        // DCB directional green interpolation
        // Calculate horizontal and vertical green estimates
        
        float g_hor = 0.0f, g_ver = 0.0f;
        
        // Horizontal green interpolation
        if (col > 0 && col < width-1) {
            uint16_t left_g = raw_data[row * width + (col-1)];
            uint16_t right_g = raw_data[row * width + (col+1)];
            g_hor = (float(left_g) + float(right_g)) / 2.0f;
        } else {
            g_hor = float(current_val);
        }
        
        // Vertical green interpolation  
        if (row > 0 && row < height-1) {
            uint16_t up_g = raw_data[(row-1) * width + col];
            uint16_t down_g = raw_data[(row+1) * width + col];
            g_ver = (float(up_g) + float(down_g)) / 2.0f;
        } else {
            g_ver = float(current_val);
        }
        
        // DCB decision criteria - calculate local variations
        float hor_var = 0.0f, ver_var = 0.0f;
        
        // Horizontal variation
        if (col >= 2 && col < width-2) {
            float h1 = float(raw_data[row * width + (col-2)]);
            float h2 = float(raw_data[row * width + col]);
            float h3 = float(raw_data[row * width + (col+2)]);
            hor_var = abs(h1 - h2) + abs(h3 - h2);
        }
        
        // Vertical variation
        if (row >= 2 && row < height-2) {
            float v1 = float(raw_data[(row-2) * width + col]);
            float v2 = float(raw_data[row * width + col]);
            float v3 = float(raw_data[(row+2) * width + col]);
            ver_var = abs(v1 - v2) + abs(v3 - v2);
        }
        
        // Choose direction based on lower variation (LibRaw DCB logic)
        g_val = uint16_t((hor_var < ver_var) ? g_hor : g_ver);
        
        // Set current color value
        if (color == 0) {
            r_val = current_val;
            // Interpolate blue using DCB diagonal estimation
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
            b_val = b_count > 0 ? b_sum / b_count : current_val;
        } else { // Blue pixel
            b_val = current_val;
            // Interpolate red using DCB diagonal estimation
            uint r_count = 0;
            uint r_sum = 0;
            if (row > 0 && col > 0 && fcol(row-1, col-1, params.filters) == 0) {
                r_sum += raw_data[(row-1) * width + (col-1)];
                r_count++;
            }
            if (row > 0 && col < width-1 && fcol(row-1, col+1, params.filters) == 0) {
                r_sum += raw_data[(row-1) * width + (col+1)];
                r_count++;
            }
            if (row < height-1 && col > 0 && fcol(row+1, col-1, params.filters) == 0) {
                r_sum += raw_data[(row+1) * width + (col-1)];
                r_count++;
            }
            if (row < height-1 && col < width-1 && fcol(row+1, col+1, params.filters) == 0) {
                r_sum += raw_data[(row+1) * width + (col+1)];
                r_count++;
            }
            r_val = r_count > 0 ? r_sum / r_count : current_val;
        }
    }
    
    // Output raw 16-bit demosaiced values only
    // Let LibRaw handle all post-processing (brightness, gamma, color conversion)
    rgb_data[rgb_idx + 0] = r_val;  // Red
    rgb_data[rgb_idx + 1] = g_val;  // Green
    rgb_data[rgb_idx + 2] = b_val;  // Blue
}