//
// white_balance_raw.metal  
// LibRaw Enhanced - RAW White Balance Metal Shaders
//
// Step 1 of LibRaw pipeline: scale_colors() - Applied to RAW sensor data
//

#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// White Balance Parameters for RAW data
struct WhiteBalanceRawParams {
    float pre_mul[4];           // White balance multipliers [R, G1, B, G2]
    float scale_mul[4];         // Scaling multipliers from LibRaw
    uint32_t width;
    uint32_t height;
    uint32_t filters;           // Bayer pattern filters (0x94949494 = RGGB)
    float bright;               // Overall brightness adjustment
    uint32_t use_camera_wb;     // Use camera white balance flag
    uint32_t padding;           // Alignment padding
};

// Bayer color channel determination
uint32_t get_bayer_color(uint32_t row, uint32_t col, uint32_t filters) {
    // LibRaw's FC macro implementation
    return ((filters >> (((row << 1 & 14) + (col & 1)) << 1)) & 3);
}

// X-Trans color channel determination  
uint32_t get_xtrans_color(uint32_t row, uint32_t col) {
    // Use the pattern from metal_common.h
    return xtrans_color(row, col);
}

// White balance kernel for RAW Bayer data
kernel void apply_white_balance_bayer(
    const device uint16_t* raw_input [[buffer(0)]],
    device uint16_t* raw_output [[buffer(1)]],
    constant WhiteBalanceRawParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    
    // Determine Bayer color channel for this pixel
    uint32_t color = get_bayer_color(gid.y, gid.x, params.filters);
    
    // Read RAW value
    float raw_value = float(raw_input[pixel_idx]);
    
    // Apply white balance scaling: value * scale_mul[color] * bright
    float wb_value = raw_value * params.scale_mul[color] * params.bright;
    
    // Clamp to 16-bit range
    wb_value = clamp(wb_value, 0.0f, 65535.0f);
    
    // Write result
    raw_output[pixel_idx] = uint16_t(wb_value);
}

// White balance kernel for RAW X-Trans data
kernel void apply_white_balance_xtrans(
    const device uint16_t* raw_input [[buffer(0)]],
    device uint16_t* raw_output [[buffer(1)]],
    constant WhiteBalanceRawParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    
    // Determine X-Trans color channel for this pixel
    uint32_t color = get_xtrans_color(gid.y, gid.x);
    
    // Read RAW value
    float raw_value = float(raw_input[pixel_idx]);
    
    // Apply white balance scaling
    float wb_value = raw_value * params.scale_mul[color] * params.bright;
    
    // Clamp to 16-bit range
    wb_value = clamp(wb_value, 0.0f, 65535.0f);
    
    // Write result
    raw_output[pixel_idx] = uint16_t(wb_value);
}