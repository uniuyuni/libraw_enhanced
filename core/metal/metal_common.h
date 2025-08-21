//
// metal_common.h
// LibRaw Enhanced - Common Metal definitions and utilities
// Shared across all Metal shaders to avoid redefinition errors
//

#ifndef METAL_COMMON_H
#define METAL_COMMON_H

#include <metal_stdlib>
using namespace metal;

struct BayerParams {
    uint32_t width;
    uint32_t height; 
    uint32_t filters;        // Bayer pattern
    float bright;            // Brightness adjustment
    float gamma_power;       // Gamma correction
    uint32_t use_camera_wb;  // White balance flag
};

struct XTransParams {
    uint32_t width;
    uint32_t height;
    uint32_t xtrans[6][6];   // X-Trans pattern 6x6
    float bright;
    float gamma_power;
    uint32_t use_camera_wb;
};

// Bayer color filter array macro - exact LibRaw implementation
inline uint fcol(uint row, uint col, uint32_t filters) {
    // LibRaw's FC() macro implementation
    // Extract 2-bit color from filters using position-based shift
    return (filters >> ((((row) << 1) & 14) + ((col) & 1)) * 2) & 3;
}

// X-Trans color pattern lookup
inline uint fcol_xtrans(uint row, uint col, const device char xtrans[6][6]) {
    return xtrans[row % 6][col % 6];
}

// X-Trans pattern (6x6 repeating) - LibRaw original pattern  
constant uint8_t xtrans_pattern[6][6] = {
    {1, 1, 0, 1, 1, 2},  // G G R G G B
    {1, 1, 2, 1, 1, 0},  // G G B G G R
    {2, 0, 1, 0, 2, 1},  // B R G R B G
    {1, 1, 2, 1, 1, 0},  // G G B G G R
    {1, 1, 0, 1, 1, 2},  // G G R G G B
    {0, 2, 1, 2, 0, 1}   // R B G B R G
};

inline uint xtrans_color(uint row, uint col) {
    return xtrans_pattern[row % 6][col % 6];
}

// Note: Brightness, gamma, and color processing are now delegated to LibRaw
// GPU shaders only perform demosaicing and output 16-bit RGB values

#endif // METAL_COMMON_H