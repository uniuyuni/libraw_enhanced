//
// color_space_matrices.metal
// LibRaw Enhanced - Color Space Matrix Definitions
//
// Centralized definitions of LibRaw color space transformation matrices
//

#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// LibRaw color space matrices (from colorconst.cpp)
// These match exactly with LibRaw's out_rgb matrices
constant float3x3 LIBRAW_OUTPUT_COLOR_MATRICES[8] = {
    // 0: sRGB (rgb_rgb) - Identity matrix
    float3x3(float3(1.0, 0.0, 0.0),
             float3(0.0, 1.0, 0.0),
             float3(0.0, 0.0, 1.0)),
    
    // 1: Adobe RGB (1998) (adobe_rgb)
    float3x3(float3(0.715146, 0.284856, 0.000000),
             float3(0.000000, 1.000000, 0.000000),
             float3(0.000000, 0.041166, 0.958839)),
    
    // 2: WideGamut D65 (wide_rgb)
    float3x3(float3(0.593087, 0.404710, 0.002206),
             float3(0.095413, 0.843149, 0.061439),
             float3(0.011621, 0.069091, 0.919288)),
    
    // 3: ProPhoto D65 (prophoto_rgb)
    float3x3(float3(0.529317, 0.330092, 0.140588),
             float3(0.098368, 0.873465, 0.028169),
             float3(0.016879, 0.117663, 0.865457)),
    
    // 4: XYZ (xyz_rgb)
    float3x3(float3(0.4124564, 0.3575761, 0.1804375),
             float3(0.2126729, 0.7151522, 0.0721750),
             float3(0.0193339, 0.1191920, 0.9503041)),
    
    // 5: ACES (aces_rgb)
    float3x3(float3(0.43968015, 0.38295299, 0.17736686),
             float3(0.08978964, 0.81343316, 0.09677734),
             float3(0.01754827, 0.11156156, 0.87089017)),
    
    // 6: DCI-P3 D65 (dcip3d65_rgb)
    float3x3(float3(0.822488, 0.177511, 0.000000),
             float3(0.033200, 0.966800, 0.000000),
             float3(0.017089, 0.072411, 0.910499)),
    
    // 7: Rec. 2020 (rec2020_rgb)
    float3x3(float3(0.627452, 0.329249, 0.043299),
             float3(0.069109, 0.919531, 0.011360),
             float3(0.016398, 0.088030, 0.895572))
};

// Color space names removed due to Metal string constant limitations

// Matrix selection parameters
struct ColorSpaceMatrixParams {
    uint32_t output_color_space;    // LibRaw output_color (0-7, maps to 1-8 in LibRaw)
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t padding[4];            // Alignment padding
};

// Helper function to get color space matrix by ID
float3x3 get_output_color_matrix(uint32_t color_space_id) {
    // Clamp to valid range
    if (color_space_id >= 8) {
        color_space_id = 0; // Fallback to sRGB
    }
    return LIBRAW_OUTPUT_COLOR_MATRICES[color_space_id];
}

// Matrix selection kernel - outputs the selected matrix to a buffer
kernel void select_color_space_matrix(
    device float* output_matrix [[buffer(0)]],           // Output: 3x3 matrix (9 floats)
    constant ColorSpaceMatrixParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // Only need one thread to copy the matrix
    if (gid != 0) return;
    
    float3x3 matrix = get_output_color_matrix(params.output_color_space);
    
    // Copy matrix to output buffer (row-major order)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            output_matrix[i * 3 + j] = matrix[i][j];
        }
    }
}

// Combined matrix multiplication kernel: out_rgb * rgb_cam
// This computes the final out_cam matrix used in LibRaw
kernel void compute_libraw_out_cam_matrix(
    constant float* rgb_cam_matrix [[buffer(0)]],        // Input: 3x4 camera matrix
    device float* out_cam_matrix [[buffer(1)]],          // Output: 3x4 out_cam matrix
    constant ColorSpaceMatrixParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread computes one element of the 3x4 result matrix
    // Total threads needed: 3 * 4 = 12
    
    if (gid >= 12) return;
    
    uint i = gid / 4;  // Row in result matrix (0-2)
    uint j = gid % 4;  // Column in result matrix (0-3)
    
    // Get the output color space matrix
    float3x3 out_rgb = get_output_color_matrix(params.output_color_space);
    
    // Compute: out_cam[i][j] = sum(out_rgb[i][k] * rgb_cam[k][j]) for k=0..2
    float sum = 0.0;
    for (uint k = 0; k < 3; k++) {
        sum += out_rgb[i][k] * rgb_cam_matrix[k * 4 + j];
    }
    
    out_cam_matrix[i * 4 + j] = sum;
}

// Direct color space conversion using selected matrix
kernel void apply_selected_color_space_matrix(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant ColorSpaceMatrixParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t rgb_idx = pixel_idx * 3;
    
    // Read RGB values and normalize to [0, 1]
    float3 input_rgb = float3(
        float(input_image[rgb_idx + 0]) / 65535.0,
        float(input_image[rgb_idx + 1]) / 65535.0,
        float(input_image[rgb_idx + 2]) / 65535.0
    );
    
    // Get color space matrix
    float3x3 color_matrix = get_output_color_matrix(params.output_color_space);
    
    // Apply transformation
    float3 output_rgb = color_matrix * input_rgb;
    
    // Clamp and convert back to 16-bit
    output_rgb = clamp(output_rgb, 0.0, 1.0);
    output_image[rgb_idx + 0] = uint16_t(output_rgb.r * 65535.0);
    output_image[rgb_idx + 1] = uint16_t(output_rgb.g * 65535.0);
    output_image[rgb_idx + 2] = uint16_t(output_rgb.b * 65535.0);
}