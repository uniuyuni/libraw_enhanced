//
// color_conversion.metal
// LibRaw Enhanced - Color Space Conversion Metal Shaders
//
// Missing functions that GPU accelerator expects
//

#include "metal_common.h"

using namespace metal;

// Parameters for color space conversion
struct ColorConversionParams {
    float out_cam[3][4];        // Combined transformation matrix
    uint32_t width;
    uint32_t height;
    uint32_t output_color;      // Output color space (0-7)
    float gamma_power;          // Gamma power value
    float gamma_slope;          // Gamma slope value
    uint32_t apply_gamma;       // Whether to apply gamma correction
    uint32_t raw_color;         // Whether this is raw color processing
    uint32_t padding[2];        // Memory alignment
};

// Parameters for white balance + color conversion
struct WhiteBalanceColorParams {
    float wb_multipliers[4];    // White balance multipliers [R, G1, B, G2]
    float color_matrix[3][3];   // Color transformation matrix
    uint32_t width;
    uint32_t height;
    uint32_t output_color;      // Output color space (0-7)
    float gamma_power;          // Gamma power value
    float gamma_slope;          // Gamma slope value
    uint32_t apply_gamma;       // Whether to apply gamma correction
    uint32_t padding[1];        // Memory alignment
};

// Basic color space conversion (identity transform for now)
kernel void color_space_convert(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant ColorConversionParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t rgb_idx = pixel_idx * 3;
    
    // For now, just copy input to output (identity transform)
    // In a full implementation, this would apply color space conversion
    output_image[rgb_idx + 0] = input_image[rgb_idx + 0];
    output_image[rgb_idx + 1] = input_image[rgb_idx + 1];
    output_image[rgb_idx + 2] = input_image[rgb_idx + 2];
}

// Camera to output color space conversion
kernel void camera_to_output_convert(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant ColorConversionParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t rgb_idx = pixel_idx * 3;
    
    // Read input RGB values and normalize
    float3 input_rgb = float3(
        float(input_image[rgb_idx + 0]) / 65535.0,
        float(input_image[rgb_idx + 1]) / 65535.0,
        float(input_image[rgb_idx + 2]) / 65535.0
    );
    
    // Apply out_cam matrix transformation
    float3 output_rgb = float3(
        params.out_cam[0][0] * input_rgb.r + params.out_cam[0][1] * input_rgb.g + params.out_cam[0][2] * input_rgb.b,
        params.out_cam[1][0] * input_rgb.r + params.out_cam[1][1] * input_rgb.g + params.out_cam[1][2] * input_rgb.b,
        params.out_cam[2][0] * input_rgb.r + params.out_cam[2][1] * input_rgb.g + params.out_cam[2][2] * input_rgb.b
    );
    
    // Apply gamma correction if enabled
    if (params.apply_gamma) {
        output_rgb = pow(max(output_rgb, 0.0f), 1.0f / params.gamma_power);
    }
    
    // Clamp and convert back to 16-bit
    output_rgb = clamp(output_rgb, 0.0f, 1.0f);
    output_image[rgb_idx + 0] = uint16_t(output_rgb.r * 65535.0);
    output_image[rgb_idx + 1] = uint16_t(output_rgb.g * 65535.0);
    output_image[rgb_idx + 2] = uint16_t(output_rgb.b * 65535.0);
}

// Combined white balance and color conversion
kernel void white_balance_and_color_convert(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant float* wb_multipliers [[buffer(2)]],
    constant WhiteBalanceColorParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t rgb_idx = pixel_idx * 3;
    
    // Read input RGB values and normalize
    float3 input_rgb = float3(
        float(input_image[rgb_idx + 0]) / 65535.0,
        float(input_image[rgb_idx + 1]) / 65535.0,
        float(input_image[rgb_idx + 2]) / 65535.0
    );
    
    // Apply white balance multipliers
    float3 wb_rgb = float3(
        input_rgb.r * wb_multipliers[0],  // Red
        input_rgb.g * wb_multipliers[1],  // Green
        input_rgb.b * wb_multipliers[2]   // Blue
    );
    
    // Apply color matrix transformation
    float3 output_rgb = float3(
        params.color_matrix[0][0] * wb_rgb.r + params.color_matrix[0][1] * wb_rgb.g + params.color_matrix[0][2] * wb_rgb.b,
        params.color_matrix[1][0] * wb_rgb.r + params.color_matrix[1][1] * wb_rgb.g + params.color_matrix[1][2] * wb_rgb.b,
        params.color_matrix[2][0] * wb_rgb.r + params.color_matrix[2][1] * wb_rgb.g + params.color_matrix[2][2] * wb_rgb.b
    );
    
    // Apply gamma correction if enabled
    if (params.apply_gamma) {
        output_rgb = pow(max(output_rgb, 0.0f), 1.0f / params.gamma_power);
    }
    
    // Clamp and convert back to 16-bit
    output_rgb = clamp(output_rgb, 0.0f, 1.0f);
    output_image[rgb_idx + 0] = uint16_t(output_rgb.r * 65535.0);
    output_image[rgb_idx + 1] = uint16_t(output_rgb.g * 65535.0);
    output_image[rgb_idx + 2] = uint16_t(output_rgb.b * 65535.0);
}

// LibRaw convert_to_rgb equivalent
kernel void libraw_convert_to_rgb(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant ColorConversionParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t rgb_idx = pixel_idx * 3;
    
    // Read input RGB values and normalize
    float3 input_rgb = float3(
        float(input_image[rgb_idx + 0]) / 65535.0,
        float(input_image[rgb_idx + 1]) / 65535.0,
        float(input_image[rgb_idx + 2]) / 65535.0
    );
    
    // Apply out_cam matrix transformation (LibRaw style)
    float3 output_rgb = float3(
        params.out_cam[0][0] * input_rgb.r + params.out_cam[0][1] * input_rgb.g + params.out_cam[0][2] * input_rgb.b + params.out_cam[0][3],
        params.out_cam[1][0] * input_rgb.r + params.out_cam[1][1] * input_rgb.g + params.out_cam[1][2] * input_rgb.b + params.out_cam[1][3],
        params.out_cam[2][0] * input_rgb.r + params.out_cam[2][1] * input_rgb.g + params.out_cam[2][2] * input_rgb.b + params.out_cam[2][3]
    );
    
    // Apply gamma correction if enabled
    if (params.apply_gamma) {
        output_rgb = pow(max(output_rgb, 0.0f), 1.0f / params.gamma_power);
    }
    
    // Clamp and convert back to 16-bit
    output_rgb = clamp(output_rgb, 0.0f, 1.0f);
    output_image[rgb_idx + 0] = uint16_t(output_rgb.r * 65535.0);
    output_image[rgb_idx + 1] = uint16_t(output_rgb.g * 65535.0);
    output_image[rgb_idx + 2] = uint16_t(output_rgb.b * 65535.0);
}

// LibRaw color conversion with matrix selection
kernel void libraw_color_convert_with_matrix_selection(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant float* rgb_cam_matrix [[buffer(2)]],
    constant ColorConversionParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t rgb_idx = pixel_idx * 3;
    
    // Read input RGB values and normalize
    float3 input_rgb = float3(
        float(input_image[rgb_idx + 0]) / 65535.0,
        float(input_image[rgb_idx + 1]) / 65535.0,
        float(input_image[rgb_idx + 2]) / 65535.0
    );
    
    // Apply rgb_cam matrix (3x4)
    float3 output_rgb = float3(
        rgb_cam_matrix[0] * input_rgb.r + rgb_cam_matrix[1] * input_rgb.g + rgb_cam_matrix[2] * input_rgb.b + rgb_cam_matrix[3],
        rgb_cam_matrix[4] * input_rgb.r + rgb_cam_matrix[5] * input_rgb.g + rgb_cam_matrix[6] * input_rgb.b + rgb_cam_matrix[7],
        rgb_cam_matrix[8] * input_rgb.r + rgb_cam_matrix[9] * input_rgb.g + rgb_cam_matrix[10] * input_rgb.b + rgb_cam_matrix[11]
    );
    
    // Apply gamma correction if enabled
    if (params.apply_gamma) {
        output_rgb = pow(max(output_rgb, 0.0f), 1.0f / params.gamma_power);
    }
    
    // Clamp and convert back to 16-bit
    output_rgb = clamp(output_rgb, 0.0f, 1.0f);
    output_image[rgb_idx + 0] = uint16_t(output_rgb.r * 65535.0);
    output_image[rgb_idx + 1] = uint16_t(output_rgb.g * 65535.0);
    output_image[rgb_idx + 2] = uint16_t(output_rgb.b * 65535.0);
}