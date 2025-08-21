//
// matrix_transform.metal
// LibRaw Enhanced - Matrix Transformation Metal Shaders
//
// Pure matrix transformation kernels for color space conversion
//

#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// 3x3 Matrix transformation parameters
struct Matrix3x3Params {
    float matrix[3][3];         // Transformation matrix
    uint32_t width;
    uint32_t height;
    uint32_t input_channels;    // Usually 3 (RGB)
    uint32_t output_channels;   // Usually 3 (RGB)
    uint32_t normalize_input;   // Whether to normalize input (0-65535 -> 0-1)
    uint32_t denormalize_output; // Whether to denormalize output (0-1 -> 0-65535)
    uint32_t padding[2];        // Alignment padding
};

// 3x4 Matrix transformation parameters (for camera matrices)
struct Matrix3x4Params {
    float matrix[3][4];         // Camera transformation matrix
    uint32_t width;
    uint32_t height;
    uint32_t input_channels;    // Usually 3 or 4
    uint32_t output_channels;   // Usually 3
    uint32_t normalize_input;   // Whether to normalize input
    uint32_t denormalize_output; // Whether to denormalize output
    uint32_t padding;           // Alignment padding
};

// Pure 3x3 matrix transformation kernel
kernel void apply_3x3_matrix_transform(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant Matrix3x3Params& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t input_idx = pixel_idx * params.input_channels;
    uint32_t output_idx = pixel_idx * params.output_channels;
    
    // Read input values
    float3 input_rgb;
    if (params.normalize_input) {
        input_rgb = float3(
            float(input_image[input_idx + 0]) / 65535.0,
            float(input_image[input_idx + 1]) / 65535.0,
            float(input_image[input_idx + 2]) / 65535.0
        );
    } else {
        input_rgb = float3(
            float(input_image[input_idx + 0]),
            float(input_image[input_idx + 1]),
            float(input_image[input_idx + 2])
        );
    }
    
    // Apply 3x3 matrix transformation
    float3 output_rgb = float3(
        params.matrix[0][0] * input_rgb.r + params.matrix[0][1] * input_rgb.g + params.matrix[0][2] * input_rgb.b,
        params.matrix[1][0] * input_rgb.r + params.matrix[1][1] * input_rgb.g + params.matrix[1][2] * input_rgb.b,
        params.matrix[2][0] * input_rgb.r + params.matrix[2][1] * input_rgb.g + params.matrix[2][2] * input_rgb.b
    );
    
    // Output values
    if (params.denormalize_output) {
        output_rgb = clamp(output_rgb, 0.0, 1.0);
        output_image[output_idx + 0] = uint16_t(output_rgb.r * 65535.0);
        output_image[output_idx + 1] = uint16_t(output_rgb.g * 65535.0);
        output_image[output_idx + 2] = uint16_t(output_rgb.b * 65535.0);
    } else {
        output_rgb = clamp(output_rgb, 0.0, 65535.0);
        output_image[output_idx + 0] = uint16_t(output_rgb.r);
        output_image[output_idx + 1] = uint16_t(output_rgb.g);
        output_image[output_idx + 2] = uint16_t(output_rgb.b);
    }
}

// Pure 3x4 matrix transformation kernel (for camera matrices)
kernel void apply_3x4_matrix_transform(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant Matrix3x4Params& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t input_idx = pixel_idx * params.input_channels;
    uint32_t output_idx = pixel_idx * params.output_channels;
    
    // Read input values (support 3 or 4 channels)
    float4 input_rgba;
    if (params.normalize_input) {
        input_rgba = float4(
            float(input_image[input_idx + 0]) / 65535.0,
            float(input_image[input_idx + 1]) / 65535.0,
            float(input_image[input_idx + 2]) / 65535.0,
            params.input_channels > 3 ? float(input_image[input_idx + 3]) / 65535.0 : 0.0
        );
    } else {
        input_rgba = float4(
            float(input_image[input_idx + 0]),
            float(input_image[input_idx + 1]),
            float(input_image[input_idx + 2]),
            params.input_channels > 3 ? float(input_image[input_idx + 3]) : 0.0
        );
    }
    
    // Apply 3x4 matrix transformation
    float3 output_rgb = float3(
        params.matrix[0][0] * input_rgba.r + params.matrix[0][1] * input_rgba.g + params.matrix[0][2] * input_rgba.b + params.matrix[0][3] * input_rgba.a,
        params.matrix[1][0] * input_rgba.r + params.matrix[1][1] * input_rgba.g + params.matrix[1][2] * input_rgba.b + params.matrix[1][3] * input_rgba.a,
        params.matrix[2][0] * input_rgba.r + params.matrix[2][1] * input_rgba.g + params.matrix[2][2] * input_rgba.b + params.matrix[2][3] * input_rgba.a
    );
    
    // Output values
    if (params.denormalize_output) {
        output_rgb = clamp(output_rgb, 0.0, 1.0);
        output_image[output_idx + 0] = uint16_t(output_rgb.r * 65535.0);
        output_image[output_idx + 1] = uint16_t(output_rgb.g * 65535.0);
        output_image[output_idx + 2] = uint16_t(output_rgb.b * 65535.0);
    } else {
        output_rgb = clamp(output_rgb, 0.0, 65535.0);
        output_image[output_idx + 0] = uint16_t(output_rgb.r);
        output_image[output_idx + 1] = uint16_t(output_rgb.g);
        output_image[output_idx + 2] = uint16_t(output_rgb.b);
    }
}

// Combined matrix multiplication kernel (rgb_cam * out_rgb)
// This performs: out_cam[i][j] = sum(out_rgb[i][k] * rgb_cam[k][j]) for k=0..2
kernel void multiply_matrices_3x3_3x4(
    constant float* rgb_cam_matrix [[buffer(0)]],        // 3x4 camera matrix (flattened)
    constant float* out_rgb_matrix [[buffer(1)]],        // 3x3 output color space matrix (flattened)
    device float* result_matrix [[buffer(2)]],           // 3x4 result matrix (flattened)
    uint gid [[thread_position_in_grid]]
) {
    // This kernel computes a single element of the result matrix
    // Total threads needed: 3 * 4 = 12
    
    if (gid >= 12) return;
    
    uint i = gid / 4;  // Row in result matrix (0-2)
    uint j = gid % 4;  // Column in result matrix (0-3)
    
    float sum = 0.0;
    for (uint k = 0; k < 3; k++) {
        // out_rgb[i][k] * rgb_cam[k][j]
        sum += out_rgb_matrix[i * 3 + k] * rgb_cam_matrix[k * 4 + j];
    }
    
    result_matrix[i * 4 + j] = sum;
}