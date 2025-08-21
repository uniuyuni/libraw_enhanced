//
// gamma_correction.metal
// LibRaw Enhanced - Gamma Correction Metal Shaders
//
// Pure gamma correction kernels separated from color space conversion
//

#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// Gamma correction parameters
struct GammaCorrectionParams {
    uint32_t width;
    uint32_t height;
    uint32_t channels;          // Number of channels (usually 3 for RGB)
    float gamma_power;          // Gamma power (e.g., 2.2, 2.4)
    float gamma_slope;          // Gamma toe slope (e.g., 4.5, 12.92)
    uint32_t gamma_mode;        // 0: generic, 1: sRGB, 2: custom
    uint32_t normalize_input;   // Whether input is normalized (0-1) or raw (0-65535)
    uint32_t denormalize_output; // Whether output should be denormalized
};

// sRGB gamma correction function (standard)
float apply_srgb_gamma_encode(float linear_value) {
    if (linear_value <= 0.0031308) {
        return 12.92 * linear_value;
    } else {
        return 1.055 * pow(linear_value, 1.0/2.4) - 0.055;
    }
}

// sRGB gamma correction decode (inverse)
float apply_srgb_gamma_decode(float gamma_value) {
    if (gamma_value <= 0.04045) {
        return gamma_value / 12.92;
    } else {
        return pow((gamma_value + 0.055) / 1.055, 2.4);
    }
}

// Generic gamma correction (LibRaw style)
float apply_generic_gamma_encode(float linear_value, float power, float slope) {
    if (linear_value < 1.0 / slope) {
        return linear_value * slope;
    } else {
        return pow(linear_value, 1.0 / power);
    }
}

// Generic gamma correction decode (inverse)
float apply_generic_gamma_decode(float gamma_value, float power, float slope) {
    if (gamma_value < 1.0) {
        return gamma_value / slope;
    } else {
        return pow(gamma_value, power);
    }
}

// Gamma correction encode kernel (linear -> gamma)
kernel void apply_gamma_correction_encode(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant GammaCorrectionParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t base_idx = pixel_idx * params.channels;
    
    for (uint32_t c = 0; c < params.channels; c++) {
        uint32_t idx = base_idx + c;
        
        // Read input value
        float input_value;
        if (params.normalize_input) {
            input_value = float(input_image[idx]) / 65535.0;
        } else {
            input_value = float(input_image[idx]);
        }
        
        // Apply gamma correction
        float gamma_corrected;
        if (params.gamma_mode == 1) { // sRGB
            gamma_corrected = apply_srgb_gamma_encode(input_value);
        } else { // Generic gamma
            gamma_corrected = apply_generic_gamma_encode(input_value, params.gamma_power, params.gamma_slope);
        }
        
        // Write output value
        if (params.denormalize_output) {
            gamma_corrected = clamp(gamma_corrected, 0.0, 1.0);
            output_image[idx] = uint16_t(gamma_corrected * 65535.0);
        } else {
            gamma_corrected = clamp(gamma_corrected, 0.0, 65535.0);
            output_image[idx] = uint16_t(gamma_corrected);
        }
    }
}

// Gamma correction decode kernel (gamma -> linear)
kernel void apply_gamma_correction_decode(
    const device uint16_t* input_image [[buffer(0)]],
    device uint16_t* output_image [[buffer(1)]],
    constant GammaCorrectionParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t base_idx = pixel_idx * params.channels;
    
    for (uint32_t c = 0; c < params.channels; c++) {
        uint32_t idx = base_idx + c;
        
        // Read input value
        float input_value;
        if (params.normalize_input) {
            input_value = float(input_image[idx]) / 65535.0;
        } else {
            input_value = float(input_image[idx]);
        }
        
        // Apply gamma decode (inverse)
        float linear_value;
        if (params.gamma_mode == 1) { // sRGB
            linear_value = apply_srgb_gamma_decode(input_value);
        } else { // Generic gamma
            linear_value = apply_generic_gamma_decode(input_value, params.gamma_power, params.gamma_slope);
        }
        
        // Write output value
        if (params.denormalize_output) {
            linear_value = clamp(linear_value, 0.0, 1.0);
            output_image[idx] = uint16_t(linear_value * 65535.0);
        } else {
            linear_value = clamp(linear_value, 0.0, 65535.0);
            output_image[idx] = uint16_t(linear_value);
        }
    }
}

// In-place gamma correction encode kernel (for performance)
kernel void apply_gamma_correction_encode_inplace(
    device uint16_t* image [[buffer(0)]],
    constant GammaCorrectionParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * params.width + gid.x;
    uint32_t base_idx = pixel_idx * params.channels;
    
    for (uint32_t c = 0; c < params.channels; c++) {
        uint32_t idx = base_idx + c;
        
        // Read input value (normalized to 0-1)
        float input_value = float(image[idx]) / 65535.0;
        
        // Apply gamma correction
        float gamma_corrected;
        if (params.gamma_mode == 1) { // sRGB
            gamma_corrected = apply_srgb_gamma_encode(input_value);
        } else { // Generic gamma
            gamma_corrected = apply_generic_gamma_encode(input_value, params.gamma_power, params.gamma_slope);
        }
        
        // Write back (clamped to 0-65535)
        gamma_corrected = clamp(gamma_corrected, 0.0, 1.0);
        image[idx] = uint16_t(gamma_corrected * 65535.0);
    }
}