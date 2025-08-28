//
// gamma_correction.metal
// LibRaw Enhanced - Gamma Correction
// GPU port of cpu_accelerator.cpp gamma_correct
//

#include "shader_common.h"

// sRGB gamma encode (exact CPU port)
inline float apply_srgb_gamma_encode(float linear_value) {
    if (linear_value <= 0.0031308f) {
        return 12.92f * linear_value;
    } else {
        return 1.055f * pow(linear_value, 1.0f / 2.4f) - 0.055f;
    }
}

// Pure power gamma encode (exact CPU port)
inline float apply_pure_power_gamma_encode(float linear_value, float power) {
    return pow(max(linear_value, 0.0f), 1.0f / power);
}

// Rec. 2020 gamma encode (exact CPU port)
inline float apply_rec2020_gamma_encode(float linear_value) {
    const float alpha = 1.09929682680944f;
    const float beta = 0.018053968510807f;
    
    if (linear_value < beta) {
        return 4.5f * linear_value;
    } else {
        return alpha * pow(linear_value, 0.45f) - (alpha - 1.0f);
    }
}

// ACES gamma encode (simplified)
inline float apply_aces_gamma_encode(float linear_value) {
    // Simplified ACES tone mapping
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    
    return clamp((linear_value * (a * linear_value + b)) / 
                 (linear_value * (c * linear_value + d) + e), 0.0f, 1.0f);
}

kernel void gamma_correct(
    const device float* rgb_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant GammaParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint pixel_count = params.width * params.height;
    
    if (gid >= pixel_count) return;
    
    // Skip gamma correction for linear color spaces (exact CPU logic)
    if (params.output_color_space == 0 || params.output_color_space == 5) {
        const uint pixel_idx = gid * 3;
        if (rgb_input != rgb_output) {
            rgb_output[pixel_idx + 0] = rgb_input[pixel_idx + 0];
            rgb_output[pixel_idx + 1] = rgb_input[pixel_idx + 1];
            rgb_output[pixel_idx + 2] = rgb_input[pixel_idx + 2];
        }
        return;
    }
    
    const uint pixel_idx = gid * 3;
    
    float3 rgb_in = {
        rgb_input[pixel_idx + 0],
        rgb_input[pixel_idx + 1],
        rgb_input[pixel_idx + 2]
    };
    
    float3 rgb_out;
    
    // Apply gamma correction based on color space (exact CPU switch logic)
    switch (params.output_color_space) {
        case 1: // sRGB
            rgb_out.r = apply_srgb_gamma_encode(rgb_in.r);
            rgb_out.g = apply_srgb_gamma_encode(rgb_in.g);
            rgb_out.b = apply_srgb_gamma_encode(rgb_in.b);
            break;
            
        case 2: // Adobe RGB (gamma 2.2)
        case 3: // Wide Gamut RGB (gamma 2.2)
        case 4: // ProPhoto RGB (gamma 1.8)
        case 6: // XYZ (linear)
        default:
            {
                float gamma_power = params.gamma_power;
                if (params.output_color_space == 4) gamma_power = 1.8f; // ProPhoto RGB
                
                rgb_out.r = apply_pure_power_gamma_encode(rgb_in.r, gamma_power);
                rgb_out.g = apply_pure_power_gamma_encode(rgb_in.g, gamma_power);
                rgb_out.b = apply_pure_power_gamma_encode(rgb_in.b, gamma_power);
            }
            break;
            
        case 7: // Rec. 2020
            rgb_out.r = apply_rec2020_gamma_encode(rgb_in.r);
            rgb_out.g = apply_rec2020_gamma_encode(rgb_in.g);
            rgb_out.b = apply_rec2020_gamma_encode(rgb_in.b);
            break;
            
        case 8: // ACES
            rgb_out.r = apply_aces_gamma_encode(rgb_in.r);
            rgb_out.g = apply_aces_gamma_encode(rgb_in.g);
            rgb_out.b = apply_aces_gamma_encode(rgb_in.b);
            break;
    }
    
    // Clamp final output
    rgb_output[pixel_idx + 0] = clamp(rgb_out.r, 0.0f, 1.0f);
    rgb_output[pixel_idx + 1] = clamp(rgb_out.g, 0.0f, 1.0f);
    rgb_output[pixel_idx + 2] = clamp(rgb_out.b, 0.0f, 1.0f);
}