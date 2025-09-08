//
// gamma_correction.metal (最適化版)
// LibRaw Enhanced - Gamma Correction
// GPU port of cpu_accelerator.cpp gamma_correct
//

#include "shader_common.h"
#include "../constants.h"

// sRGB gamma encode
inline float3 apply_srgb_gamma_encode(float3 linear_value) {
    return select(1.055f * pow(linear_value, 1.0f / 2.4f) - 0.055f,
                  12.92f * linear_value,
                  linear_value <= 0.0031308f);
}

// ACES gamma encode
inline float3 apply_aces_gamma_encode(float3 linear_value) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    
    float3 numerator = linear_value * (a * linear_value + b);
    float3 denominator = linear_value * (c * linear_value + d) + e;
    return numerator / denominator;
}

// Rec. 2020 gamma encode
inline float3 apply_rec2020_gamma_encode(float3 linear_value) {
    const float alpha = 1.09929682680944f;
    const float beta = 0.018053968510807f;
    
    return select(alpha * pow(linear_value, 0.45f) - (alpha - 1.0f),
                  4.5f * linear_value,
                  linear_value < beta);
}

// Pure power gamma encode
inline float3 apply_pure_power_gamma_encode(float3 linear_value, float power) {
    return pow(linear_value, 1.0f / power);
}

// Pure power gamma encode with slope support
inline float3 apply_pure_power_gamma_encode_with_slope(float3 linear_value, float power, float slope) {
    if (slope <= 0.f) {
        return apply_pure_power_gamma_encode(linear_value, power);
    }

    float threshold = slope / power;    
    return select(pow((power * linear_value + (1.0f - slope)) / power, 1.0f / power),
                  power * linear_value / slope,
                  linear_value < threshold);
}


kernel void gamma_correct(
    const device float* rgb_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant GammaParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint pixel_idx = (gid.y * params.width + gid.x) * 3;
    
    float3 rgb_in = float3(
        rgb_input[pixel_idx + 0],
        rgb_input[pixel_idx + 1],
        rgb_input[pixel_idx + 2]
    );
    
    float3 rgb_out;
    
    // 色空間に応じたガンマ補正
    switch (params.output_color_space) {
    case ColorSpace::sRGB: // sRGB
    case ColorSpace::P3D65: // Display P3
        rgb_out = apply_srgb_gamma_encode(rgb_in);
        break;
    case ColorSpace::AdobeRGB: // Adobe RGB
    case ColorSpace::WideGamutRGB: // Wide Gamut RGB
        rgb_out = apply_pure_power_gamma_encode(rgb_in, 2.222f);
        break;
    case ColorSpace::ProPhotoRGB: // ProPhoto RGB
        rgb_out = apply_pure_power_gamma_encode(rgb_in, 1.8f);
        break;
    //case ColorSpace::ACES: // ACES
    //    rgb_out = apply_aces_gamma_encode(rgb_in);
    //    break;
    case ColorSpace::Rec2020: // Rec. 2020
        rgb_out = apply_rec2020_gamma_encode(rgb_in);
        break;
    default:
        rgb_out = apply_pure_power_gamma_encode_with_slope(rgb_in, params.gamma_power, params.gamma_slope);
        break;
    }
    
    // 結果を書き込み
    //rgb_out = clamp(rgb_out, 0.0f, 1.0f);
    rgb_output[pixel_idx + 0] = rgb_out.r;
    rgb_output[pixel_idx + 1] = rgb_out.g;
    rgb_output[pixel_idx + 2] = rgb_out.b;
}