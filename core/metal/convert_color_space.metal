//
// color_space_convert.metal
// LibRaw Enhanced - Color Space Conversion
// GPU port of cpu_accelerator.cpp convert_color_space
//
#include "shader_types.h"
#include "shader_common.h"

kernel void convert_color_space(
    const device float* rgb_input [[buffer(0)]],
    device float* rgb_output [[buffer(1)]],
    constant ColorSpaceParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    const uint pixel_idx = (gid.y * params.width + gid.x) * 3;
    
    // 入力RGBをfloat3ベクタとして読み込み
    float3 rgb_in = {
        rgb_input[pixel_idx + 0],
        rgb_input[pixel_idx + 1],
        rgb_input[pixel_idx + 2]
    };
    
    // 行列の行をfloat4ベクタとして定義
    const float4 row0 = params.transform[0];
    const float4 row1 = params.transform[1];
    const float4 row2 = params.transform[2];
    
    // ベクタ演算で変換計算（オフセット込み）
    float3 rgb_out = {
        dot(row0.rgb, rgb_in) + row0.w,
        dot(row1.rgb, rgb_in) + row1.w,
        dot(row2.rgb, rgb_in) + row2.w
    };
    
    // 結果をクランプして出力
    //rgb_out = clamp(rgb_out, 0.0f, 1.0f);
    rgb_output[pixel_idx + 0] = rgb_out.x;
    rgb_output[pixel_idx + 1] = rgb_out.y;
    rgb_output[pixel_idx + 2] = rgb_out.z;
}
