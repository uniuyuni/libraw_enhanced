
kernel void preprocess_enhance_micro_contrast(
    const device packed_float3* rgb_input [[buffer(0)]],
    device packed_float4* rgb_output [[buffer(1)]],
    constant EnhanceMicroContrastParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint pixel_idx = (gid.y * params.width + gid.x);

    float3 rgb = rgb_input[pixel_idx];

    rgb = max(0.f, rgb);
//    rgb = select(rgb, pow(rgb, params.pow_scale), rgb.g > 1.f);
    rgb = select(rgb, params.threshold + pow(2.f, rgb - params.threshold) - 1, rgb.g >= params.threshold);

    rgb_output[pixel_idx] = float4(rgb.r, rgb.g, rgb.b, 1.f);    
}

kernel void enhance_micro_contrast(
    const device packed_float3* rgb_input [[buffer(0)]],
    const device packed_float4* local_mean [[buffer(1)]],
    device packed_float4* local_var [[buffer(2)]],
    device packed_float3* rgb_output [[buffer(3)]],
    device EnhanceMicroContrastParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint idx = (gid.y * params.width + gid.x);

    // 局所標準偏差の計算（後半）
    local_var[idx] = local_var[idx] - local_mean[idx] * local_mean[idx];

    // コントラストマップの正規化
    float4 local_std = sqrt(max(local_var[idx], 0.f));
    params.max_local_std = max(params.max_local_std, max(max(local_std.x, local_std.y), local_std.z));
    threadgroup_barrier(mem_flags::mem_device);

    float4 contrast_map = local_std / params.max_local_std;

    // 強調係数の計算 - コントラストが低い領域ほど強く強調
    float4 enhance_factor = select(0.f,
                                   params.strength * (params.target_contrast - contrast_map) / params.target_contrast, 
                                   contrast_map < params.target_contrast);

    // 高周波成分の抽出
    float4 high_freq = float4(rgb_input[idx].x, rgb_input[idx].y, rgb_input[idx].z, 0.f) - local_mean[idx];

    // 適応的な強調
    float4 enhanced_high_freq = high_freq * (1.f + enhance_factor);

    // 画像の再構成
    if (rgb_input[idx].y >= params.threshold) {
        rgb_output[idx].x = local_mean[idx].x + enhanced_high_freq.x;
        rgb_output[idx].y = local_mean[idx].y + enhanced_high_freq.y;
        rgb_output[idx].z = local_mean[idx].z + enhanced_high_freq.z;
    } else {
        rgb_output[idx] = rgb_input[idx];
    }
}
