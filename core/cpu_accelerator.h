//
// cpu_accelerator.h
// LibRaw Enhanced - CPU-specific acceleration implementation
//

#pragma once

// Include common definitions from accelerator.h
#include "accelerator.h"
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace libraw_enhanced {

// テンプレート化された汎用ヘルパー関数群
template<typename T>
inline T sqr(T x) {
    return x * x;
}

// これにより、ulim_generic(int, uint16_t, uint16_t) のような呼び出しでもエラーになりません。
template<typename T, typename U, typename V>
inline constexpr T ulim_generic(T val, U upper, V lower) {
    if (val > upper) return static_cast<T>(upper);
    if (val < lower) return static_cast<T>(lower);
    return val;
}

// Helper function to find the median of three values
template<typename T>
inline T median(T a, T b, T c) {
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

// Bayerパターン判定関数 (変更なし)
inline int fcol_bayer(int row, int col, uint32_t filters) {
    return (filters >> ((((row) << 1 & 14) | ((col) & 1)) << 1) & 3);
};

class CPUAccelerator {
public:
    CPUAccelerator();
    ~CPUAccelerator();
    
    // 初期化・状態管理
    bool initialize();
    bool is_available() const;
    void release_resources();

    // Pre-processing methods
    bool pre_interpolate(ImageBuffer& image_buffer, uint32_t filters, 
                        const char (&xtrans)[6][6], bool half_size = false);
                         
    // Bayer specific acceleration methods
    bool demosaic_bayer_linear(const ImageBuffer& raw_buffer,
                            ImageBuffer& rgb_buffer,
                            uint32_t filters);

    bool demosaic_bayer_aahd(const ImageBuffer& raw_buffer,
                            ImageBuffer& rgb_buffer,
                            uint32_t filters);
                                       
    bool demosaic_bayer_dcb(const ImageBuffer& raw_buffer,
                            ImageBuffer& rgb_buffer,
                            uint32_t filters,
                            int iterations = 1,
                            bool dcb_enhance = true);
                         
    bool demosaic_bayer_amaze(const ImageBuffer& raw_buffer,
                            ImageBuffer& rgb_buffer,
                            uint32_t filters);
                           
    // X-Trans specific acceleration methods
    bool demosaic_xtrans_3pass(const ImageBuffer& raw_buffer,
                                ImageBuffer& rgb_buffer,
                                const char (&xtrans)[6][6],
                                const float (&color_matrix)[3][4]);
                           
    bool demosaic_xtrans_1pass(const ImageBuffer& raw_buffer,
                                ImageBuffer& rgb_buffer,
                                const char (&xtrans)[6][6],
                                const float (&color_matrix)[3][4]);
                               
    // White balance methods
    bool apply_white_balance(const ImageBufferFloat32& rgb_input,
                            ImageBufferFloat32& rgb_output,
                            const float wb_multipliers[4]);
                            
    // Camera matrix-based color space conversion
    bool convert_color_space(const ImageBufferFloat32& rgb_input,
                            ImageBufferFloat32& rgb_output,
                            const float transform[3][4]);

    // Gamma correction method
    bool gamma_correct(const ImageBufferFloat32& rgb_input,
                        ImageBufferFloat32& rgb_output,
                        float gamma_power = 2.2f,
                        float gamma_slope = 4.5f,
                        int output_color_space = 1);
                           
    double get_last_processing_time() const;
    size_t get_memory_usage() const;
    void set_debug_mode(bool enable);
    std::string get_device_info() const;
    
private:
    float apply_srgb_gamma_encode(float linear_value) const;
    float apply_pure_power_gamma_encode(float linear_value, float power) const;
    float apply_rec2020_gamma_encode(float linear_value) const;
    float apply_aces_gamma_encode(float linear_value) const;
    
    static constexpr float d65_white[3] = {0.95047f, 1.0f, 1.08883f};
    static constexpr float xyz_rgb[3][3] = {
        {0.4124564f, 0.3575761f, 0.1804375f},
        {0.2126729f, 0.7151522f, 0.0721750f},
        {0.0193339f, 0.1191920f, 0.9503041f}
    };
    static constexpr float rgb_cam_default_[3][4] = {
        {1.f, 0.f, 0.f, 0.f},
        {0.f, 1.f, 0.f, 0.f},
        {0.f, 0.f, 1.f, 0.f}
    };

    void border_interpolate(const ImageBuffer& raw_buffer, uint32_t filters, int border);
    void linear_interpolate_loop_cpu(const ImageBuffer& raw_buffer, int* code, int size, int colors);
 
    bool initialized_ = false;
    bool debug_mode_ = false;
    double last_processing_time_ = 0.0;
    std::string device_name_ = "Apple Silicon CPU";
    BufferManager buffer_manager_;
};

bool is_apple_silicon();
bool is_accelerate_available();
std::vector<std::string> get_cpu_device_list();

} // namespace libraw_enhanced