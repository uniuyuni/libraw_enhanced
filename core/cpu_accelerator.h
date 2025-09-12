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

namespace libraw_enhanced {


class CPUAccelerator {
public:
    CPUAccelerator();
    ~CPUAccelerator();
    
    // 初期化・状態管理
    bool initialize();
    bool is_available() const;
    void release_resources();

    // White balance methods
    bool apply_white_balance(const ImageBuffer& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            const float wb_multipliers[4],
                            uint32_t filters,
                            const char xtrans[6][6]);

    // Pre-processing methods
    bool pre_interpolate(ImageBufferFloat& rgb_buffer, uint32_t filters, 
                        const char (&xtrans)[6][6], bool half_size = false);
                         
    // Bayer specific acceleration methods
    bool demosaic_bayer_linear(const ImageBuffer& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            uint16_t maximum_value);

    bool demosaic_bayer_aahd(const ImageBuffer& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            uint16_t maximum_value);
                                       
    bool demosaic_bayer_dcb(const ImageBuffer& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            uint16_t maximum_value,
                            int iterations = 1,
                            bool dcb_enhance = true);
                         
    bool demosaic_bayer_amaze(const ImageBuffer& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            const float (&cam_mul)[4],
                            uint16_t maximum_value);
                           
    // X-Trans specific acceleration methods
    bool demosaic_xtrans_3pass(const ImageBuffer& raw_buffer,
                                ImageBufferFloat& rgb_buffer,
                                const char (&xtrans)[6][6],
                                const float (&color_matrix)[3][4],
                                uint16_t maximum_value);
                           
    bool demosaic_xtrans_1pass(const ImageBuffer& raw_buffer,
                                ImageBufferFloat& rgb_buffer,
                                const char (&xtrans)[6][6],
                                const float (&color_matrix)[3][4],
                                uint16_t maximum_value);

    bool demosaic_xtrans_adaptive(const ImageBuffer& raw_buffer,
                                   ImageBufferFloat& rgb_buffer,
                                   const char (&xtrans)[6][6],
                                   const float (&color_matrix)[3][4],
                                   uint16_t maximum_value);
                            
    // Camera matrix-based color space conversion
    bool convert_color_space(const ImageBufferFloat& rgb_input,
                            ImageBufferFloat& rgb_output,
                            const float transform[3][4]);

    // Gamma correction method
    bool gamma_correct(const ImageBufferFloat& rgb_input,
                        ImageBufferFloat& rgb_output,
                        float gamma_power = 0.f, //2.2f,
                        float gamma_slope = 0.f, // 4.5f,
                        int output_color_space = 1);

    bool tone_mapping(const ImageBufferFloat& rgb_input,
                            ImageBufferFloat& rgb_output,
                            float after_scale);

    double get_last_processing_time() const;
    size_t get_memory_usage() const;
    std::string get_device_info() const;
        
private: 
    bool initialized_ = false;
    double last_processing_time_ = 0.0;
    std::string device_name_ = "Apple Silicon CPU";

    void apply_wb_bayer(const ImageBuffer& raw_buffer,
                        ImageBufferFloat& rgb_buffer,
                        const float wb_multipliers[4],
                        uint32_t filters);

    void apply_wb_xtrans(const ImageBuffer& raw_buffer,
                        ImageBufferFloat& rgb_buffer,
                        const float wb_multipliers[4],
                        const char xtrans[6][6]);

    float apply_srgb_gamma_encode(float linear_value) const;
    float apply_aces_gamma_encode(float linear_value) const;
    float apply_rec2020_gamma_encode(float linear_value) const;
    float apply_pure_power_gamma_encode(float linear_value, float power) const;
    float apply_pure_power_gamma_encode_with_slope(float v, float p, float s) const;
    
    void border_interpolate(const ImageBufferFloat& raw_buffer, uint32_t filters, int border);
};

} // namespace libraw_enhanced