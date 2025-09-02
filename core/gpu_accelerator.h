//
// gpu_accelerator.h
// LibRaw Enhanced - True GPU Metal Accelerator Interface
//

#pragma once

#include "cpu_accelerator.h"
#include <memory>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace libraw_enhanced {

class GPUAccelerator {
public:
    GPUAccelerator();
    ~GPUAccelerator();
    
    // Initialization and device info
    bool initialize();
    bool is_available() const;
    std::string get_device_info() const;
    
    // Bayer demosaicing methods
    bool demosaic_bayer_linear(const ImageBuffer& raw_buffer,
                              ImageBufferFloat& rgb_buffer, 
                              uint32_t filters,
                              uint16_t maximum_value);
                           
    bool demosaic_bayer_amaze(const ImageBuffer& raw_buffer,
                             ImageBufferFloat& rgb_buffer,
                             uint32_t filters,
                             const float (&cam_mul)[4],
                             uint16_t maximum_value);
    
    // X-Trans demosaicing methods
    bool demosaic_xtrans_1pass(const ImageBuffer& raw_buffer,
                              ImageBufferFloat& rgb_buffer,
                              const char (&xtrans)[6][6],
                              const float (&color_matrix)[3][4],
                              uint16_t maximum_value);
                              
    bool demosaic_xtrans_3pass(const ImageBuffer& raw_buffer,
                              ImageBufferFloat& rgb_buffer,
                              const char (&xtrans)[6][6],
                              const float (&color_matrix)[3][4],
                              uint16_t maximum_value);

    bool demosaic_xtrans_adaptive(const ImageBuffer& raw_buffer,
                                 ImageBufferFloat& rgb_buffer,
                                 const char (&xtrans)[6][6],
                                 const float (&color_matrix)[3][4],
                                 uint16_t maximum_value);
    
    // Float processing pipeline methods (used by accelerator.cpp)
    bool apply_white_balance(const ImageBufferFloat& rgb_input,
                            ImageBufferFloat& rgb_output,
                            const float wb_multipliers[4]);
                            
    bool convert_color_space(const ImageBufferFloat& rgb_input,
                            ImageBufferFloat& rgb_output,
                            const float transform[3][4]);
                            
    bool gamma_correct(const ImageBufferFloat& rgb_input,
                      ImageBufferFloat& rgb_output,
                      float gamma_power = 2.2f,
                      float gamma_slope = 4.5f,
                      int output_color_space = 1);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    
    // Shader management - 新しい遅延ロード方式
    bool load_shaders();
    std::string load_shader_file(const std::string& filename);
    std::string load_all_shader_sources();
    bool compile_individual_shaders();
    bool create_compute_pipelines();
    
    // 遅延ロード + メモリキャッシュ方式
#ifdef __OBJC__
    id<MTLComputePipelineState> get_pipeline(const std::string& shader_name);  // 必要時にパイプライン取得
    id<MTLLibrary> compile_and_cache_shader(const std::string& shader_name);   // 個別シェーダーコンパイル（メモリキャッシュのみ）
#endif
    
    // XTrans 3-pass helper functions
    void calculate_allhex_and_sg(const char (&xtrans)[6][6], short allhex_out[2][3][3][8], uint16_t* sgrow_out, uint16_t* sgcol_out);
};

} // namespace libraw_enhanced