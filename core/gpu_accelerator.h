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
    
    // Apple Silicon unified memory optimization
    void* map_unified_memory(size_t size);
    void unmap_unified_memory(void* ptr, size_t size);
    
    // Bayer demosaicing methods
    bool demosaic_bayer_linear(const ImageBuffer& raw_buffer,
                              ImageBuffer& rgb_buffer, 
                              const ProcessingParams& params);
                              
    bool demosaic_bayer_vng(const ImageBuffer& raw_buffer,
                           ImageBuffer& rgb_buffer,
                           const ProcessingParams& params);
                           
    bool demosaic_bayer_ahd(const ImageBuffer& raw_buffer,
                           ImageBuffer& rgb_buffer,
                           const ProcessingParams& params);
                           
    bool demosaic_bayer_dcb(const ImageBuffer& raw_buffer,
                           ImageBuffer& rgb_buffer,
                           const ProcessingParams& params);
                           
    bool demosaic_bayer_amaze(const ImageBuffer& raw_buffer,
                             ImageBuffer& rgb_buffer,
                             const ProcessingParams& params);
    
    // X-Trans demosaicing methods  
    bool demosaic_xtrans_3pass(const ImageBuffer& raw_buffer,
                              ImageBuffer& rgb_buffer,
                              const ProcessingParams& params);
                              
    bool demosaic_xtrans_1pass(const ImageBuffer& raw_buffer,
                              ImageBuffer& rgb_buffer,
                              const ProcessingParams& params);
    
    // LibRaw separated pipeline methods
    
    // Step 1: White balance (scale_colors) - Applied to RAW data
    bool apply_white_balance_raw_bayer(const ImageBuffer& raw_input,
                                     ImageBuffer& raw_output,
                                     const float scale_mul[4],
                                     const float pre_mul[4],
                                     uint32_t filters,
                                     float bright = 1.0f);
                                     
    bool apply_white_balance_raw_xtrans(const ImageBuffer& raw_input,
                                      ImageBuffer& raw_output,
                                      const float scale_mul[4],
                                      const float pre_mul[4],
                                      float bright = 1.0f);
    
    // Step 3: Color space conversion (convert_to_rgb) - Applied to RGB data
    bool libraw_convert_to_rgb(const ImageBuffer& rgb_input,
                             ImageBuffer& rgb_output,
                             const float out_cam[3][4],
                             int output_color,
                             bool raw_color = false,
                             float gamma_power = 2.2f,
                             float gamma_slope = 4.5f,
                             bool apply_gamma = false);
                             
    // Alternative: Combined conversion with matrix calculation
    bool libraw_color_convert_with_matrix_selection(const ImageBuffer& rgb_input,
                                                  ImageBuffer& rgb_output,
                                                  const float rgb_cam[3][4],
                                                  int output_color,
                                                  bool raw_color = false,
                                                  float gamma_power = 2.2f,
                                                  float gamma_slope = 4.5f,
                                                  bool apply_gamma = false);

    // Modular processing kernels (can be combined in custom pipelines)
    
    // Matrix transformations
    bool apply_3x3_matrix_transform(const ImageBuffer& input,
                                   ImageBuffer& output,
                                   const float matrix[3][3],
                                   bool normalize_input = true,
                                   bool denormalize_output = true);
                                   
    bool apply_3x4_matrix_transform(const ImageBuffer& input,
                                   ImageBuffer& output,
                                   const float matrix[3][4],
                                   bool normalize_input = true,
                                   bool denormalize_output = true);
    
    // Gamma correction
    bool apply_gamma_correction_encode(const ImageBuffer& input,
                                     ImageBuffer& output,
                                     float gamma_power = 2.2f,
                                     float gamma_slope = 4.5f,
                                     bool srgb_mode = false);
                                     
    bool apply_gamma_correction_decode(const ImageBuffer& input,
                                     ImageBuffer& output,
                                     float gamma_power = 2.2f,
                                     float gamma_slope = 4.5f,
                                     bool srgb_mode = false);
                                     
    bool apply_gamma_correction_encode_inplace(ImageBuffer& image,
                                             float gamma_power = 2.2f,
                                             float gamma_slope = 4.5f,
                                             bool srgb_mode = false);
    
    // Color space utilities
    bool compute_libraw_out_cam_matrix(const float rgb_cam[3][4],
                                     float out_cam[3][4],
                                     int output_color_space);
                                     
    bool apply_selected_color_space_matrix(const ImageBuffer& input,
                                          ImageBuffer& output,
                                          int output_color_space);
    
    // Pipeline orchestration
    bool libraw_complete_rgb_pipeline(const ImageBuffer& rgb_input,
                                    ImageBuffer& rgb_output,
                                    const float rgb_cam[3][4],
                                    int output_color,
                                    bool raw_color = false,
                                    float gamma_power = 2.2f,
                                    float gamma_slope = 4.5f,
                                    bool apply_gamma = false,
                                    bool srgb_mode = false);
                                    
    bool libraw_color_conversion_step_only(const ImageBuffer& rgb_input,
                                          ImageBuffer& rgb_output,
                                          const float rgb_cam[3][4],
                                          int output_color,
                                          bool raw_color = false);
                                          
    bool libraw_gamma_correction_step_only(const ImageBuffer& rgb_input,
                                          ImageBuffer& rgb_output,
                                          float gamma_power = 2.2f,
                                          float gamma_slope = 4.5f,
                                          bool srgb_mode = false);

    // Legacy method (for backward compatibility)
    bool convert_color_space(const ImageBuffer& rgb_input,
                           ImageBuffer& rgb_output,
                           int output_color_space,
                           const float rgb_cam[3][4],
                           float gamma_power = 2.2f,
                           float gamma_slope = 4.5f,
                           bool apply_gamma = false);
                           
    bool apply_white_balance_and_color_conversion(const ImageBuffer& rgb_input,
                                                ImageBuffer& rgb_output,
                                                const float wb_multipliers[4],
                                                int output_color_space,
                                                float gamma_power = 2.2f,
                                                float gamma_slope = 4.5f,
                                                bool apply_gamma = false);
    
    // White balance on RAW data
    bool apply_white_balance_to_raw(const ImageBuffer& raw_buffer,
                                   const float wb_multipliers[4],
                                   unsigned int filters);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    
    // Shader management
    bool load_shaders();
    std::string load_shader_file(const std::string& filename);
    std::string load_all_shader_sources();
    bool compile_individual_shaders();
    bool create_compute_pipelines();
    
#ifdef __OBJC__
    // Metal execution helpers
    bool execute_bayer_kernel(id<MTLComputePipelineState> pipeline,
                             const ImageBuffer& raw_buffer,
                             ImageBuffer& rgb_buffer,
                             const ProcessingParams& params);
                             
    bool execute_xtrans_kernel(id<MTLComputePipelineState> pipeline,
                              const ImageBuffer& raw_buffer,
                              ImageBuffer& rgb_buffer,
                              const ProcessingParams& params);
#endif
};

} // namespace libraw_enhanced