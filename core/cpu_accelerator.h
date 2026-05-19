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
    std::string get_device_info() const;

    // Pre-processing methods
    bool pre_interpolate(ImageBufferFloat& rgb_buffer, uint32_t filters, 
                        const char (&xtrans)[6][6], bool half_size = false);
                         
    // Bayer specific acceleration methods
    bool demosaic_bayer_linear(const ImageBufferFloat& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            float maximum_value);

    bool demosaic_bayer_aahd(const ImageBufferFloat& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            float maximum_value);
                                       
    bool demosaic_bayer_dcb(const ImageBufferFloat& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            float maximum_value,
                            int iterations = 1,
                            bool dcb_enhance = true);
                         
    bool demosaic_bayer_amaze(const ImageBufferFloat& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            const float (&cam_mul)[4],
                            float maximum_value);
                           
    // X-Trans specific acceleration methods
    bool demosaic_xtrans_3pass(const ImageBufferFloat& raw_buffer,
                                ImageBufferFloat& rgb_buffer,
                                const char (&xtrans)[6][6],
                                const float (&color_matrix)[3][4],
                                float maximum_value);
                           
    bool demosaic_xtrans_1pass(const ImageBufferFloat& raw_buffer,
                                ImageBufferFloat& rgb_buffer,
                                const char (&xtrans)[6][6],
                                const float (&color_matrix)[3][4],
                                float maximum_value);
                            
    // White balance methods
    bool apply_white_balance(const ImageBuffer& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            const float wb_multipliers[4],
                            uint32_t filters,
                            const char xtrans[6][6]);

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

    // Tone Mapping method
    bool tone_mapping(const ImageBufferFloat& rgb_input,
                            ImageBufferFloat& rgb_output,
                            float after_scale);

    // Enhance Micro Contrast method.  `mask` (optional) is a width*height
    // float plane in [0,1] selecting where to apply enhancement.
    bool enhance_micro_contrast(const ImageBufferFloat& rgb_input,
                        ImageBufferFloat& rgb_output,
                        float threshold,
                        float strength,
                        float target_contrast,
                        const float* mask = nullptr);

    // Defringe: linear RGB, guide-green chroma-ratio suppression.
    // Designed to run before output color-space conversion and gamma.
    bool defringe(const ImageBufferFloat& rgb_input,
                  ImageBufferFloat& rgb_output,
                  float radius              = 10.0f,  // Gaussian blur radius (px)
                  float edge_threshold      = 0.1f,   // Normalized Sobel threshold [0,1]
                  float chroma_threshold    = 0.15f,  // Relative chroma excess threshold
                  float strength            = 10.0f,  // Purple-branch correction strength + detection sensitivity
                  bool  defringe_green      = false,  // Also correct green fringes
                  float green_strength_scale = 0.3f); // Green-branch amount multiplier [0,1]

    // Lateral chromatic aberration registration (post-demosaic dense RGB).
    //
    // Estimates per-cell sub-pixel shifts of R and B channels relative to G
    // using pyramidal Lucas-Kanade optical flow on the green channel as a
    // reference structure.  Each cell produces (dx, dy) shifts which are
    // smoothed across the image and applied via per-pixel bilinear
    // resampling.  Designed to run AFTER demosaic (dense R/G/B) and BEFORE
    // defringe / color-matrix / gamma, so chroma cleanup sees properly
    // registered channels.
    //
    //   cell_size      : LK estimation cell footprint in pixels at full
    //                    resolution (default 96).  Smaller → more spatial
    //                    detail but noisier estimates.
    //   max_iterations : LK iteration count per pyramid level (default 3).
    //   max_shift      : Hard cap on per-axis shift magnitude in pixels at
    //                    the finest level (clamp + safety; default 6.0).
    //   min_confidence : Cells whose gradient structure tensor minimum
    //                    eigenvalue is below this fraction of the image
    //                    maximum are treated as untrustworthy and fall back
    //                    to zero shift before smoothing (default 0.02).
    //   pyramid_levels : Number of resolution levels used by the pyramidal
    //                    LK.  N=1 reduces to a single-level estimate.  N=3
    //                    (default) handles shifts up to ~24 px at the
    //                    original resolution.
    //
    // Returns false on input failure.  In-place safe (output may alias input).
    bool ca_register_lateral(const ImageBufferFloat& rgb_input,
                             ImageBufferFloat& rgb_output,
                             int   cell_size      = 96,
                             int   max_iterations = 3,
                             float max_shift      = 6.0f,
                             float min_confidence = 0.02f,
                             int   pyramid_levels = 3);

    // Cross-channel guided filter for axial chromatic aberration removal.
    //
    // Axial (longitudinal) CA shows up as colored halos around bright/dark
    // points or thin edges where one chroma channel is defocused relative
    // to the green channel.  A geometric shift cannot fix this — instead we
    // apply the guided-image filter of He et al. (2010) to R (and B) using
    // G as the structural guide.  Within a local window of radius r the
    // filter constrains the output to a linear model
    //
    //     R_out(p)  ≈  a · G(p) + b           (per-window a, b)
    //
    // which retains R's correct local hue (because a and b can adapt to it)
    // while forcing R's high-frequency structure to follow G.  This is
    // exactly the behaviour needed to tighten defocused chroma halos.
    //
    // The `strength` parameter blends the filtered output with the input
    // (0 = no change, 1 = fully filtered).  The `radius` controls how far
    // halos can extend — larger → catches wider halos but smooths chroma
    // more aggressively.
    //
    // Designed to run AFTER lateral CA registration (so we work on
    // already-aligned channels) and BEFORE defringe.  In-place safe.
    bool ca_axial_cleanup(const ImageBufferFloat& rgb_input,
                          ImageBufferFloat& rgb_output,
                          int   radius   = 6,
                          float epsilon  = 1e-4f,
                          float strength = 0.3f);

private: 
    bool initialized_ = false;
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
    float apply_rec2020_gamma_encode(float v) const;    
    float apply_pure_power_gamma_encode(float linear_value, float power) const;
    float apply_pure_power_gamma_encode_with_slope(float v, float p, float s) const;
    
    void border_interpolate(const ImageBufferFloat& raw_buffer, uint32_t filters, size_t border);
};

} // namespace libraw_enhanced
