
#include "libraw_wrapper.h"
#include "constants.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <cstring>

// LibRaw ヘッダー
#include <libraw/libraw.h>

#ifdef __arm64__
#include "accelerator.h"
#include "camera_matrices.h"
#ifdef __OBJC__
#import <Metal/Metal.h>
#endif
#endif

namespace libraw_enhanced {

void convert_float_to_float(const ImageBufferFloat& rgb_buffer, float* dst) {
    size_t count = rgb_buffer.width * rgb_buffer.height * rgb_buffer.channels;
    
    float* src = reinterpret_cast<float*>(rgb_buffer.image);
    for (size_t i = 0; i < count; i++) {
        float val = *src++;
        val = std::max(0.0f, std::min(1.f, val));
        *dst++ = val;
    }
}

void convert_float_to_uint16(const ImageBufferFloat& rgb_buffer, uint16_t* dst) {
    size_t count = rgb_buffer.width * rgb_buffer.height * rgb_buffer.channels;
    const float scale = 65535.f;
    
    float* src = reinterpret_cast<float*>(rgb_buffer.image);
    for (size_t i = 0; i < count; i++) {
        float val = *src++ * scale;
        val = std::max(0.0f, std::min(scale, val));
        *dst++ = static_cast<uint8_t>(val + 0.5f);
    }
}

void convert_float_to_uint8(const ImageBufferFloat& rgb_buffer, uint8_t* dst) {
    size_t count = rgb_buffer.width * rgb_buffer.height * rgb_buffer.channels;
    const float scale = 255.f;
    
    float* src = reinterpret_cast<float*>(rgb_buffer.image);
    for (size_t i = 0; i < count; i++) {
        float val = *src++ * scale;
        val = std::max(0.0f, std::min(scale, val));
        *dst++ = static_cast<uint8_t>(val + 0.5f);
    }
}

class LibRawWrapper::Impl {
public:
    LibRaw processor;
    bool debug_mode = false;
    ProcessingTimes timing_info;  // 処理時間情報
    
#ifdef __arm64__
    std::unique_ptr<Accelerator> accelerator;

    ProcessingParams current_params;
#endif

    //===============================================================
    // 高精度タイマーユーティリティ
    //===============================================================
    std::chrono::high_resolution_clock::time_point timer_start;
    
    void start_timer() {
        timer_start = std::chrono::high_resolution_clock::now();
    }
    
    double get_elapsed_time() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - timer_start);
        return duration.count() / 1000000.0;  // 秒単位で返す
    }
    
    //===============================================================
    // LibRaw-compatible black level correction (internal function)
    //===============================================================
    void apply_black_level_correction(ImageBuffer& raw_buffer) {
        std::cout << "📋 Apply black level subtraction: " << processor.imgdata.color.black << std::endl;
        
        // Extract black level data from LibRaw
        int cblack[4];
        for (int i = 0; i < 4; i++) {
            cblack[i] = processor.imgdata.color.cblack[i];
        }
        
        size_t total_pixels = raw_buffer.width * raw_buffer.height;
        
        // Check if we have positional black level data (cblack[4] and cblack[5])
        if (processor.imgdata.color.cblack[4] && processor.imgdata.color.cblack[5]) {
            // Complex black level with position-dependent correction
            std::cout << "📋 Using position-dependent black level correction" << std::endl;
            
            for (size_t q = 0; q < total_pixels; q++) {
                for (int c = 0; c < 4; c++) {
                    int val = raw_buffer.image[q][c];
                    
                    // Position-dependent black level correction (LibRaw formula)
                    int row = q / raw_buffer.width;
                    int col = q % raw_buffer.width;
                    int pos_black = processor.imgdata.color.cblack[6 + 
                        (row % processor.imgdata.color.cblack[4]) * processor.imgdata.color.cblack[5] +
                        (col % processor.imgdata.color.cblack[5])];
                    
                    val -= pos_black;
                    val -= cblack[c];
                    
                    // Clamp to valid range
                    if (val < 0) val = 0;
                    if (val > (int)processor.imgdata.color.maximum) val = processor.imgdata.color.maximum;
                    
                    raw_buffer.image[q][c] = val;
                }
            }

        } else if (cblack[0] || cblack[1] || cblack[2] || cblack[3]) {
            // Simple per-channel black level correction
            std::cout << "📋 Using per-channel black level correction: " 
                    << cblack[0] << "," << cblack[1] << "," << cblack[2] << "," << cblack[3] << std::endl;
            
            for (size_t q = 0; q < total_pixels; q++) {
                for (int c = 0; c < 4; c++) {
                    int val = raw_buffer.image[q][c];
                    val -= cblack[c];
                    
                    // Clamp to valid range
                    if (val < 0) val = 0;
                    if (val > (int)processor.imgdata.color.maximum) val = processor.imgdata.color.maximum;
                    
                    raw_buffer.image[q][c] = val;
                }
            }

        } else {
            // Fallback: simple global black level
            std::cout << "📋 Using global black level: " << processor.imgdata.color.black << std::endl;
            
            for (size_t q = 0; q < total_pixels; q++) {
                for (int c = 0; c < 4; c++) {
                    int val = raw_buffer.image[q][c];
                    val -= processor.imgdata.color.black;
                    
                    // Clamp to valid range
                    if (val < 0) val = 0;
                    if (val > (int)processor.imgdata.color.maximum) val = processor.imgdata.color.maximum;
                    
                    raw_buffer.image[q][c] = val;
                }
            }
        }
        
        std::cout << "✅ Black level subtraction completed for " << total_pixels << " pixels" << std::endl;
    }

    //===============================================================
    // LibRaw-compatible green matching (internal function)
    //===============================================================
    void apply_green_matching(ImageBuffer& raw_buffer, uint32_t filters) {
        std::cout << "📋 Apply green matching for G1/G2 equilibration" << std::endl;
        
        // Skip for XTrans sensors (only for Bayer)
        if (filters == FILTERS_XTRANS) {
            std::cout << "📋 Skipping green matching for XTrans sensor" << std::endl;
            return;
        }
        
        const int margin = 3;
        const float thr = 0.01f;
        const int width = raw_buffer.width;
        const int height = raw_buffer.height;
        const int maximum = processor.imgdata.color.maximum; // Dynamic maximum value
        
        // Find G2 pixel position in Bayer pattern
        int oj = 2, oi = 2;
        
        // In RGGB Bayer pattern: R(0,0), G1(0,1), G2(1,0), B(1,1)
        // We look for G2 positions which are typically at (1,0) pattern
        // For more robust detection, we assume standard RGGB and start from (1,0)
        oj = 1; // G2 row offset
        oi = 0; // G2 col offset
        
        // Create working copy of image data
        uint16_t (*img)[4] = (uint16_t(*)[4])calloc(height * width, sizeof(uint16_t[4]));
        if (!img) {
            std::cerr << "❌ Failed to allocate memory for green matching" << std::endl;
            return;
        }
        
        // Copy original data
        memcpy(img, raw_buffer.image, height * width * sizeof(uint16_t[4]));
        
        int processed_pixels = 0;
        
        for (int j = oj + 2; j < height - margin; j += 2) {
            for (int i = oi + 2; i < width - margin; i += 2) {
                // Ensure we don't go out of bounds
                if (j - 1 < 0 || j + 1 >= height || i - 1 < 0 || i + 1 >= width ||
                    j - 2 < 0 || j + 2 >= height || i - 2 < 0 || i + 2 >= width) {
                    continue;
                }
                
                // Get surrounding G1 pixels (channel[1])
                int o1_1 = img[(j - 1) * width + i - 1][1];
                int o1_2 = img[(j - 1) * width + i + 1][1];
                int o1_3 = img[(j + 1) * width + i - 1][1];
                int o1_4 = img[(j + 1) * width + i + 1][1];
                
                // Get surrounding G2 pixels (channel[3])
                int o2_1 = img[(j - 2) * width + i][3];
                int o2_2 = img[(j + 2) * width + i][3];
                int o2_3 = img[j * width + i - 2][3];
                int o2_4 = img[j * width + i + 2][3];
                
                // Calculate averages
                double m1 = (o1_1 + o1_2 + o1_3 + o1_4) / 4.0;
                double m2 = (o2_1 + o2_2 + o2_3 + o2_4) / 4.0;
                
                // Calculate consistency (variation) in each group
                double c1 = (abs(o1_1 - o1_2) + abs(o1_1 - o1_3) + abs(o1_1 - o1_4) +
                            abs(o1_2 - o1_3) + abs(o1_3 - o1_4) + abs(o1_2 - o1_4)) / 6.0;
                double c2 = (abs(o2_1 - o2_2) + abs(o2_1 - o2_3) + abs(o2_1 - o2_4) +
                            abs(o2_2 - o2_3) + abs(o2_3 - o2_4) + abs(o2_2 - o2_4)) / 6.0;
                
                // Apply correction only in flat areas and non-saturated pixels
                if ((img[j * width + i][3] < maximum * 0.95) && 
                    (c1 < maximum * thr) && 
                    (c2 < maximum * thr) && 
                    (m2 > 0.1)) { // Avoid division by zero
                    
                    float correction = raw_buffer.image[j * width + i][3] * m1 / m2;
                    raw_buffer.image[j * width + i][3] = correction > maximum ? maximum : (uint16_t)correction;
                    processed_pixels++;
                }
            }
        }
        
        free(img);
        std::cout << "✅ Green matching completed: processed " << processed_pixels << " G2 pixels" << std::endl;
    }

    //===============================================================
    // CFA-aware white balance (experimental pre-demosaic implementation)
    //===============================================================
    // Main CFA-aware white balance function
    void apply_white_balance_to_cfa_data(
        ImageBuffer& raw_buffer, 
        const float wb_multipliers[4], 
        uint32_t filters,
        const char xtrans[6][6],
        float sensor_maximum
    ) {
        std::cout << "🧪 EXPERIMENTAL: Applying CFA-aware white balance (pre-demosaic)" << std::endl;
        std::cout << "   - Sensor Maximum: " << sensor_maximum << std::endl;
        std::cout << "   - WB Multipliers: [" << wb_multipliers[0] << ", " << wb_multipliers[1] 
                  << ", " << wb_multipliers[2] << ", " << wb_multipliers[3] << "]" << std::endl;
        
        if (filters == FILTERS_XTRANS) {
            apply_wb_xtrans(raw_buffer, wb_multipliers, xtrans, sensor_maximum);
        } else {
            apply_wb_bayer(raw_buffer, wb_multipliers, filters, sensor_maximum);
        }
        
        std::cout << "✅ CFA-aware white balance completed" << std::endl;
    }
    
    // Bayer CFA white balance implementation
    void apply_wb_bayer(
        ImageBuffer& raw_buffer,
        const float wb_multipliers[4],
        uint32_t filters,
        float sensor_maximum
    ) {
        std::cout << "🔧 Applying Bayer CFA white balance..." << std::endl;
        
        const size_t total_pixels = raw_buffer.width * raw_buffer.height;
        size_t processed_pixels = 0;
        
        // Process each pixel in the raw buffer
        for (size_t pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++) {
            int row = pixel_idx / raw_buffer.width;
            int col = pixel_idx % raw_buffer.width;
            
            // Get color channel for this pixel position using LibRaw's fcol logic
            int color_channel = (filters >> ((((row) << 1 & 14) | ((col) & 1)) << 1)) & 3;
            
            // Apply white balance multiplier to the native color channel
            uint16_t original_value = raw_buffer.image[pixel_idx][color_channel];
            if (original_value > 0) {  // Skip black pixels
                float adjusted_value = original_value * wb_multipliers[color_channel];
                
                // Clamp to sensor maximum
                if (adjusted_value > sensor_maximum) {
                    adjusted_value = sensor_maximum;
                }
                
                raw_buffer.image[pixel_idx][color_channel] = static_cast<uint16_t>(adjusted_value);
                
                // Special handling for G2 channel (channel 3) - copy to G1 if it's G2 pixel
                if (color_channel == 3) {
                    // This is a G2 pixel, also update the G1 channel for averaging
                    float g1_adjusted = original_value * wb_multipliers[1];  // Use G1 multiplier
                    if (g1_adjusted > sensor_maximum) {
                        g1_adjusted = sensor_maximum;
                    }
                    raw_buffer.image[pixel_idx][1] = static_cast<uint16_t>(g1_adjusted);
                }
                
                processed_pixels++;
            }
        }
        
        std::cout << "✅ Bayer WB processed " << processed_pixels << " pixels" << std::endl;
    }
    
    // X-Trans CFA white balance implementation
    void apply_wb_xtrans(
        ImageBuffer& raw_buffer,
        const float wb_multipliers[4],
        const char xtrans[6][6],
        float sensor_maximum
    ) {
        std::cout << "🔧 Applying X-Trans CFA white balance..." << std::endl;
        
        const size_t total_pixels = raw_buffer.width * raw_buffer.height;
        size_t processed_pixels = 0;
        
        // Process each pixel in the raw buffer
        for (size_t pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++) {
            int row = pixel_idx / raw_buffer.width;
            int col = pixel_idx % raw_buffer.width;
            
            // Get color channel using X-Trans pattern
            int color_channel = xtrans[row % 6][col % 6];
            
            // Apply white balance multiplier to the native color channel
            uint16_t original_value = raw_buffer.image[pixel_idx][color_channel];
            if (original_value > 0) {  // Skip black pixels
                float adjusted_value = original_value * wb_multipliers[color_channel];
                
                // Clamp to sensor maximum
                if (adjusted_value > sensor_maximum) {
                    adjusted_value = sensor_maximum;
                }
                
                raw_buffer.image[pixel_idx][color_channel] = static_cast<uint16_t>(adjusted_value);
                processed_pixels++;
            }
        }
        
        std::cout << "✅ X-Trans WB processed " << processed_pixels << " pixels" << std::endl;
    }

    //===============================================================
    // LibRaw-compatible adjust_maximum implementation
    //===============================================================
    void adjust_maximum(const ImageBuffer& raw_buffer, float threshold) {
        std::cout << "📋 Apply adjust_maximum for dynamic maximum value adjustment (threshold: " << threshold << ")" << std::endl;
        
        // Early return if threshold is too small (LibRaw compatibility)
        if (threshold < 0.00001f) {
            std::cout << "📋 Skipping adjust_maximum: threshold too small (" << threshold << ")" << std::endl;
            return;
        }
        
        // Use default threshold if too large (LibRaw compatibility)
        float auto_threshold = threshold;
        if (threshold > 0.99999f) {
            auto_threshold = 0.75f; // LIBRAW_DEFAULT_ADJUST_MAXIMUM_THRESHOLD
            std::cout << "📋 Using default threshold: " << auto_threshold << std::endl;
        }
        
        // Calculate data_maximum if not already set (LibRaw compatibility)
        uint16_t real_max = processor.imgdata.color.data_maximum;
        if (real_max == 0 && raw_buffer.image != nullptr) {
            std::cout << "📋 Calculating data_maximum by scanning image data..." << std::endl;
            
            size_t total_pixels = raw_buffer.width * raw_buffer.height;
            uint16_t max_value = 0;
            
            for (size_t i = 0; i < total_pixels; i++) {
                for (int c = 0; c < 4; c++) {
                    uint16_t val = raw_buffer.image[i][c];
                    if (val > max_value) {
                        max_value = val;
                    }
                }
            }
            
            real_max = max_value;
            processor.imgdata.color.data_maximum = real_max;
            std::cout << "📋 Calculated data_maximum: " << real_max << std::endl;
        }
        
        uint16_t current_max = processor.imgdata.color.maximum;
        std::cout << "📋 Current maximum: " << current_max << ", data_maximum: " << real_max << std::endl;
        
        // Apply LibRaw's adjust_maximum logic
        if (real_max > 0 && real_max < current_max && 
            real_max > current_max * auto_threshold) {
            
            processor.imgdata.color.maximum = real_max;
            std::cout << "✅ Adjusted maximum value: " << current_max << " → " << real_max 
                      << " (threshold: " << auto_threshold << ")" << std::endl;
        } else {
            std::cout << "📋 No adjustment needed - conditions not met" << std::endl;
            std::cout << "   real_max > 0: " << (real_max > 0) << std::endl;
            std::cout << "   real_max < current_max: " << (real_max < current_max) << std::endl;
            std::cout << "   real_max > current_max * threshold: " << (real_max > current_max * auto_threshold) 
                      << " (" << real_max << " > " << (current_max * auto_threshold) << ")" << std::endl;
        }
    }

    //===============================================================
    // Main RAW to RGB processing pipeline
    //===============================================================
    bool process_raw_to_rgb(ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, const ProcessingParams& params) {
        std::cout << "🎯 Starting unified RAW→RGB processing pipeline" << std::endl;
        std::cout << "📋 Parameters: demosaic=" << params.demosaic_algorithm << std::endl;

        // Step 1: Initialize LibRaw and check for raw data
        if (!accelerator) {
            std::cerr << "❌ Accelerator not initialized" << std::endl;
            return false;
        }
        
        // Set GPU acceleration flag from processing parameters
        accelerator->set_use_gpu_acceleration(params.use_gpu_acceleration);

        // Apply LibRaw-compatible black level subtraction
        apply_black_level_correction(raw_buffer);

        // Apply adjust_maximum for dynamic maximum value adjustment (must be after black level correction)
        adjust_maximum(raw_buffer, params.adjust_maximum_thr);

        // set filters and xtrans
        uint32_t filters = processor.imgdata.idata.filters;        
        char (&xtrans)[6][6] = processor.imgdata.idata.xtrans;        
        std::cout << "🔍 Filters value: " << filters << " (FILTERS_XTRANS=" << FILTERS_XTRANS << ")" << std::endl;

        /*
        if (filters > 0xff) {
            for (uint32_t row = 0; row < raw_buffer.height; ++row) {
                for (uint32_t col = 0; col < raw_buffer.width; ++col) {
                    char c = fcol_bayer(row, col, filters);
                    if (c == 1) {
                        uint32_t idx = row * raw_buffer.width + col;
                        if (raw_buffer.image[idx][1] < 1000) {
                            raw_buffer.image[idx][1] = 1000;
                        } else
                        if (raw_buffer.image[idx][1] > processor.imgdata.color.maximum-1000) {
                            //raw_buffer.image[idx][1] = (raw_buffer.image[idx][0] + raw_buffer.image[idx][2]) / 2;
                            raw_buffer.image[idx][1] = processor.imgdata.color.maximum-1000;
                        }
                    }
                }
            }
        }
        */

        // Apply green matching for Bayer sensors (after black level, before demosaic)
        apply_green_matching(raw_buffer, filters);

        // EXPERIMENTAL: Apply CFA-aware white balance before demosaic
        std::cout << "🧪 EXPERIMENTAL: Applying pre-demosaic white balance..." << std::endl;
        
        // Calculate white balance multipliers (same logic as original)
        float effective_wb[4];
        if (params.use_camera_wb && processor.imgdata.color.cam_mul[1] > 0) {
            float dmin = *std::min_element(std::begin(processor.imgdata.color.cam_mul), std::end(processor.imgdata.color.cam_mul) - 1);
            effective_wb[0] = processor.imgdata.color.cam_mul[0] / dmin;
            effective_wb[1] = processor.imgdata.color.cam_mul[1] / dmin;
            effective_wb[2] = processor.imgdata.color.cam_mul[2] / dmin;
            effective_wb[3] = processor.imgdata.color.cam_mul[3] / dmin;
            std::cout << "📷 Using camera WB from EXIF (max-normalized cam_mul):" << std::endl;
        } else if (params.use_auto_wb && processor.imgdata.color.pre_mul[1] > 0) {
            float dmin = *std::min_element(std::begin(processor.imgdata.color.pre_mul), std::end(processor.imgdata.color.pre_mul) - 1);
            effective_wb[0] = processor.imgdata.color.pre_mul[0] / dmin;
            effective_wb[1] = processor.imgdata.color.pre_mul[1] / dmin;
            effective_wb[2] = processor.imgdata.color.pre_mul[2] / dmin;
            effective_wb[3] = processor.imgdata.color.pre_mul[3] / dmin;
            std::cout << "📷 Using computed WB from LibRaw (max-normalized pre_mul):" << std::endl;
        } else {
            // Use user-specified white balance or default
            effective_wb[0] = params.user_wb[0];
            effective_wb[1] = params.user_wb[1];
            effective_wb[2] = params.user_wb[2];
            effective_wb[3] = params.user_wb[3];
            std::cout << "👤 Using user/default WB:" << std::endl;
        }
        
        // Determine CFA type and apply appropriate WB processing
        float sensor_max = static_cast<float>(processor.imgdata.color.maximum);
        
        apply_white_balance_to_cfa_data(
            raw_buffer, 
            effective_wb, 
            filters, 
            xtrans, 
            sensor_max
        );

        // Step 2: Apply pre-interpolation processing
        std::cout << "🔧 Applying pre-interpolation processing..." << std::endl;
        bool pre_success = accelerator->pre_interpolate(raw_buffer, filters, xtrans, params.half_size);
        if (!pre_success) {
            std::cerr << "❌ Pre-interpolation processing failed" << std::endl;
            return false;
        }
        std::cout << "✅ Pre-interpolation completed successfully" << std::endl;

        // Step 3: Camera matrix-based color space conversion (reuse float_rgb buffer)
        const char* camera_make = processor.imgdata.idata.make;
        const char* camera_model = processor.imgdata.idata.model;        
        std::cout << "📷 Camera: " << camera_make << " " << camera_model << std::endl;
        
        // Get camera-specific color transformation matrix
        ColorTransformMatrix camera_matrix = compute_camera_transform(camera_make, camera_model, (int)ColorSpace::XYZ);
        if(!camera_matrix.valid) {
            std::cout << "⚠️ Camera not in database, using fallback matrix" << std::endl;
            // Use fallback identity-like matrix for unknown cameras
            camera_matrix.set_default();
        }

        // Step 5: Demosaic processing (unified CPU/GPU selection via accelerator)
        // Phase 5: Pass LibRaw cam_mul for dynamic initialGain calculation and maximum_value for precise normalization
        bool demosaic_success = accelerator->demosaic_compute(raw_buffer, rgb_buffer, params.demosaic_algorithm, filters, xtrans, camera_matrix.transform, processor.imgdata.color.cam_mul, processor.imgdata.color.maximum);
        if (!demosaic_success) {
            std::cerr << "❌ Demosaic processing failed" << std::endl;
            return false;
        }


        // EXPERIMENTAL: White balance moved to pre-demosaic (COMMENTED OUT)
        // Step 6: Apply white balance to RAW data (float32 interface)        
/*        
        float effective_wb[4];
        if (params.use_camera_wb && processor.imgdata.color.cam_mul[1] > 0) {
            float dmin = *std::min_element(std::begin(processor.imgdata.color.cam_mul), std::end(processor.imgdata.color.cam_mul) - 1);
            effective_wb[0] = processor.imgdata.color.cam_mul[0] / dmin;
            effective_wb[1] = processor.imgdata.color.cam_mul[1] / dmin;
            effective_wb[2] = processor.imgdata.color.cam_mul[2] / dmin;
            effective_wb[3] = processor.imgdata.color.cam_mul[3] / dmin;
            std::cout << "📷 Using camera WB from EXIF (max-normalized cam_mul):" << std::endl;

        } else if (params.use_auto_wb && processor.imgdata.color.pre_mul[1] > 0) {
            float dmin = *std::min_element(std::begin(processor.imgdata.color.pre_mul), std::end(processor.imgdata.color.pre_mul) - 1);
            effective_wb[0] = processor.imgdata.color.pre_mul[0] / dmin;
            effective_wb[1] = processor.imgdata.color.pre_mul[1] / dmin;
            effective_wb[2] = processor.imgdata.color.pre_mul[2] / dmin;
            effective_wb[3] = processor.imgdata.color.pre_mul[3] / dmin;
            std::cout << "📷 Using computed WB from LibRaw (max-normalized pre_mul):" << std::endl;

        } else {
            // Use user-specified white balance or default
            effective_wb[0] = params.user_wb[0];
            effective_wb[1] = params.user_wb[1];
            effective_wb[2] = params.user_wb[2];
            effective_wb[3] = params.user_wb[3];
            std::cout << "👤 Using user/default WB:" << std::endl;
        }
        std::cout << "[" << effective_wb[0] << ", " << effective_wb[1] 
                << ", " << effective_wb[2] << ", " << effective_wb[3] << "]" << std::endl;
        if (!accelerator->apply_white_balance(rgb_buffer, rgb_buffer, effective_wb)) {
            std::cerr << "❌ White balance failed" << std::endl;
            return false;
        }
*/        

        // Get camera-specific color transformation matrix
        camera_matrix = compute_camera_transform(camera_make, camera_model, params.output_color_space);        
        if (!camera_matrix.valid) {
            std::cout << "⚠️ Camera not in database, using fallback matrix" << std::endl;
            camera_matrix.set_default();
        }
        if (!accelerator->convert_color_space(rgb_buffer, rgb_buffer, camera_matrix.transform)) {
            std::cerr << "❌ Camera matrix color conversion failed" << std::endl;
            return false;
        }
                
        // Step 6: In-place gamma correction with color space awareness (reuse float_rgb buffer)  
        if (!accelerator->gamma_correct(rgb_buffer, rgb_buffer, params.gamma_power, params.gamma_slope, params.output_color_space)) {
            std::cerr << "❌ Gamma correction failed" << std::endl;
            return false;
        }
        
        // Step 6.5: Highlight recovery (after gamma correction, before final output)
        if (params.highlight_mode > 2) {
            std::cout << "🔧 Applying highlight recovery (mode " << params.highlight_mode << ")..." << std::endl;
            if (!recover_highlights(rgb_buffer, params.highlight_mode)) {
                std::cerr << "❌ Highlight recovery failed" << std::endl;
                return false;
            }
            std::cout << "✅ Highlight recovery completed" << std::endl;
        }
        
        std::cout << "✅ Unified RAW→RGB processing pipeline completed successfully" << std::endl;
        return true;
    }

    //===============================================================
    // LibRaw raw2image_ex equivalent implementation (excluding subtract_black)
    //===============================================================
    int convert_raw_to_image() {
        std::cout << "🔧 Converting raw data to image..." << std::endl;
        
        // Step 1: raw2image_start equivalent - initialization
        raw2image_start();
        
        // Step 2: Handle existing processed image
        if (processor.imgdata.image) {
            std::cout << "ℹ️ Image data already exists, skipping conversion" << std::endl;
            return 0;
        }
        
        // Step 3: Check for raw data availability
        if (!processor.imgdata.rawdata.raw_image && 
            !processor.imgdata.rawdata.color4_image && 
            !processor.imgdata.rawdata.color3_image) {
            std::cerr << "❌ No raw data available for conversion" << std::endl;
            return LIBRAW_REQUEST_FOR_NONEXISTENT_IMAGE;
        }
        
        // Step 4: Calculate allocation dimensions
        int alloc_width = processor.imgdata.sizes.iwidth;
        int alloc_height = processor.imgdata.sizes.iheight;
        
        
        // Step 5: Allocate image buffer
        size_t alloc_sz = alloc_width * alloc_height;
        processor.imgdata.image = (unsigned short (*)[4])calloc(alloc_sz, sizeof(*processor.imgdata.image));
        
        if (!processor.imgdata.image) {
            std::cerr << "❌ Failed to allocate image buffer" << std::endl;
            return LIBRAW_UNSUFFICIENT_MEMORY;
        }
        
        std::cout << "✅ Allocated image buffer (" << alloc_sz << " pixels)" << std::endl;
        
        // Step 6: Copy data based on source type
        if (processor.imgdata.rawdata.color4_image) {
            std::cout << "🔧 Copying from color4_image..." << std::endl;
            copy_color4_image();
        } else if (processor.imgdata.rawdata.color3_image) {
            std::cout << "🔧 Copying from color3_image..." << std::endl;
            copy_color3_image();
        } else if (processor.imgdata.rawdata.raw_image) {
            std::cout << "🔧 Copying from raw_image (Bayer/X-Trans)..." << std::endl;
            copy_bayer_image();
        } else {
            std::cerr << "❌ Unsupported raw data format" << std::endl;
            return LIBRAW_UNSUPPORTED_THUMBNAIL;
        }
        
        std::cout << "✅ Raw to image conversion completed successfully" << std::endl;
        return 0;
    }
    
    // raw2image_start equivalent - setup and initialization
    void raw2image_start() {
        std::cout << "🔧 raw2image_start: Initializing conversion parameters..." << std::endl;
        
        // Restore metadata from raw data structures
        if (processor.imgdata.rawdata.color.maximum > 0) {
            memcpy(&processor.imgdata.color, &processor.imgdata.rawdata.color, sizeof(processor.imgdata.color));
        }
        
        // Calculate image dimensions
        auto& S = processor.imgdata.sizes;
        auto& O = processor.imgdata.params;
        
        // Handle half-size processing
        bool shrink = !processor.imgdata.rawdata.color4_image && 
                     !processor.imgdata.rawdata.color3_image && 
                     processor.imgdata.idata.filters && O.half_size;
        
        // Calculate final image dimensions
        if (shrink) {
            S.iheight = (S.height + 1) >> 1;
            S.iwidth = (S.width + 1) >> 1;
            std::cout << "   - Half-size mode: " << S.iwidth << "x" << S.iheight << std::endl;
        } else {
            S.iheight = S.height;
            S.iwidth = S.width;
            std::cout << "   - Full-size mode: " << S.iwidth << "x" << S.iheight << std::endl;
        }
        
        std::cout << "✅ raw2image_start completed" << std::endl;
    }
    
    // Copy from 4-channel processed data
    void copy_color4_image() {
        auto& sizes = processor.imgdata.sizes;
        size_t total_pixels = sizes.iwidth * sizes.iheight;
        
        for (size_t i = 0; i < total_pixels; i++) {
            for (int c = 0; c < 4; c++) {
                processor.imgdata.image[i][c] = processor.imgdata.rawdata.color4_image[i][c];
            }
        }
        
        std::cout << "✅ Copied " << total_pixels << " pixels from color4_image" << std::endl;
    }
    
    // Copy from 3-channel processed data  
    void copy_color3_image() {
        auto& sizes = processor.imgdata.sizes;
        size_t total_pixels = sizes.iwidth * sizes.iheight;
        
        for (size_t i = 0; i < total_pixels; i++) {
            for (int c = 0; c < 3; c++) {
                processor.imgdata.image[i][c] = processor.imgdata.rawdata.color3_image[i][c];
            }
            processor.imgdata.image[i][3] = 0; // Alpha channel
        }
        
        std::cout << "✅ Copied " << total_pixels << " pixels from color3_image" << std::endl;
    }
    
    // Copy from raw Bayer/X-Trans data
    void copy_bayer_image() {
        auto& sizes = processor.imgdata.sizes;
        auto& params = processor.imgdata.params;
        auto& idata = processor.imgdata.idata;
        
        std::cout << "🔧 Processing Bayer/X-Trans pattern..." << std::endl;
        std::cout << "   - Raw size: " << sizes.raw_width << "x" << sizes.raw_height << std::endl;
        std::cout << "   - Output size: " << sizes.iwidth << "x" << sizes.iheight << std::endl;
        std::cout << "   - Filters: 0x" << std::hex << idata.filters << std::dec << std::endl;
        
        bool shrink = params.half_size && idata.filters;
        int shrink_factor = shrink ? 1 : 0;
        
        // Initialize all image data to zero
        size_t total_pixels = sizes.iwidth * sizes.iheight;
        memset(processor.imgdata.image, 0, total_pixels * sizeof(*processor.imgdata.image));
        
        // Handle Foveon X3 sensors (special case)
        if (idata.is_foveon) {
            copy_foveon_image();
            return;
        }
        
        // Handle special formats (Phase One, Leaf, etc.)
        if (copy_special_formats()) {
            return;
        }

        if (idata.filters == FILTERS_XTRANS) {
            copy_xtrans_image(shrink_factor);
            return;
        }
        
        copy_bayer_image(shrink_factor);
        
        std::cout << "✅ Copied Bayer data to image buffer" << std::endl;
    }
    
    // Handle Foveon X3 sensors 
    void copy_foveon_image() {
        auto& sizes = processor.imgdata.sizes;
        size_t total_pixels = sizes.iwidth * sizes.iheight;
        
        std::cout << "🔧 Processing Foveon X3 sensor..." << std::endl;
        
        // Foveon has 3 color values per pixel position
        for (size_t i = 0; i < total_pixels; i++) {
            // In Foveon, each layer corresponds to a color
            processor.imgdata.image[i][0] = processor.imgdata.rawdata.raw_image[i * 3 + 0]; // Red
            processor.imgdata.image[i][1] = processor.imgdata.rawdata.raw_image[i * 3 + 1]; // Green  
            processor.imgdata.image[i][2] = processor.imgdata.rawdata.raw_image[i * 3 + 2]; // Blue
            processor.imgdata.image[i][3] = 0; // Alpha
        }
        
        std::cout << "✅ Copied Foveon X3 data" << std::endl;
    }
    
    // Handle special camera formats (Phase One, Leaf, Kodak, etc.)
    bool copy_special_formats() {
        auto& idata = processor.imgdata.idata;
        
        // Phase One cameras
        if (strstr(idata.make, "Phase One") || strstr(idata.model, "Phase One")) {
            std::cout << "🔧 Processing Phase One format..." << std::endl;
            return copy_phase_one_image();
        }
        
        // Leaf cameras  
        if (strstr(idata.make, "Leaf") || strstr(idata.model, "Leaf")) {
            std::cout << "🔧 Processing Leaf format..." << std::endl;
            return copy_leaf_image();
        }
        
        // Kodak cameras
        if (strstr(idata.make, "KODAK") || strstr(idata.model, "KODAK")) {
            std::cout << "🔧 Processing Kodak format..." << std::endl;
            return copy_kodak_image();
        }
        
        // Hasselblad cameras
        if (strstr(idata.make, "Hasselblad") || strstr(idata.model, "Hasselblad")) {
            std::cout << "🔧 Processing Hasselblad format..." << std::endl;
            return copy_hasselblad_image();
        }
        
        return false; // No special format detected
    }
    
    // Phase One specific processing
    bool copy_phase_one_image() {
        auto& sizes = processor.imgdata.sizes;
        size_t total_pixels = sizes.iwidth * sizes.iheight;
        
        // Phase One uses specific channel ordering
        for (size_t i = 0; i < total_pixels; i++) {
            int row = i / sizes.iwidth;
            int col = i % sizes.iwidth;
            
            // Phase One color filter array pattern
            int color_channel = get_phase_one_color(row, col);
            unsigned short val = processor.imgdata.rawdata.raw_image[i];
            
            processor.imgdata.image[i][color_channel] = val;
        }
        
        std::cout << "✅ Processed Phase One format" << std::endl;
        return true;
    }
    
    // Leaf specific processing
    bool copy_leaf_image() {
        auto& sizes = processor.imgdata.sizes;
        size_t total_pixels = sizes.iwidth * sizes.iheight;
        
        for (size_t i = 0; i < total_pixels; i++) {
            int row = i / sizes.iwidth;
            int col = i % sizes.iwidth;
            
            int color_channel = get_leaf_color(row, col);
            unsigned short val = processor.imgdata.rawdata.raw_image[i];
            
            processor.imgdata.image[i][color_channel] = val;
        }
        
        std::cout << "✅ Processed Leaf format" << std::endl;
        return true;
    }
    
    // Kodak specific processing
    bool copy_kodak_image() {
        auto& sizes = processor.imgdata.sizes;
        size_t total_pixels = sizes.iwidth * sizes.iheight;
        
        for (size_t i = 0; i < total_pixels; i++) {
            int row = i / sizes.iwidth;
            int col = i % sizes.iwidth;
            
            int color_channel = get_kodak_color(row, col);
            unsigned short val = processor.imgdata.rawdata.raw_image[i];
            
            processor.imgdata.image[i][color_channel] = val;
        }
        
        std::cout << "✅ Processed Kodak format" << std::endl;
        return true;
    }
    
    // Hasselblad specific processing
    bool copy_hasselblad_image() {
        auto& sizes = processor.imgdata.sizes;
        size_t total_pixels = sizes.iwidth * sizes.iheight;
        
        for (size_t i = 0; i < total_pixels; i++) {
            int row = i / sizes.iwidth;
            int col = i % sizes.iwidth;
            
            int color_channel = get_hasselblad_color(row, col);
            unsigned short val = processor.imgdata.rawdata.raw_image[i];
            
            processor.imgdata.image[i][color_channel] = val;
        }
        
        std::cout << "✅ Processed Hasselblad format" << std::endl;
        return true;
    }

    bool copy_xtrans_image(int shrink_factor) {
        auto& sizes = processor.imgdata.sizes;

        // Standard X-Trans processing
        for (int row = 0; row < sizes.height && row < sizes.raw_height - sizes.top_margin; row++) {
            for (int col = 0; col < sizes.width && col < sizes.raw_width - sizes.left_margin; col++) {
                
                // Calculate source pixel position
                int src_row = row + sizes.top_margin;
                int src_col = col + sizes.left_margin;
                int src_idx = src_row * (sizes.raw_width) + src_col;
                
                // Calculate destination pixel position (with potential shrinking)
                int dst_row = row >> shrink_factor;
                int dst_col = col >> shrink_factor;
                int dst_idx = dst_row * sizes.iwidth + dst_col;
                
                // Skip if destination is out of bounds
                if (dst_row >= sizes.iheight || dst_col >= sizes.iwidth) continue;
                
                // Get raw pixel value
                unsigned short val = processor.imgdata.rawdata.raw_image[src_idx];
                
                // Determine color channel using filter pattern
                int color_channel = get_xtrans_color(row, col);
                
                // Store pixel value in appropriate channel
                processor.imgdata.image[dst_idx][color_channel] = val;
            }
        }
        return true;
    }

    bool copy_bayer_image(int shrink_factor) {
        auto& sizes = processor.imgdata.sizes;

        // Standard bayer processing
        for (int row = 0; row < sizes.height && row < sizes.raw_height - sizes.top_margin; row++) {
            for (int col = 0; col < sizes.width && col < sizes.raw_width - sizes.left_margin; col++) {
                
                // Calculate source pixel position
                int src_row = row + sizes.top_margin;
                int src_col = col + sizes.left_margin;
                int src_idx = src_row * (sizes.raw_width) + src_col;
                
                // Calculate destination pixel position (with potential shrinking)
                int dst_row = row >> shrink_factor;
                int dst_col = col >> shrink_factor;
                int dst_idx = dst_row * sizes.iwidth + dst_col;
                
                // Skip if destination is out of bounds
                if (dst_row >= sizes.iheight || dst_col >= sizes.iwidth) continue;
                
                // Get raw pixel value
                unsigned short val = processor.imgdata.rawdata.raw_image[src_idx];
                
                // Determine color channel using filter pattern
                int color_channel = get_bayer_color(row, col);
                
                // Store pixel value in appropriate channel
                processor.imgdata.image[dst_idx][color_channel] = val;
                if (color_channel == 3) {
                    // If G2 channel, copy to G1 for averaging
                    processor.imgdata.image[dst_idx][1] = val; // G1
                }
            }
        }
        return true;
    }

    // Get color channel for Bayer/X-Trans pattern
    inline int get_bayer_color(int row, int col) {
        auto& idata = processor.imgdata.idata;
        
        return (idata.filters >> ((((row) << 1 & 14) | ((col) & 1)) << 1)) & 3;
    }

    inline int get_xtrans_color(int row, int col) {
        auto& idata = processor.imgdata.idata;
        
        return idata.xtrans[row % 6][col % 6];
    }
    
    // Phase One color channel determination
    int get_phase_one_color(int row, int col) {
        // Phase One specific Bayer pattern variations
        // Usually RGGB but with specific modifications for Phase One sensors
        return ((row & 1) << 1) | (col & 1); // Simplified for now
    }
    
    // Leaf color channel determination  
    int get_leaf_color(int row, int col) {
        // Leaf cameras typically use standard Bayer patterns
        // RGGB pattern: R=0, G=1, B=2
        if ((row & 1) == 0) {
            return (col & 1) == 0 ? 0 : 1; // R or G
        } else {
            return (col & 1) == 0 ? 1 : 2; // G or B
        }
    }
    
    // Kodak color channel determination
    int get_kodak_color(int row, int col) {
        // Kodak often uses unique color filter arrays
        // Some use GRBG, others use RGGB variants
        if ((row & 1) == 0) {
            return (col & 1) == 0 ? 1 : 0; // G or R
        } else {
            return (col & 1) == 0 ? 2 : 1; // B or G
        }
    }
    
    // Hasselblad color channel determination
    int get_hasselblad_color(int row, int col) {
        // Hasselblad typically uses standard Bayer RGGB
        if ((row & 1) == 0) {
            return (col & 1) == 0 ? 0 : 1; // R or G
        } else {
            return (col & 1) == 0 ? 1 : 2; // G or B
        }
    }

    Impl() {
#ifdef __arm64__
        // Metal加速器初期化
        accelerator = std::make_unique<Accelerator>();
        accelerator->initialize();
#else
        std::cout << "Metal acceleration not compiled in" << std::endl;
#endif
    }
    
    int load_file(const std::string& filename) {
        start_timer();
        int result = processor.open_file(filename.c_str());
        timing_info.file_load_time = get_elapsed_time();
        
        if (debug_mode) {
            std::cout << "⏱️  File load time: " << timing_info.file_load_time << "s" << std::endl;
        }
        
        return result;
    }
    
    int unpack() {
        start_timer();
        int result = processor.unpack();
        timing_info.unpack_time = get_elapsed_time();
        
        if (debug_mode) {
            std::cout << "⏱️  Unpack time: " << timing_info.unpack_time << "s" << std::endl;
        }
        
        return result;
    }
    
    int process() {
        start_timer();
        
        // CRITICAL: Ensure default parameters are initialized if not already set
        if (current_params.user_wb[0] == 0.0f && current_params.user_wb[1] == 0.0f && 
            current_params.user_wb[2] == 0.0f && current_params.user_wb[3] == 0.0f) {
            std::cout << "⚠️  current_params appears uninitialized, setting defaults" << std::endl;
            ProcessingParams defaults;
            set_processing_params(defaults);
        }
        
#ifdef __arm64__
        // Use unified pipeline for accelerated processing
        if (accelerator && accelerator->is_available()) {
            std::cout << "🚀 Using unified accelerated pipeline (automatic GPU/CPU selection)" << std::endl;
            
            // Step 1: Convert raw sensor data to processed image (LibRaw raw2image_ex equivalent)
            std::cout << "🔧 Converting raw sensor data to processed image format..." << std::endl;
            int raw2image_result = convert_raw_to_image();
            if (raw2image_result != 0) {
                std::cerr << "❌ Raw to image conversion failed: " << raw2image_result << std::endl;
                return false;
            }
            
            // Prepare RAW data buffer (now properly initialized)
            ImageBuffer raw_buffer;
            raw_buffer.width = processor.imgdata.sizes.iwidth;   // Use iwidth (processed width)
            raw_buffer.height = processor.imgdata.sizes.iheight; // Use iheight (processed height)
            raw_buffer.channels = 4;
            raw_buffer.image = processor.imgdata.image;          // Now guaranteed non-null
            
            // Prepare output RGB buffer
            rgb_buffer.width = processor.imgdata.sizes.iwidth;   // Use processed width, not raw width
            rgb_buffer.height = processor.imgdata.sizes.iheight; // Use processed height, not raw height
            rgb_buffer.channels = 3;
            
            // Allocate output buffer - CRITICAL: allocate for float32, then convert to uint8
            size_t float_elements = rgb_buffer.width * rgb_buffer.height * 3;
            rgb_buffer_image.resize(float_elements); // Resize vector to hold float data
            rgb_buffer.image = reinterpret_cast<float (*)[3]>(rgb_buffer_image.data());
            
            // Use unified processing pipeline
            if (process_raw_to_rgb(raw_buffer, rgb_buffer, current_params)) {
                timing_info.total_time = get_elapsed_time();
                
                if (debug_mode) {
                    std::cout << "⏱️  Total unified pipeline processing time: " << timing_info.total_time << "s" << std::endl;
                }
                
                return LIBRAW_SUCCESS;
            } else {
                std::cout << "❌ Unified pipeline failed, NO FALLBACK (testing mode)" << std::endl;
                return LIBRAW_UNSPECIFIED_ERROR;  // フォールバック無効
            }
        }
#endif
        // FALLBACK DISABLED FOR TESTING
        std::cout << "❌ LibRaw dcraw_process fallback DISABLED for testing" << std::endl;
        timing_info.total_time = get_elapsed_time();
        return LIBRAW_UNSPECIFIED_ERROR;
        
        // Fall back to standard LibRaw CPU processing (DISABLED)
        // int result = processor.dcraw_process();
        // if (debug_mode) {
        //     std::cout << "⏱️  Total LibRaw dcraw_process time: " << timing_info.total_time << "s" << std::endl;
        // }
        // return result;
    }
    
#ifdef __arm64__
    // Store processing results
    std::vector<float> rgb_buffer_image;
    ImageBufferFloat rgb_buffer;
#endif
    
    ProcessedImageData get_processed_image() {
        ProcessedImageData result;
        
#ifdef __arm64__
        // Check if we have Metal-processed data
        std::cout << "🔧 LibRaw_Wrapper rgb_buffer: " << rgb_buffer.image << " width: " << rgb_buffer.width << " height: " << rgb_buffer.height << std::endl;
        if (rgb_buffer.is_valid()) {
            result.width = rgb_buffer.width;
            result.height = rgb_buffer.height;
            result.channels = rgb_buffer.channels;

            std::cout << "🔧 LibRaw_Wrapper output bps:" << current_params.output_bps << std::endl;
            switch(current_params.output_bps) {
            case 8:     result.bits_per_sample = 8;     break;
            case 16:    result.bits_per_sample = 16;    break;
            default:    result.bits_per_sample = 32;    break;
            }
            result.data = reinterpret_cast<float*>(rgb_buffer.image);
            
            result.error_code = LIBRAW_SUCCESS;
            result.timing_info = timing_info;  // 計測情報を含める
            
            return result;
        }
#endif
        
        // Standard LibRaw processing
        int error_code;
        libraw_processed_image_t* processed_image = processor.dcraw_make_mem_image(&error_code);
        
        if (!processed_image) {
            result.error_code = error_code;
            return result;
        }
        
        result.width = processed_image->width;
        result.height = processed_image->height;
        result.channels = processed_image->colors;
        result.bits_per_sample = processed_image->bits;
        
        // データをコピー
        //size_t data_size = result.width * result.height * result.channels * (result.bits_per_sample / 8);
        //result.data.resize(data_size);
        //memcpy(result.data.data(), processed_image->data, data_size);
        
        result.error_code = LIBRAW_SUCCESS;
        result.timing_info = timing_info;  // 計測情報を含める
        
        LibRaw::dcraw_clear_mem(processed_image);
        return result;
    }
    
    void set_debug_mode(bool enable) {
        debug_mode = enable;
        // Debug mode setting simplified
    }

#ifdef __arm64__
    void set_processing_params(const ProcessingParams& params) {
        current_params = params;
        
        // Map all parameters to LibRaw
        
        // Basic processing parameters
        processor.imgdata.params.use_camera_wb = params.use_camera_wb ? 1 : 0;
        processor.imgdata.params.half_size = params.half_size ? 1 : 0;
        processor.imgdata.params.four_color_rgb = params.four_color_rgb ? 1 : 0;
        processor.imgdata.params.output_bps = params.output_bps;
        processor.imgdata.params.user_flip = params.user_flip;
        
        // Demosaicing parameters
        processor.imgdata.params.user_qual = params.demosaic_algorithm;
        processor.imgdata.params.dcb_iterations = params.dcb_iterations;
        processor.imgdata.params.dcb_enhance_fl = params.dcb_enhance ? 1 : 0;
        
        // Noise reduction parameters
        processor.imgdata.params.fbdd_noiserd = params.fbdd_noise_reduction;
        processor.imgdata.params.threshold = params.noise_thr;
        processor.imgdata.params.med_passes = params.median_filter_passes;
        
        // White balance parameters
        processor.imgdata.params.use_auto_wb = params.use_auto_wb ? 1 : 0;
        processor.imgdata.params.user_mul[0] = params.user_wb[0];
        processor.imgdata.params.user_mul[1] = params.user_wb[1];
        processor.imgdata.params.user_mul[2] = params.user_wb[2];
        processor.imgdata.params.user_mul[3] = params.user_wb[3];
        
        // Color space and output parameters
        processor.imgdata.params.output_color = params.output_color_space;
        
        // Brightness and exposure parameters
        processor.imgdata.params.bright = params.bright;
        processor.imgdata.params.no_auto_bright = params.no_auto_bright ? 1 : 0;
        processor.imgdata.params.auto_bright_thr = params.auto_bright_thr;
        processor.imgdata.params.adjust_maximum_thr = params.adjust_maximum_thr;
        
        // Highlight processing
        processor.imgdata.params.highlight = params.highlight_mode;
        
        // Exposure correction parameters
        processor.imgdata.params.exp_shift = params.exp_shift;
        processor.imgdata.params.exp_preser = params.exp_preserve_highlights;
        
        // Gamma correction parameters
        processor.imgdata.params.gamm[0] = 1.0 / params.gamma_power;
        processor.imgdata.params.gamm[1] = params.gamma_slope;
        // Set no_auto_scale
        if (params.no_auto_scale) {
            processor.imgdata.params.no_auto_scale = 1;
        }
        
        // Color correction parameters
        processor.imgdata.params.aber[0] = params.chromatic_aberration_red;
        processor.imgdata.params.aber[1] = 1.0;  // Green channel (no correction)
        processor.imgdata.params.aber[2] = params.chromatic_aberration_blue;
        processor.imgdata.params.aber[3] = 1.0;  // Green channel (no correction)
        
        // User adjustments
        if (params.user_black >= 0) {
            processor.imgdata.params.user_black = params.user_black;
        }
        if (params.user_sat >= 0) {
            processor.imgdata.params.user_sat = params.user_sat;
        }
        
        // File-based corrections
        if (!params.bad_pixels_path.empty()) {
            processor.imgdata.params.bad_pixels = const_cast<char*>(params.bad_pixels_path.c_str());
        }
    }
    
    std::string get_device_info() const {
        if (accelerator) {
            return accelerator->get_device_info();
        }
        return "Metal not available";
    }
    
    std::vector<uint16_t> get_raw_image() {
        if (!processor.imgdata.rawdata.raw_image) {
            throw std::runtime_error("RAW data not available - call unpack() first");
        }
        
        size_t raw_width = processor.imgdata.sizes.raw_width;
        size_t raw_height = processor.imgdata.sizes.raw_height;
        size_t pixel_count = raw_width * raw_height;
        
        std::vector<uint16_t> raw_data(pixel_count);
        std::copy(processor.imgdata.rawdata.raw_image, 
                  processor.imgdata.rawdata.raw_image + pixel_count, 
                  raw_data.begin());
        
        return raw_data;
    }
    
    py::array_t<uint16_t> get_raw_image_as_numpy() {
        if (!processor.imgdata.rawdata.raw_image) {
            throw std::runtime_error("RAW data not available - call unpack() first");
        }
        
        size_t raw_width = processor.imgdata.sizes.raw_width;
        size_t raw_height = processor.imgdata.sizes.raw_height;
        size_t total_pixels = raw_width * raw_height;
        
        // Create numpy array with copied data (safer than direct memory reference)
        auto result = py::array_t<uint16_t>({raw_height, raw_width});
        auto buf = result.request();
        uint16_t* ptr = static_cast<uint16_t*>(buf.ptr);
        
        // Copy data from LibRaw buffer
        std::copy(processor.imgdata.rawdata.raw_image,
                  processor.imgdata.rawdata.raw_image + total_pixels,
                  ptr);
        
        return result;
    }
    
    
    // LibRaw recover_highlights equivalent for float32 processing
    bool recover_highlights(ImageBufferFloat& rgb_buffer, int highlight_mode) {
        std::cout << "🔧 Starting highlight recovery (float32)..." << std::endl;
        
        highlight_mode = 4;
        if (highlight_mode <= 2) {
            return true; // No highlight recovery needed
        }
        
        const float grow = std::pow(2.0f, 4.0f - highlight_mode);
        const int SCALE = 4; // Fixed scale factor
        
        // Calculate saturation thresholds for each channel
        float hsat[4];
        const float max_value = 1.0f; // For float32 [0,1] range
        const auto& pre_mul = processor.imgdata.color.pre_mul;
        
        for (int c = 0; c < 4; c++) {
            hsat[c] = 0.95f * max_value * pre_mul[c]; // 95% of max as saturation threshold
        }
        
        // Find the channel with minimum pre_mul (reference channel)
        int kc = 0;
        for (int c = 1; c < 3; c++) {
            if (pre_mul[kc] < pre_mul[c]) {
                kc = c;
            }
        }
        
        const size_t width = rgb_buffer.width;
        const size_t height = rgb_buffer.height;
        const size_t channels = rgb_buffer.channels;
        
        const size_t high = height / SCALE;
        const size_t wide = width / SCALE;
        
        // Allocate highlight recovery map
        std::vector<float> map(high * wide, 0.0f);
        
        // Process each color channel (except reference channel)
        for (int c = 0; c < (int)channels; c++) {
            if (c == kc) continue;
            
            std::cout << "🔧 Processing channel " << c << " for highlight recovery..." << std::endl;
            
            // Clear map for this channel
            std::fill(map.begin(), map.end(), 0.0f);
            
            // Calculate initial ratios for saturated regions
            for (size_t mrow = 0; mrow < high; mrow++) {
                for (size_t mcol = 0; mcol < wide; mcol++) {
                    float sum = 0.0f, wgt = 0.0f;
                    int count = 0;
                    
                    // Sample the SCALE x SCALE block
                    for (size_t row = mrow * SCALE; row < (mrow + 1) * SCALE && row < height; row++) {
                        for (size_t col = mcol * SCALE; col < (mcol + 1) * SCALE && col < width; col++) {
                            const size_t pixel_idx = row * width + col;
                            const float pixel_c = rgb_buffer.image[pixel_idx][c];
                            const float pixel_kc = rgb_buffer.image[pixel_idx][kc];
                            
                            // Check if pixel is saturated in channel c but not in reference channel
                            if (pixel_c >= hsat[c] * 0.99f && pixel_kc > 0.8f * max_value) {
                                sum += pixel_c;
                                wgt += pixel_kc;
                                count++;
                            }
                        }
                    }
                    
                    // If entire block is saturated, calculate ratio
                    if (count == SCALE * SCALE) {
                        map[mrow * wide + mcol] = (wgt > 0) ? sum / wgt : 1.0f;
                    }
                }
            }
            
            // Iteratively fill gaps in the map using neighbor averaging
            for (int spread = static_cast<int>(32 / grow); spread > 0; spread--) {
                bool changed = false;
                std::vector<float> temp_map = map;
                
                for (size_t mrow = 0; mrow < high; mrow++) {
                    for (size_t mcol = 0; mcol < wide; mcol++) {
                        if (map[mrow * wide + mcol] != 0.0f) continue;
                        
                        float sum = 0.0f;
                        int count = 0;
                        
                        // Check 8-connected neighbors
                        const int dir[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}};
                        
                        for (int d = 0; d < 8; d++) {
                            int y = static_cast<int>(mrow) + dir[d][0];
                            int x = static_cast<int>(mcol) + dir[d][1];
                            
                            if (y >= 0 && y < static_cast<int>(high) && 
                                x >= 0 && x < static_cast<int>(wide) && 
                                map[y * wide + x] > 0) {
                                int weight = 1 + (d & 1); // Diagonal neighbors get weight 2, orthogonal get weight 1
                                sum += weight * map[y * wide + x];
                                count += weight;
                            }
                        }
                        
                        if (count > 3) {
                            temp_map[mrow * wide + mcol] = (sum + grow) / (count + grow);
                            changed = true;
                        }
                    }
                }
                
                map = temp_map;
                if (!changed) break;
            }
            
            // Fill remaining zeros with 1.0
            for (size_t i = 0; i < map.size(); i++) {
                if (map[i] == 0.0f) map[i] = 1.0f;
            }
            
            // Apply highlight recovery to the image
            for (size_t mrow = 0; mrow < high; mrow++) {
                for (size_t mcol = 0; mcol < wide; mcol++) {
                    const float ratio = map[mrow * wide + mcol];
                    
                    for (size_t row = mrow * SCALE; row < (mrow + 1) * SCALE && row < height; row++) {
                        for (size_t col = mcol * SCALE; col < (mcol + 1) * SCALE && col < width; col++) {
                            const size_t pixel_idx = row * width + col;
                            
                            if (rgb_buffer.image[pixel_idx][c] >= hsat[c] * 0.99f) {
                                float val = rgb_buffer.image[pixel_idx][kc] * ratio;
                                if (rgb_buffer.image[pixel_idx][c] < val) {
                                    rgb_buffer.image[pixel_idx][c] = std::min(val, max_value);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        std::cout << "✅ Highlight recovery completed successfully" << std::endl;
        return true;
    }
#endif
};

// LibRawWrapper実装
LibRawWrapper::LibRawWrapper() : pimpl(std::make_unique<Impl>()) {
}

LibRawWrapper::~LibRawWrapper() {
}

int LibRawWrapper::load_file(const std::string& filename) {
    return pimpl->load_file(filename);
}

int LibRawWrapper::unpack() {
    return pimpl->unpack();
}

int LibRawWrapper::process() {
    return pimpl->process();
}

ProcessedImageData LibRawWrapper::get_processed_image() {
    return pimpl->get_processed_image();
}

void LibRawWrapper::set_debug_mode(bool enable) {
    pimpl->set_debug_mode(enable);
}

#ifdef __arm64__
void LibRawWrapper::set_processing_params(const ProcessingParams& params) {
    pimpl->set_processing_params(params);
}

void LibRawWrapper::enable_gpu_acceleration(bool enable) {
    pimpl->accelerator->set_use_gpu_acceleration(enable);
}

std::string LibRawWrapper::get_device_info() const {
    return pimpl->get_device_info();
}

// New methods for high-level API support
int LibRawWrapper::load_buffer(const std::vector<uint8_t>& buffer) {
    // For prototype: just return success, actual implementation would use LibRaw buffer loading
    return pimpl->load_file(""); // Placeholder
}

std::vector<uint16_t> LibRawWrapper::get_raw_image() {
    return pimpl->get_raw_image();
}

py::array_t<uint16_t> LibRawWrapper::get_raw_image_as_numpy() {
    return pimpl->get_raw_image_as_numpy();
}

ImageInfo LibRawWrapper::get_image_info() {
    ImageInfo info;
    // Extract info from LibRaw
    libraw_image_sizes_t& sizes = pimpl->processor.imgdata.sizes;
    libraw_iparams_t& iparams = pimpl->processor.imgdata.idata;
    
    info.width = sizes.width;
    info.height = sizes.height;
    info.raw_width = sizes.raw_width;
    info.raw_height = sizes.raw_height;
    info.colors = iparams.colors;
    info.camera_make = std::string(iparams.make);
    info.camera_model = std::string(iparams.model);
    
    // Copy white balance multipliers if available
    if (pimpl->processor.imgdata.color.cam_mul[0] > 0) {
        for (int i = 0; i < 4; i++) {
            info.cam_mul[i] = pimpl->processor.imgdata.color.cam_mul[i];
            info.pre_mul[i] = pimpl->processor.imgdata.color.pre_mul[i];
        }
    }
    
    return info;
}

std::string LibRawWrapper::get_camera_make() const {
    return std::string(pimpl->processor.imgdata.idata.make);
}

std::string LibRawWrapper::get_camera_model() const {
    return std::string(pimpl->processor.imgdata.idata.model);
}

ProcessedImageData LibRawWrapper::process_with_dict(
    const std::map<std::string, float>& float_params,
    const std::map<std::string, int>& int_params, 
    const std::map<std::string, bool>& bool_params,
    const std::map<std::string, std::string>& string_params) {
    
    // Convert parameters to ProcessingParams
    ProcessingParams params;
    
    // Extract parameters from maps
    for (const auto& p : bool_params) {
        if (p.first == "use_camera_wb") params.use_camera_wb = p.second;
        else if (p.first == "half_size") params.half_size = p.second;
        else if (p.first == "four_color_rgb") params.four_color_rgb = p.second;
        else if (p.first == "use_auto_wb") params.use_auto_wb = p.second;
        else if (p.first == "no_auto_bright") params.no_auto_bright = p.second;
        else if (p.first == "dcb_enhance") params.dcb_enhance = p.second;
        else if (p.first == "no_auto_scale") params.no_auto_scale = p.second;
        else if (p.first == "use_gpu_acceleration") params.use_gpu_acceleration = p.second;
    }
    
    for (const auto& p : int_params) {
        if (p.first == "demosaic_algorithm") params.demosaic_algorithm = p.second;
        else if (p.first == "output_bps") params.output_bps = p.second;
        else if (p.first == "user_flip") params.user_flip = p.second;
        else if (p.first == "dcb_iterations") params.dcb_iterations = p.second;
        else if (p.first == "fbdd_noise_reduction") params.fbdd_noise_reduction = p.second;
        else if (p.first == "median_filter_passes") params.median_filter_passes = p.second;
        else if (p.first == "output_color_space") params.output_color_space = p.second;
        else if (p.first == "highlight_mode") params.highlight_mode = p.second;
        else if (p.first == "user_black") params.user_black = p.second;
        else if (p.first == "user_sat") params.user_sat = p.second;
    }
    
    for (const auto& p : float_params) {
        if (p.first == "noise_thr") params.noise_thr = p.second;
        else if (p.first == "bright") params.bright = p.second;
        else if (p.first == "auto_bright_thr") params.auto_bright_thr = p.second;
        else if (p.first == "adjust_maximum_thr") params.adjust_maximum_thr = p.second;
        else if (p.first == "exp_shift") params.exp_shift = p.second;
        else if (p.first == "exp_preserve_highlights") params.exp_preserve_highlights = p.second;
        else if (p.first == "gamma_power") params.gamma_power = p.second;
        else if (p.first == "gamma_slope") params.gamma_slope = p.second;
        else if (p.first == "chromatic_aberration_red") params.chromatic_aberration_red = p.second;
        else if (p.first == "chromatic_aberration_blue") params.chromatic_aberration_blue = p.second;
    }
    
    for (const auto& p : string_params) {
        if (p.first == "bad_pixels_path") params.bad_pixels_path = p.second;
    }
    
    // Apply parameters and process
    pimpl->set_processing_params(params);
    
    // CRITICAL: Unpack the RAW data before processing (only if not already unpacked)
    if (!pimpl->processor.imgdata.rawdata.raw_image) {
        int unpack_result = pimpl->unpack();
        if (unpack_result != LIBRAW_SUCCESS) {
            ProcessedImageData error_result;
            error_result.error_code = unpack_result;
            return error_result;
        }
    }
    
    // Process the image
    int process_result = pimpl->process();
    if (process_result != LIBRAW_SUCCESS) {
        ProcessedImageData error_result;
        error_result.error_code = process_result;
        return error_result;
    }
    
    return pimpl->get_processed_image();
}

void LibRawWrapper::close() {
    // Reset LibRaw processor
    pimpl->processor.recycle();
}

// rawpy完全互換性のための処理パラメータ変換関数
ProcessingParams create_params_from_rawpy_args(
    // Basic parameters
    bool use_camera_wb,
    bool half_size,
    bool four_color_rgb,
    int output_bps,
    int user_flip,
    
    // Demosaicing parameters
    int demosaic_algorithm,
    int dcb_iterations,
    bool dcb_enhance,
    
    // Noise reduction parameters
    int fbdd_noise_reduction,
    float noise_thr,
    int median_filter_passes,
    
    // White balance parameters
    bool use_auto_wb,
    const std::array<float, 4>& user_wb,
    
    // Color and output parameters
    int output_color,
    
    // Brightness and exposure parameters
    float bright,
    bool no_auto_bright,
    float auto_bright_thr,
    float adjust_maximum_thr,
    
    // Highlight processing
    int highlight_mode,
    
    // Exposure correction parameters
    float exp_shift,
    float exp_preserve_highlights,
    
    // Gamma and scaling
    const std::pair<float, float>& gamma,
    bool no_auto_scale,
    
    // Color correction parameters
    float chromatic_aberration_red,
    float chromatic_aberration_blue,
    
    // User adjustments
    int user_black,
    int user_sat,
    
    // File-based corrections
    const std::string& bad_pixels_path,
    
    // LibRaw Enhanced extensions
    bool use_gpu_acceleration
) {
    ProcessingParams params;
    
    // Map all parameters to ProcessingParams structure
    
    // Basic processing parameters
    params.use_camera_wb = use_camera_wb;
    params.half_size = half_size;
    params.four_color_rgb = four_color_rgb;
    params.output_bps = output_bps;
    params.user_flip = user_flip;
    
    // Demosaicing parameters
    params.demosaic_algorithm = demosaic_algorithm;
    params.dcb_iterations = dcb_iterations;
    params.dcb_enhance = dcb_enhance;
    
    // Noise reduction parameters
    params.fbdd_noise_reduction = fbdd_noise_reduction;
    params.noise_thr = noise_thr;
    params.median_filter_passes = median_filter_passes;
    
    // White balance parameters
    params.use_auto_wb = use_auto_wb;
    params.user_wb[0] = user_wb[0];
    params.user_wb[1] = user_wb[1];
    params.user_wb[2] = user_wb[2];
    params.user_wb[3] = user_wb[3];
    
    // Color space and output parameters
    params.output_color_space = output_color;
    
    // Brightness and exposure parameters
    params.bright = bright;
    params.no_auto_bright = no_auto_bright;
    params.auto_bright_thr = auto_bright_thr;
    params.adjust_maximum_thr = adjust_maximum_thr;
    
    // Highlight processing
    params.highlight_mode = highlight_mode;
    
    // Exposure correction parameters
    params.exp_shift = exp_shift;
    params.exp_preserve_highlights = exp_preserve_highlights;
    
    // Gamma correction parameters
    params.gamma_power = gamma.first;
    params.gamma_slope = gamma.second;
    params.no_auto_scale = no_auto_scale;
    
    // Color correction parameters
    params.chromatic_aberration_red = chromatic_aberration_red;
    params.chromatic_aberration_blue = chromatic_aberration_blue;
    
    // User adjustments
    params.user_black = user_black;
    params.user_sat = user_sat;
    
    // File-based corrections
    params.bad_pixels_path = bad_pixels_path;
    
    // Metal-specific settings
    params.use_gpu_acceleration = use_gpu_acceleration;
    
    return params;
}

// Platform detection functions implementation
bool is_apple_silicon() {
#ifdef __arm64__
    return true;
#else
    return false;
#endif
}

bool is_available() {
    // Backward compatibility - check if Metal is available without creating instances
#ifdef __arm64__
    return is_available(); // Simple check - Apple Silicon has Metal support
#else
    return false;
#endif
}

std::vector<std::string> get_device_list() {
    std::vector<std::string> device_list;
    
#ifdef __arm64__
#ifdef __OBJC__
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            std::string device_name = std::string([device.name UTF8String]);
            device_list.push_back(device_name);
        }
    }
#endif
#endif
    
    return device_list;
}

#endif

} // namespace libraw_enhanced