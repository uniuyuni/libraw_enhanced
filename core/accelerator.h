//
// accelerator.h
// LibRaw Enhanced - Unified GPU/CPU Acceleration
// Common definitions and unified interface
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <mutex>

namespace libraw_enhanced {

// ✨ D50白色点 (ProPhotoRGBに準拠)
static constexpr float d50_white[3] = {0.9642f, 1.0f, 0.8249f};

// ✨ XYZから線形ProPhotoRGBへの変換マトリクス
static constexpr float xyz_rgb[3][3] = {
    {1.34578f, -0.04008f, -0.04033f},
    {-0.25556f, 1.15949f, 0.09109f},
    {0.00912f, -0.00994f, 1.00693f}
};

// ===== COMMON DEFINITIONS =====
// These definitions are shared across CPU, GPU, and wrapper components

// Core image buffer structure
struct ImageBuffer {
    uint16_t (*image)[4] = nullptr;
    size_t width = 0;
    size_t height = 0;
    size_t channels = 0;

    bool is_valid() const { return image != nullptr && width > 0 && height > 0 && channels > 0; }
};

// Float32 processing structures
struct ImageBufferFloat {
    float (*image)[3] = nullptr;
    size_t width = 0;
    size_t height = 0;
    size_t channels = 3;  // Always 3 (RGB)
    
    bool is_valid() const { return image != nullptr && width > 0 && height > 0; }
};

// Processing parameters for all components
struct ProcessingParams {
    // Basic processing parameters
    bool use_camera_wb = true;
    bool half_size = false;
    bool four_color_rgb = false;
    int output_bps = 16;
    int user_flip = -1;  // -1: auto, 0,1,2,3: rotation angles
    
    // Demosaicing parameters
    int demosaic_algorithm = 1;  // 0: bilinear, 1: VNG, 2: PPG, 3: AHD, 4: DCB
    int dcb_iterations = 0;
    bool dcb_enhance = false;
    
    // Noise reduction parameters
    int fbdd_noise_reduction = 0;  // 0: off, 1: light, 2: full
    float noise_thr = 0.0f;
    int median_filter_passes = 0;
    
    // White balance parameters
    bool use_auto_wb = false;
    float user_wb[4] = {1.0f, 1.0f, 1.0f, 1.0f};  // RGBG multipliers
    
    // Color space and output parameters
    int output_color_space = 1;  // 1: sRGB, 2: Adobe RGB, 3: ProPhoto RGB, 4: XYZ, 5: ACES
    float color_matrix[9];
    
    // Brightness and exposure parameters
    float bright = 1.0f;
    bool no_auto_bright = false;
    float auto_bright_thr = 0.01f;
    float adjust_maximum_thr = 0.75f;
    
    // Highlight processing
    int highlight_mode = 0;  // 0: clip, 1: unclip, 2: blend, 3+: rebuild
    
    // Exposure correction (new rawpy-compatible parameters)
    float exp_shift = 1.0f;  // Linear exposure shift (0.25-8.0 range)
    float exp_preserve_highlights = 0.0f;  // Highlight preservation (0.0-1.0)
    
    // Gamma correction parameters
    float gamma_power = 0.f; // 2.222f;
    float gamma_slope = 0.f; // 4.5f;
    bool no_auto_scale = false;
    
    // Color correction parameters
    float chromatic_aberration_red = 1.0f;
    float chromatic_aberration_blue = 1.0f;
    
    // User adjustments
    int user_black = -1;  // -1: auto, or custom black level
    int user_sat = -1;    // -1: auto, or custom saturation
    
    // File-based corrections
    std::string bad_pixels_path;
    
    // Acceleration settings
    bool use_gpu_acceleration = false;   // Use GPU Metal acceleration
    
    ProcessingParams() {
        // Initialize color matrix to identity
        for (int i = 0; i < 9; i++) {
            color_matrix[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        }
    }
};

#define FILTERS_XTRANS 9

// Demosaicing algorithm enumeration - see constants.h for rawpy/LibRaw compatible definitions

// Generic memory manager with malloc/free for any data type
class BufferManager {
private:
    std::unordered_map<void*, size_t> allocated_blocks_;  // {pointer, size_in_bytes}
    mutable std::mutex mutex_;  // Thread safety
    
public:
    BufferManager() = default;
    
    ~BufferManager() { 
        clear(); 
    }
    
    // Allocate memory block of specified size in bytes
    void* allocate(size_t bytes) {
        void* ptr = std::malloc(bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        allocated_blocks_[ptr] = bytes;
        return ptr;
    }
    
    // Deallocate specific memory block
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocated_blocks_.find(ptr);
        if (it != allocated_blocks_.end()) {
            std::free(ptr);
            allocated_blocks_.erase(it);
        }
    }
    
    // Clear all allocated memory (called by destructor)
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : allocated_blocks_) {
            std::free(pair.first);
        }
        allocated_blocks_.clear();
    }
    
    // Debug information
    size_t get_allocated_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocated_blocks_.size();
    }
    
    size_t get_total_memory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = 0;
        for (const auto& pair : allocated_blocks_) {
            total += pair.second;
        }
        return total;
    }
};


// ===== UNIFIED ACCELERATOR INTERFACE =====

class Accelerator {
public:
    Accelerator();
    ~Accelerator();
    
    // Initialization and device info
    bool initialize();
    bool is_available() const;

    std::string get_device_info() const;
    void set_debug_mode(bool enable) { /* Simplified debug mode */ }
    
    // GPU acceleration control
    void set_use_gpu_acceleration(bool enable);
    
    // Pre-interpolation processing (border handling, hot pixels, etc.)
    bool pre_interpolate(ImageBuffer& image_buffer, uint32_t filters, const char (&xtrans)[6][6], bool half_size = false);
    
    // Bayer demosaicing methods
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
                            int iterations = 1,   // デフォルト値
                            bool dcb_enhance = true); // デフォルト値
                           
    bool demosaic_bayer_amaze(const ImageBuffer& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            uint32_t filters,
                            const float (&cam_mul)[4],
                            uint16_t maximum_value);
    
    // Unified demosaic compute method with CPU/GPU selection
    bool demosaic_compute(const ImageBuffer& raw_buffer,
                            ImageBufferFloat& rgb_buffer,
                            int algorithm,
                            uint32_t filters,
                            const char (&xtrans)[6][6],
                            const float (&color_matrix)[3][4],
                            const float (&cam_mul)[4],
                            uint16_t maximum_value);
    
    // X-Trans demosaicing methods  
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
                                   
    // White balance methods
    bool apply_white_balance(const ImageBufferFloat& rgb_input,
                            ImageBufferFloat& rgb_output,
                            const float wb_multipliers[4]);

    // Camera matrix-based color space conversion
    bool convert_color_space(const ImageBufferFloat& rgb_input,
                            ImageBufferFloat& rgb_output,
                            const float transform[3][4]);

    // Gamma correction method
    bool gamma_correct(const ImageBufferFloat& rgb_input,
                        ImageBufferFloat& rgb_output,
                        float gamma_power = 0.f, //2.2f,
                        float gamma_slope = 0.f, //4.5f,
                        int output_color_space = 1); // 1=sRGB, 2=Adobe RGB, etc.

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    
    // GPU/CPU selection logic
    bool should_use_gpu() const;
};

} // namespace libraw_enhanced