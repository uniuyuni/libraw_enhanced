//
// accelerator.cpp
// LibRaw Enhanced - Unified GPU/CPU Acceleration
//

#include "accelerator.h"
#include "cpu_accelerator.h"
#include "gpu_accelerator.h"
#include "constants.h"
#include <iostream>
#include <memory>

namespace libraw_enhanced {

class Accelerator::Impl {
public:
    std::unique_ptr<CPUAccelerator> cpu_accelerator;
    std::unique_ptr<GPUAccelerator> gpu_accelerator;
    bool gpu_available = false;
    
    Impl() : cpu_accelerator(std::make_unique<CPUAccelerator>()),
             gpu_accelerator(std::make_unique<GPUAccelerator>()) {}
};

Accelerator::Accelerator() : pimpl_(std::make_unique<Impl>()) {}
Accelerator::~Accelerator() = default;

bool Accelerator::initialize() {
    std::cout << "🚀 Initializing Unified Metal Acceleration..." << std::endl;
    
    // Always initialize CPU accelerator
    if (!pimpl_->cpu_accelerator->initialize()) {
        std::cerr << "❌ Failed to initialize CPU accelerator" << std::endl;
        return false;
    }
    
    // Try to initialize GPU accelerator
    if (pimpl_->gpu_accelerator->initialize()) {
        pimpl_->gpu_available = true;
        std::cout << "✅ GPU Metal acceleration available: " << pimpl_->gpu_accelerator->get_device_info() << std::endl;
    } else {
        std::cout << "⚠️  GPU Metal acceleration not available, using CPU only" << std::endl;
    }
    
    return true;
}

bool Accelerator::should_use_gpu(const ProcessingParams& params) const {
    // Use GPU if:
    // 1. GPU is available
    // 2. Metal acceleration is enabled in params
    // Note: Image size check is handled elsewhere
    return pimpl_->gpu_available && params.use_gpu_acceleration;
}

bool Accelerator::is_available() const {
    return pimpl_->cpu_accelerator->is_available() || pimpl_->gpu_available;
}

std::string Accelerator::get_device_info() const {
    if (pimpl_->gpu_available) {
        return pimpl_->cpu_accelerator->get_device_info() + " + " + pimpl_->gpu_accelerator->get_device_info();
    }
    return pimpl_->cpu_accelerator->get_device_info();
}

// Processing methods with GPU/CPU selection
bool Accelerator::demosaic_bayer_linear(const ImageBuffer& raw_buffer,
                                           ImageBuffer& rgb_buffer,
                                           uint32_t filters) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Trying GPU Linear demosaic" << std::endl;
        // GPU accelerator needs to be updated to accept ImageBufferFloat32
        // For now, fall back to CPU immediately
        std::cout << "⚠️ GPU not yet updated for ImageBuffer, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU Linear demosaic" << std::endl;
    return pimpl_->cpu_accelerator->demosaic_bayer_linear(raw_buffer, rgb_buffer, filters);
}

bool Accelerator::demosaic_bayer_aahd(const ImageBuffer& raw_buffer,
                                        ImageBuffer& rgb_buffer,
                                        uint32_t filters) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Trying GPU AAHD demosaic" << std::endl;
        // GPU accelerator needs to be updated to accept ImageBufferFloat32
        // For now, fall back to CPU immediately
        std::cout << "⚠️ GPU not yet updated for ImageBuffer, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU AHD demosaic" << std::endl;
    return pimpl_->cpu_accelerator->demosaic_bayer_aahd(raw_buffer, rgb_buffer, filters);
}

bool Accelerator::demosaic_bayer_dcb(const ImageBuffer& raw_buffer,
                                        ImageBuffer& rgb_buffer,
                                        uint32_t filters,
                                        int iterations,
                                        bool dcb_enhance) {

    if (pimpl_->gpu_available) {
        std::cout << "🎯 Trying GPU DCB demosaic" << std::endl;
        // GPU accelerator needs to be updated to accept ImageBufferFloat32
        // For now, fall back to CPU immediately
        std::cout << "⚠️ GPU not yet updated for ImageBuffer, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU DCB demosaic" << std::endl;
    return pimpl_->cpu_accelerator->demosaic_bayer_dcb(raw_buffer, rgb_buffer, filters, iterations, dcb_enhance);
}

bool Accelerator::demosaic_bayer_amaze(const ImageBuffer& raw_buffer,
                                          ImageBuffer& rgb_buffer,
                                          uint32_t filters) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Trying GPU AMaZE demosaic" << std::endl;
        // GPU accelerator needs to be updated to accept ImageBufferFloat32
        // For now, fall back to CPU immediately
        std::cout << "⚠️ GPU not yet updated for ImageBuffer, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU AMaZE demosaic" << std::endl;
    return pimpl_->cpu_accelerator->demosaic_bayer_amaze(raw_buffer, rgb_buffer, filters);
}

// Unified demosaic computation with CPU/GPU selection
bool Accelerator::demosaic_compute(const ImageBuffer& raw_buffer,
                                    ImageBuffer& rgb_buffer,
                                    int algorithm,
                                    uint32_t filters,
                                    const char (&xtrans)[6][6],
                                    const float (&color_matrix)[3][4]) {
    std::cout << "🎯 Unified demosaic compute (algorithm " << algorithm << ")" << std::endl;

    if (filters == FILTERS_XTRANS) {
        // xtrans data
        switch (algorithm) {
            case static_cast<int>(DemosaicAlgorithm::Linear): return demosaic_xtrans_1pass(raw_buffer, rgb_buffer, xtrans, color_matrix);
            default:
                return demosaic_xtrans_3pass(raw_buffer, rgb_buffer, xtrans, color_matrix);
        }
    } else {
    
        // Map algorithm to specific methods using enum constants
        switch (algorithm) {
            case static_cast<int>(DemosaicAlgorithm::Linear): 
                std::cout << "🔧 Calling demosaic_bayer_linear..." << std::endl;
                return demosaic_bayer_linear(raw_buffer, rgb_buffer, filters);
            case static_cast<int>(DemosaicAlgorithm::AAHD): 
                std::cout << "🔧 Calling demosaic_bayer_aahd..." << std::endl;
                return demosaic_bayer_aahd(raw_buffer, rgb_buffer, filters);
            case static_cast<int>(DemosaicAlgorithm::DCB): 
                std::cout << "🔧 Calling demosaic_bayer_dcb..." << std::endl;
                return demosaic_bayer_dcb(raw_buffer, rgb_buffer, filters);
            case static_cast<int>(DemosaicAlgorithm::AMaZE): 
                std::cout << "🔧 Calling demosaic_bayer_amaze..." << std::endl;
                return demosaic_bayer_amaze(raw_buffer, rgb_buffer, filters);
            default:
                std::cout << "⚠️ Unknown algorithm " << algorithm << ", using Linear" << std::endl;
                return demosaic_bayer_linear(raw_buffer, rgb_buffer, filters);
        }
    }
}

bool Accelerator::demosaic_xtrans_3pass(const ImageBuffer& raw_buffer,
                                        ImageBuffer& rgb_buffer,
                                        const char (&xtrans)[6][6],
                                        const float (&color_matrix)[3][4]) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Using GPU X-Trans 3-pass demosaic" << std::endl;
        // GPU accelerator needs to be updated to accept ImageBufferFloat32
        // For now, fallback to CPU
        std::cout << "⚠️ GPU not yet updated for ImageBuffer, using CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU X-Trans 3-pass demosaic" << std::endl;
    return pimpl_->cpu_accelerator->demosaic_xtrans_3pass(raw_buffer, rgb_buffer, xtrans, color_matrix);
}

bool Accelerator::demosaic_xtrans_1pass(const ImageBuffer& raw_buffer,
                                        ImageBuffer& rgb_buffer,
                                        const char (&xtrans)[6][6],
                                        const float (&color_matrix)[3][4]) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Using GPU X-Trans 1-pass demosaic" << std::endl;
        // GPU accelerator needs to be updated to accept ImageBufferFloat32
        // For now, fallback to CPU
        std::cout << "⚠️ GPU not yet updated for ImageBuffer, using CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU X-Trans 1-pass demosaic" << std::endl;
    return pimpl_->cpu_accelerator->demosaic_xtrans_1pass(raw_buffer, rgb_buffer, xtrans, color_matrix);
}

bool Accelerator::apply_white_balance(const ImageBufferFloat32& rgb_input,
                                            ImageBufferFloat32& rgb_output,
                                            const float wb_multipliers[4]) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Using GPU white balance on RAW" << std::endl;
        // GPU accelerator needs to be updated to accept ImageBufferFloat32
        // For now, fallback to CPU
        std::cout << "⚠️ GPU not yet updated for ImageBufferFloat32, using CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU white balance on RAW" << std::endl;
    return pimpl_->cpu_accelerator->apply_white_balance(rgb_input, rgb_output, wb_multipliers);
}

// Color space conversion methods
// Camera matrix-based color space conversion
bool Accelerator::convert_color_space(const ImageBufferFloat32& rgb_input,
                                      ImageBufferFloat32& rgb_output,
                                      const float transform[3][4]) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Using GPU camera matrix color conversion" << std::endl;
        // GPU accelerator needs to be updated for camera matrix support
        // For now, fallback to CPU
        std::cout << "⚠️ GPU not yet updated for camera matrix, using CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU camera matrix color conversion" << std::endl;
    return pimpl_->cpu_accelerator->convert_color_space(rgb_input, rgb_output, transform);
}

bool Accelerator::gamma_correct(const ImageBufferFloat32& rgb_input,
                               ImageBufferFloat32& rgb_output,
                               float gamma_power,
                               float gamma_slope,
                               int output_color_space) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Using GPU gamma correction" << std::endl;
        // GPU accelerator needs to be updated to accept ImageBufferFloat32
        // For now, fallback to CPU
        std::cout << "⚠️ GPU not yet updated for ImageBufferFloat32, using CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU gamma correction" << std::endl;
    return pimpl_->cpu_accelerator->gamma_correct(rgb_input, rgb_output, gamma_power, gamma_slope, output_color_space);
}

bool Accelerator::pre_interpolate(ImageBuffer& image_buffer, uint32_t filters, const char (&xtrans)[6][6], bool half_size) {
    if (pimpl_->gpu_available) {
        std::cout << "🎯 Trying GPU pre-interpolation" << std::endl;
        // GPU accelerator needs to be updated for pre-interpolation
        // For now, fallback to CPU immediately
        std::cout << "⚠️ GPU not yet updated for pre-interpolation, using CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU pre-interpolation" << std::endl;
    return pimpl_->cpu_accelerator->pre_interpolate(image_buffer, filters, xtrans, half_size);
}

// Note: Pipeline processing has been moved to libraw_wrapper.cpp
// This class now serves as a pure CPU/GPU dispatch layer

} // namespace libraw_enhanced