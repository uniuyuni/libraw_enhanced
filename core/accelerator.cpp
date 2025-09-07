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

    bool use_gpu_acceleration = false;  // Default to true
    
    Impl() : cpu_accelerator(std::make_unique<CPUAccelerator>()),
             gpu_accelerator(std::make_unique<GPUAccelerator>()) {}
};

Accelerator::Accelerator() : pimpl_(std::make_unique<Impl>()) {}
Accelerator::~Accelerator() = default;

bool Accelerator::initialize() {
    std::cout << "🚀 Initializing Unified Acceleration..." << std::endl;
    
    // Always initialize CPU accelerator
    if (!pimpl_->cpu_accelerator->initialize()) {
        std::cerr << "❌ Failed to initialize CPU accelerator" << std::endl;
        return false;
    }
    
    // Try to initialize GPU accelerator
    if (false == pimpl_->gpu_accelerator->initialize()) {
        std::cout << "⚠️ GPU Acceleration not available, using CPU only" << std::endl;
    }
    
    return true;
}

void Accelerator::set_use_gpu_acceleration(bool enable) {
    pimpl_->use_gpu_acceleration = enable;
}

bool Accelerator::should_use_gpu() const {
    return pimpl_->gpu_accelerator->is_available() && pimpl_->use_gpu_acceleration;
}

bool Accelerator::is_available() const {
    return pimpl_->cpu_accelerator->is_available() || pimpl_->gpu_accelerator->is_available();
}

std::string Accelerator::get_device_info() const {
    if (pimpl_->gpu_accelerator->is_available()) {
        return pimpl_->cpu_accelerator->get_device_info() + " + " + pimpl_->gpu_accelerator->get_device_info();
    }
    return pimpl_->cpu_accelerator->get_device_info();
}

//===================================================================
// Pre interpolate
//===================================================================

bool Accelerator::pre_interpolate(ImageBuffer& image_buffer, uint32_t filters, const char (&xtrans)[6][6], bool half_size) {
    if (should_use_gpu()) {
        // GPU accelerator needs to be updated for pre-interpolation
        // For now, fallback to CPU immediately
        std::cout << "⚠️  GPU not yet updated for pre-interpolation, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU pre-interpolation" << std::endl;
    if (pimpl_->cpu_accelerator->pre_interpolate(image_buffer, filters, xtrans, half_size)) {
        std::cout << "✅ Pre-interpolation completed successfully" << std::endl;
        return true;
    }
    std::cout << "❌ CPU Pre-interpolation failed" << std::endl;
    return false;
}

//===================================================================
// Demosaic 処理分岐
//===================================================================

// Unified demosaic computation with CPU/GPU selection
bool Accelerator::demosaic_compute(const ImageBuffer& raw_buffer,
                                    ImageBufferFloat& rgb_buffer,
                                    int algorithm,
                                    uint32_t filters,
                                    const char (&xtrans)[6][6],
                                    const float (&color_matrix)[3][4],
                                    const float (&cam_mul)[4],
                                    uint16_t maximum_value) {

    if (filters == FILTERS_XTRANS) {
        // xtrans data
        switch (algorithm) {
            case DemosaicAlgorithm::Linear: 
                std::cout << "🔧 Calling demosaic_xtrans_1pass (fast)..." << std::endl;
                return demosaic_xtrans_1pass(raw_buffer, rgb_buffer, xtrans, color_matrix, maximum_value);
            default:
                std::cout << "🔧 Calling demosaic_xtrans_3pass (high quality)..." << std::endl;
                return demosaic_xtrans_3pass(raw_buffer, rgb_buffer, xtrans, color_matrix, maximum_value);
        }
    } else {
    
        // Map algorithm to specific methods using enum constants
        switch (algorithm) {
            case static_cast<int>(DemosaicAlgorithm::Linear): 
                std::cout << "🔧 Calling demosaic_bayer_linear..." << std::endl;
                return demosaic_bayer_linear(raw_buffer, rgb_buffer, filters, maximum_value);
            case static_cast<int>(DemosaicAlgorithm::AAHD): 
                std::cout << "🔧 Calling demosaic_bayer_aahd..." << std::endl;
                return demosaic_bayer_aahd(raw_buffer, rgb_buffer, filters, maximum_value);
            case static_cast<int>(DemosaicAlgorithm::DCB): 
                std::cout << "🔧 Calling demosaic_bayer_dcb..." << std::endl;
                return demosaic_bayer_dcb(raw_buffer, rgb_buffer, filters, maximum_value);
            case static_cast<int>(DemosaicAlgorithm::AMaZE): 
                std::cout << "🔧 Calling demosaic_bayer_amaze..." << std::endl;
                return demosaic_bayer_amaze(raw_buffer, rgb_buffer, filters, cam_mul, maximum_value);
            default:
                std::cout << "⚠️ Unknown algorithm " << algorithm << ", using Linear" << std::endl;
                return demosaic_bayer_linear(raw_buffer, rgb_buffer, filters, maximum_value);
        }
    }
}

//===================================================================
// Demosaic Bayer
//===================================================================

// Processing methods with GPU/CPU selection
bool Accelerator::demosaic_bayer_linear(const ImageBuffer& raw_buffer,
                                           ImageBufferFloat& rgb_buffer,
                                           uint32_t filters,
                                           uint16_t maximum_value) {
    // GPU acceleration check
    if (should_use_gpu()) {
        std::cout << "🎯 Trying GPU Bayer Linear demosaic..." << std::endl;
        if (pimpl_->gpu_accelerator->demosaic_bayer_linear(raw_buffer, rgb_buffer, filters, maximum_value)) {
            std::cout << "✅ GPU Bayer Linear demosaic completed successfully" << std::endl;
            return true;
        } else {
            std::cout << "⚠️  GPU Bayer demosaic Linear failed, falling back to CPU" << std::endl;
        }
    }
    
    std::cout << "🔧 Using CPU Bayer Linear demosaic" << std::endl;
    if (pimpl_->cpu_accelerator->demosaic_bayer_linear(raw_buffer, rgb_buffer, filters, maximum_value)) {
        std::cout << "✅ CPU Linear demosaic completed successfully" << std::endl;
        return true;
    }
    std::cout << "❌ CPU Linear demosaic failed, falling back to CPU" << std::endl;
    return false;
}

bool Accelerator::demosaic_bayer_aahd(const ImageBuffer& raw_buffer,
                                        ImageBufferFloat& rgb_buffer,
                                        uint32_t filters,
                                        uint16_t maximum_value) {

    if (should_use_gpu()) {
        // GPU accelerator needs to be updated for demosaic_bayer_aahd
        // For now, fallback to CPU immediately
        std::cout << "⚠️  GPU not yet updated for demosaic_bayer_aahd, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU AAHD demosaic" << std::endl;
    if (pimpl_->cpu_accelerator->demosaic_bayer_aahd(raw_buffer, rgb_buffer, filters, maximum_value)) {
        std::cout << "✅ CPU AAHD demosaic completed successfully" << std::endl;
        return true;
    }
    std::cout << "❌ CPU AAHD demosaic failed, falling back to CPU" << std::endl;
    return false;
}

bool Accelerator::demosaic_bayer_dcb(const ImageBuffer& raw_buffer,
                                        ImageBufferFloat& rgb_buffer,
                                        uint32_t filters,
                                        uint16_t maximum_value,
                                        int iterations,
                                        bool dcb_enhance) {

    if (should_use_gpu()) {
        // GPU accelerator needs to be updated for demosaic_bayer_dcb
        // For now, fallback to CPU immediately
        std::cout << "⚠️  GPU not yet updated for demosaic_bayer_dcb, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU DCB demosaic" << std::endl;
    if (pimpl_->cpu_accelerator->demosaic_bayer_dcb(raw_buffer, rgb_buffer, filters, maximum_value, iterations, dcb_enhance)) {
        std::cout << "✅ CPU DCB demosaic completed successfully" << std::endl;
        return true;
    }
    std::cout << "❌ CPU DCB demosaic failed, falling back to CPU" << std::endl;
    return false;
}

bool Accelerator::demosaic_bayer_amaze(const ImageBuffer& raw_buffer,
                                          ImageBufferFloat& rgb_buffer,
                                          uint32_t filters,
                                          const float (&cam_mul)[4],
                                          uint16_t maximum_value) {
    //if (should_use_gpu()) {
    if (false) {
        std::cout << "🎯 Trying GPU AMaZE demosaic..." << std::endl;
        if (pimpl_->gpu_accelerator->demosaic_bayer_amaze(raw_buffer, rgb_buffer, filters, cam_mul, maximum_value)) {
            std::cout << "✅ GPU AMaZE demosaic completed successfully" << std::endl;
            return true;
        }
        std::cout << "⚠️ GPU AMaZE demosaic failed, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU AMaZE demosaic" << std::endl;
    if (pimpl_->cpu_accelerator->demosaic_bayer_amaze(raw_buffer, rgb_buffer, filters, cam_mul, maximum_value)) {
        std::cout << "✅ CPU AMaZE demosaic completed successfully" << std::endl;
        return true;
    }
    std::cout << "❌ CPU AMaZE demosaic failed, falling back to CPU" << std::endl;
    return false;
}

//===================================================================
// Demosaic Xtrans
//===================================================================

bool Accelerator::demosaic_xtrans_1pass(const ImageBuffer& raw_buffer,
                                        ImageBufferFloat& rgb_buffer,
                                        const char (&xtrans)[6][6],
                                        const float (&color_matrix)[3][4],
                                        uint16_t maximum_value) {
    //if (should_use_gpu()) {
    if (false) { // Temporarily disable GPU for testing
        std::cout << "🎯 Trying GPU X-Trans 1-pass demosaic..." << std::endl;
        if (pimpl_->gpu_accelerator->demosaic_xtrans_1pass(raw_buffer, rgb_buffer, xtrans, color_matrix, maximum_value)) {
            std::cout << "✅ GPU X-Trans 1-pass demosaic completed successfully" << std::endl;
            return true;
        }
        std::cout << "⚠️ GPU X-Trans 1-pass failed, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU X-Trans 1-pass demosaic" << std::endl;
    if (pimpl_->cpu_accelerator->demosaic_xtrans_1pass(raw_buffer, rgb_buffer, xtrans, color_matrix, maximum_value)) {
        std::cout << "✅ CPU X-Trans 1-pass demosaic completed successfully" << std::endl;
        return true;
    }
    std::cout << "❌ CPU X-Trans 1-pass demosaic failed, falling back to CPU" << std::endl;
    return false;
}

bool Accelerator::demosaic_xtrans_3pass(const ImageBuffer& raw_buffer,
                                        ImageBufferFloat& rgb_buffer,
                                        const char (&xtrans)[6][6],
                                        const float (&color_matrix)[3][4],
                                        uint16_t maximum_value) {
    //if (should_use_gpu()) {
    if (false) {
        std::cout << "🎯 Trying GPU X-Trans 3-pass demosaic..." << std::endl;
        if (pimpl_->gpu_accelerator->demosaic_xtrans_3pass(raw_buffer, rgb_buffer, xtrans, color_matrix, maximum_value)) {
            std::cout << "✅ GPU X-Trans 3-pass demosaic successful" << std::endl;
            return true;
        }
        std::cout << "⚠️ GPU X-Trans 3-pass demosaic failed, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU X-Trans 3-pass demosaic" << std::endl;
    if (pimpl_->cpu_accelerator->demosaic_xtrans_3pass(raw_buffer, rgb_buffer, xtrans, color_matrix, maximum_value)) {
        std::cout << "✅ CPU X-Trans 3-pass demosaic successful" << std::endl;
        return true;
    }
    std::cout << "❌ CPU X-Trans 3-pass demosaic failed, falling back to CPU" << std::endl;
    return false;
}

//===================================================================
// Apply white balance
//===================================================================

bool Accelerator::apply_white_balance(const ImageBufferFloat& rgb_input,
                                            ImageBufferFloat& rgb_output,
                                            const float wb_multipliers[4]) {
    if (should_use_gpu()) {
        // GPU accelerator needs to be updated to accept ImageBufferFloat
        // For now, fallback to CPU
        std::cout << "⚠️  GPU not yet updated for ImageBufferFloat, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU white balance on RAW" << std::endl;
    return pimpl_->cpu_accelerator->apply_white_balance(rgb_input, rgb_output, wb_multipliers);
}

//===================================================================
// Color space conversion methods
// Camera matrix-based color space conversion
//===================================================================

bool Accelerator::convert_color_space(const ImageBufferFloat& rgb_input,
                                      ImageBufferFloat& rgb_output,
                                      const float transform[3][4]) {
    if (should_use_gpu()) {
        std::cout << "🎯 Trying GPU convert color space..." << std::endl;
        if (pimpl_->gpu_accelerator->convert_color_space(rgb_input, rgb_output, transform)) {
            std::cout << "✅ GPU convert color space completed successfully" << std::endl;
            return true;
        }
        std::cout << "⚠️ GPU convert color space failed, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU convert color space" << std::endl;
    if (pimpl_->cpu_accelerator->convert_color_space(rgb_input, rgb_output, transform)) {
        std::cout << "✅ CPU convert color space completed successfully" << std::endl;
        return true;
    }
    std::cout << "❌ CPU convert color space failed, falling back to CPU" << std::endl;
    return false;
}

//===================================================================
// Gamma Correct
//===================================================================

bool Accelerator::gamma_correct(const ImageBufferFloat& rgb_input,
                               ImageBufferFloat& rgb_output,
                               float gamma_power,
                               float gamma_slope,
                               int output_color_space) {

    if (gamma_power == 1.f && gamma_slope == 1.f) {
        std::cout << "✅ Gamma power and slope are 1.0, skipping gamma correction" << std::endl;
        return true;
    } else
    if (output_color_space == ColorSpace::Raw || output_color_space == ColorSpace::XYZ || output_color_space == ColorSpace::ACEScg) {
        std::cout << "✅ Color space Raw or XYZ or ACEScg, skipping gamma correction" << std::endl;
        return true;
    } else
    if (gamma_power > 0.f && gamma_slope > 0.f) {
        std::cout << "ℹ️ Both gamma power and slope are set, custom gamma corrention" << std::endl;
        std::cout << "   Gamma power: " << gamma_power << " slope: " << gamma_slope << std::endl;
        output_color_space = -1;
    }

    if (should_use_gpu()) {
        std::cout << "🎯 Trying GPU gamma correct..." << std::endl;
        if (pimpl_->gpu_accelerator->gamma_correct(rgb_input, rgb_output, gamma_power, gamma_slope, output_color_space)) {
            std::cout << "✅ GPU gamma correct completed successfully" << std::endl;
            return true;
        }
        std::cout << "⚠️  GPU gamma correct failed, falling back to CPU" << std::endl;
    }
    
    std::cout << "🔧 Using CPU gamma correction" << std::endl;
    if (pimpl_->cpu_accelerator->gamma_correct(rgb_input, rgb_output, gamma_power, gamma_slope, output_color_space)) {
        std::cout << "✅ CPU gamma correct completed successfully" << std::endl;
        return true;
    }
    std::cout << "❌ CPU gamma correct failed, falling back to CPU" << std::endl;
    return false;
}


} // namespace libraw_enhanced