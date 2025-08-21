//
// gpu_accelerator.cpp
// LibRaw Enhanced - True GPU Metal Accelerator Implementation
//

#include "gpu_accelerator.h"
#include "accelerator.h"
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace libraw_enhanced {

// Global GPU state for Apple Silicon optimization
static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static id<MTLLibrary> g_shader_library = nil;
static bool g_gpu_initialized = false;

class GPUAccelerator::Impl {
public:
#ifdef __OBJC__
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;
    
    // Compute pipelines for each algorithm
    id<MTLComputePipelineState> bayer_linear_pipeline;
    id<MTLComputePipelineState> bayer_vng_pipeline;
    id<MTLComputePipelineState> bayer_ahd_pipeline;
    id<MTLComputePipelineState> bayer_dcb_pipeline;
    id<MTLComputePipelineState> bayer_amaze_pipeline;
    id<MTLComputePipelineState> xtrans_3pass_pipeline;
    id<MTLComputePipelineState> xtrans_1pass_pipeline;
    
    // Color space conversion pipelines
    id<MTLComputePipelineState> color_space_convert_pipeline;
    id<MTLComputePipelineState> camera_to_output_convert_pipeline;
    id<MTLComputePipelineState> white_balance_and_color_convert_pipeline;
    
    // LibRaw separated pipeline states
    id<MTLComputePipelineState> white_balance_raw_bayer_pipeline;
    id<MTLComputePipelineState> white_balance_raw_xtrans_pipeline;
    id<MTLComputePipelineState> libraw_convert_to_rgb_pipeline;
    id<MTLComputePipelineState> libraw_color_convert_matrix_selection_pipeline;
    
    // New separated kernel pipelines
    id<MTLComputePipelineState> apply_3x3_matrix_transform_pipeline;
    id<MTLComputePipelineState> apply_3x4_matrix_transform_pipeline;
    id<MTLComputePipelineState> multiply_matrices_3x3_3x4_pipeline;
    id<MTLComputePipelineState> gamma_correction_encode_pipeline;
    id<MTLComputePipelineState> gamma_correction_decode_pipeline;
    id<MTLComputePipelineState> gamma_correction_encode_inplace_pipeline;
    id<MTLComputePipelineState> select_color_space_matrix_pipeline;
    id<MTLComputePipelineState> compute_libraw_out_cam_matrix_pipeline;
    id<MTLComputePipelineState> apply_selected_color_space_matrix_pipeline;
    id<MTLComputePipelineState> libraw_complete_pipeline_rgb_only_pipeline;
    id<MTLComputePipelineState> libraw_color_conversion_step_pipeline;
    id<MTLComputePipelineState> libraw_gamma_correction_step_pipeline;
#endif
    
    bool initialized;
    std::string device_info;
    
    Impl() : initialized(false) {
#ifdef __OBJC__
        device = nil;
        command_queue = nil;
        library = nil;
        bayer_linear_pipeline = nil;
        bayer_vng_pipeline = nil;
        bayer_ahd_pipeline = nil;
        bayer_dcb_pipeline = nil;
        bayer_amaze_pipeline = nil;
        xtrans_3pass_pipeline = nil;
        xtrans_1pass_pipeline = nil;
        
        // Color space conversion pipelines
        color_space_convert_pipeline = nil;
        camera_to_output_convert_pipeline = nil;
        white_balance_and_color_convert_pipeline = nil;
        
        // LibRaw separated pipeline states
        white_balance_raw_bayer_pipeline = nil;
        white_balance_raw_xtrans_pipeline = nil;
        libraw_convert_to_rgb_pipeline = nil;
        libraw_color_convert_matrix_selection_pipeline = nil;
        
        // New separated kernel pipelines
        apply_3x3_matrix_transform_pipeline = nil;
        apply_3x4_matrix_transform_pipeline = nil;
        multiply_matrices_3x3_3x4_pipeline = nil;
        gamma_correction_encode_pipeline = nil;
        gamma_correction_decode_pipeline = nil;
        gamma_correction_encode_inplace_pipeline = nil;
        select_color_space_matrix_pipeline = nil;
        compute_libraw_out_cam_matrix_pipeline = nil;
        apply_selected_color_space_matrix_pipeline = nil;
        libraw_complete_pipeline_rgb_only_pipeline = nil;
        libraw_color_conversion_step_pipeline = nil;
        libraw_gamma_correction_step_pipeline = nil;
#endif
    }
};

GPUAccelerator::GPUAccelerator() : pimpl_(std::make_unique<Impl>()) {}
GPUAccelerator::~GPUAccelerator() = default;

bool GPUAccelerator::initialize() {
    // Check if already globally initialized for Apple Silicon optimization
    if (g_gpu_initialized) {
        std::cout << "♻️  Reusing global GPU Metal state..." << std::endl;
        pimpl_->device = g_metal_device;
        pimpl_->command_queue = g_command_queue;
        pimpl_->library = g_shader_library;
        
        // CRITICAL: Create compute pipelines for this instance
        if (!create_compute_pipelines()) {
            std::cerr << "❌ Failed to create compute pipelines for reused GPU state" << std::endl;
            return false;
        }
        
        pimpl_->initialized = true;
        pimpl_->device_info = std::string([pimpl_->device.name UTF8String]) + " (GPU)";
        return true;
    }
    
    std::cout << "🎯 Initializing True GPU Metal Acceleration..." << std::endl;
    
#ifdef METAL_ACCELERATION_AVAILABLE
#ifdef __OBJC__
    @autoreleasepool {
        // Create Metal device
        pimpl_->device = MTLCreateSystemDefaultDevice();
        if (!pimpl_->device) {
            std::cerr << "❌ No Metal device available" << std::endl;
            return false;
        }
        
        // Create command queue
        pimpl_->command_queue = [pimpl_->device newCommandQueue];
        if (!pimpl_->command_queue) {
            std::cerr << "❌ Failed to create Metal command queue" << std::endl;
            return false;
        }
        
        // Load and compile shaders
        if (!load_shaders()) {
            std::cerr << "❌ Failed to load Metal shaders" << std::endl;
            return false;
        }
        
        // Cache globally for Apple Silicon optimization
        g_metal_device = pimpl_->device;
        g_command_queue = pimpl_->command_queue;
        g_shader_library = pimpl_->library;
        g_gpu_initialized = true;
        
        pimpl_->device_info = std::string([pimpl_->device.name UTF8String]) + " (GPU)";
        pimpl_->initialized = true;
        
        std::cout << "✅ GPU Metal acceleration initialized: " << pimpl_->device_info << std::endl;
        return true;
    }
#else
    std::cerr << "❌ __OBJC__ is NOT defined - Metal requires Objective-C++ compilation" << std::endl;
    return false;
#endif
#else
    std::cerr << "❌ METAL_ACCELERATION_AVAILABLE is NOT defined" << std::endl;
    return false;
#endif
}

bool GPUAccelerator::load_shaders() {
#ifdef __OBJC__
    @autoreleasepool {
        NSError* error = nil;
        
        // Try to load compiled metallib first, then fallback to source compilation
        NSString* metallib_path = [[NSBundle mainBundle] pathForResource:@"libraw_enhanced_shaders" ofType:@"metallib"];
        
        if (metallib_path) {
            std::cout << "📦 Loading compiled Metal library..." << std::endl;
            pimpl_->library = [pimpl_->device newLibraryWithFile:metallib_path error:&error];
        }
        
        if (!pimpl_->library) {
            std::cout << "🔧 Compiling shaders from source..." << std::endl;
            
            // Load and compile each shader individually to avoid redefinition errors
            if (!compile_individual_shaders()) {
                std::cerr << "❌ Failed to compile individual shaders" << std::endl;
                return false;
            }
        }
        
        if (!pimpl_->library) {
            std::cerr << "❌ Failed to create Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        // Create compute pipelines for each algorithm
        if (!create_compute_pipelines()) {
            std::cerr << "❌ Failed to create compute pipelines" << std::endl;
            return false;
        }
        
        std::cout << "✅ All Metal shaders loaded successfully" << std::endl;
        return true;
    }
#else
    return false;
#endif
}

std::string GPUAccelerator::load_shader_file(const std::string& filename) {
    // Try multiple possible paths for shader files
    std::vector<std::string> possible_paths = {
        std::string("core/metal/") + filename,                    // Current directory
        std::string("../core/metal/") + filename,                 // Up one directory
        std::string("../../core/metal/") + filename,              // Up two directories
        std::string("/Users/uniuyuni/PythonProjects/libraw_enhanced/core/metal/") + filename  // Absolute path
    };
    
    for (const auto& full_path : possible_paths) {
        std::ifstream file(full_path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::cout << "📄 Loaded shader from: " << full_path << std::endl;
            return buffer.str();
        }
    }
    
    std::cerr << "❌ Could not find shader file: " << filename << std::endl;
    std::cerr << "  Tried paths:" << std::endl;
    for (const auto& path : possible_paths) {
        std::cerr << "    " << path << std::endl;
    }
    return "";
}

bool GPUAccelerator::compile_individual_shaders() {
#ifdef __OBJC__
    @autoreleasepool {
        NSError* error = nil;
        
        // Load common header once
        std::string common_header = load_shader_file("metal_common.h");
        if (common_header.empty()) {
            std::cerr << "❌ Could not load metal_common.h" << std::endl;
            return false;
        }
        
        // Load all shader sources but replace #include "metal_common.h" with actual content
        std::string combined = common_header + "\n\n";
        
        std::vector<std::string> shader_files = {
            "bayer_linear.metal",
            "bayer_vng.metal", 
            "bayer_ahd.metal",
            "bayer_dcb.metal",
            "bayer_amaze.metal",
            "xtrans_3pass.metal",
            "xtrans_1pass.metal",
            "white_balance_raw.metal",
            "matrix_transform.metal",
            "gamma_correction.metal", 
            "color_space_matrices.metal",
            "color_conversion.metal"
        };
        
        for (const auto& filename : shader_files) {
            std::string source = load_shader_file(filename);
            if (!source.empty()) {
                // Remove the #include "metal_common.h" line since we already added it
                size_t include_pos = source.find("#include \"metal_common.h\"");
                if (include_pos != std::string::npos) {
                    size_t end_pos = source.find("\n", include_pos);
                    if (end_pos != std::string::npos) {
                        source.erase(include_pos, end_pos - include_pos + 1);
                    } else {
                        // Handle case where include is at end of file
                        source.erase(include_pos);
                    }
                }
                
                // Also remove any remaining empty lines after the include
                while (source.find("\n\n") == 0) {
                    source.erase(0, 1);
                }
                
                combined += "// " + filename + "\n";
                combined += source;
                combined += "\n\n";
                std::cout << "📄 Added shader to combined source: " << filename << std::endl;
            }
        }
        
        NSString* nsSource = [NSString stringWithUTF8String:combined.c_str()];
        pimpl_->library = [pimpl_->device newLibraryWithSource:nsSource options:nil error:&error];
        
        if (!pimpl_->library) {
            std::cerr << "❌ Failed to create Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        std::cout << "✅ Successfully compiled Metal shaders with unified common header" << std::endl;
        return true;
    }
#else
    return false;
#endif
}

std::string GPUAccelerator::load_all_shader_sources() {
    std::string combined;
    
    std::vector<std::string> shader_files = {
        "bayer_linear.metal",
        "bayer_vng.metal", 
        "bayer_ahd.metal",
        "bayer_dcb.metal",
        "bayer_amaze.metal",
        "xtrans_3pass.metal",
        "xtrans_1pass.metal",
        "color_space_conversion.metal"
    };
    
    for (const auto& filename : shader_files) {
        std::string source = load_shader_file(filename);
        if (!source.empty()) {
            combined += "// " + filename + "\\n";
            combined += source;
            combined += "\\n\\n";
            std::cout << "📄 Loaded shader: " << filename << std::endl;
        }
    }
    
    return combined;
}

bool GPUAccelerator::create_compute_pipelines() {
#ifdef __OBJC__
    @autoreleasepool {
        NSError* error = nil;
        
        // Bayer Linear
        id<MTLFunction> bayer_linear_func = [pimpl_->library newFunctionWithName:@"bayer_linear_demosaic"];
        if (bayer_linear_func) {
            pimpl_->bayer_linear_pipeline = [pimpl_->device newComputePipelineStateWithFunction:bayer_linear_func error:&error];
            if (!pimpl_->bayer_linear_pipeline) {
                std::cerr << "❌ Failed to create Linear pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
                return false;
            }
        }
        
        // Bayer VNG
        id<MTLFunction> bayer_vng_func = [pimpl_->library newFunctionWithName:@"bayer_vng_demosaic"];
        if (bayer_vng_func) {
            pimpl_->bayer_vng_pipeline = [pimpl_->device newComputePipelineStateWithFunction:bayer_vng_func error:&error];
        }
        
        // Bayer AHD
        id<MTLFunction> bayer_ahd_func = [pimpl_->library newFunctionWithName:@"bayer_ahd_demosaic"];
        if (bayer_ahd_func) {
            pimpl_->bayer_ahd_pipeline = [pimpl_->device newComputePipelineStateWithFunction:bayer_ahd_func error:&error];
        }
        
        // Bayer DCB
        id<MTLFunction> bayer_dcb_func = [pimpl_->library newFunctionWithName:@"bayer_dcb_demosaic"];
        if (bayer_dcb_func) {
            pimpl_->bayer_dcb_pipeline = [pimpl_->device newComputePipelineStateWithFunction:bayer_dcb_func error:&error];
        }
        
        // Bayer AMaZE
        id<MTLFunction> bayer_amaze_func = [pimpl_->library newFunctionWithName:@"bayer_amaze_demosaic"];
        if (bayer_amaze_func) {
            pimpl_->bayer_amaze_pipeline = [pimpl_->device newComputePipelineStateWithFunction:bayer_amaze_func error:&error];
        }
        
        // X-Trans 3-pass
        id<MTLFunction> xtrans_3pass_func = [pimpl_->library newFunctionWithName:@"xtrans_3pass_demosaic"];
        if (xtrans_3pass_func) {
            pimpl_->xtrans_3pass_pipeline = [pimpl_->device newComputePipelineStateWithFunction:xtrans_3pass_func error:&error];
            if (!pimpl_->xtrans_3pass_pipeline) {
                std::cerr << "❌ Failed to create X-Trans 3-pass pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ X-Trans 3-pass pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ X-Trans 3-pass function 'xtrans_3pass_demosaic' not found in library" << std::endl;
        }
        
        // X-Trans 1-pass 
        id<MTLFunction> xtrans_1pass_func = [pimpl_->library newFunctionWithName:@"xtrans_1pass_demosaic"];
        if (xtrans_1pass_func) {
            pimpl_->xtrans_1pass_pipeline = [pimpl_->device newComputePipelineStateWithFunction:xtrans_1pass_func error:&error];
            if (!pimpl_->xtrans_1pass_pipeline) {
                std::cerr << "❌ Failed to create X-Trans 1-pass pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ X-Trans 1-pass pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ X-Trans 1-pass function 'xtrans_1pass_demosaic' not found in library" << std::endl;
        }
        
        // Color Space Conversion
        id<MTLFunction> color_space_convert_func = [pimpl_->library newFunctionWithName:@"color_space_convert"];
        if (color_space_convert_func) {
            pimpl_->color_space_convert_pipeline = [pimpl_->device newComputePipelineStateWithFunction:color_space_convert_func error:&error];
            if (!pimpl_->color_space_convert_pipeline) {
                std::cerr << "❌ Failed to create color space conversion pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ Color space conversion pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ Color space conversion function 'color_space_convert' not found in library" << std::endl;
        }
        
        // Camera to Output Conversion
        id<MTLFunction> camera_to_output_func = [pimpl_->library newFunctionWithName:@"camera_to_output_convert"];
        if (camera_to_output_func) {
            pimpl_->camera_to_output_convert_pipeline = [pimpl_->device newComputePipelineStateWithFunction:camera_to_output_func error:&error];
            if (!pimpl_->camera_to_output_convert_pipeline) {
                std::cerr << "❌ Failed to create camera-to-output conversion pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ Camera-to-output conversion pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ Camera-to-output conversion function 'camera_to_output_convert' not found in library" << std::endl;
        }
        
        // White Balance + Color Conversion
        id<MTLFunction> wb_color_convert_func = [pimpl_->library newFunctionWithName:@"white_balance_and_color_convert"];
        if (wb_color_convert_func) {
            pimpl_->white_balance_and_color_convert_pipeline = [pimpl_->device newComputePipelineStateWithFunction:wb_color_convert_func error:&error];
            if (!pimpl_->white_balance_and_color_convert_pipeline) {
                std::cerr << "❌ Failed to create white balance + color conversion pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ White balance + color conversion pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ White balance + color conversion function 'white_balance_and_color_convert' not found in library" << std::endl;
        }
        
        // LibRaw Separated Pipeline Functions
        
        // White Balance RAW Bayer
        id<MTLFunction> wb_raw_bayer_func = [pimpl_->library newFunctionWithName:@"apply_white_balance_bayer"];
        if (wb_raw_bayer_func) {
            pimpl_->white_balance_raw_bayer_pipeline = [pimpl_->device newComputePipelineStateWithFunction:wb_raw_bayer_func error:&error];
            if (!pimpl_->white_balance_raw_bayer_pipeline) {
                std::cerr << "❌ Failed to create white balance RAW Bayer pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ White balance RAW Bayer pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ White balance RAW Bayer function 'apply_white_balance_bayer' not found in library" << std::endl;
        }
        
        // White Balance RAW X-Trans
        id<MTLFunction> wb_raw_xtrans_func = [pimpl_->library newFunctionWithName:@"apply_white_balance_xtrans"];
        if (wb_raw_xtrans_func) {
            pimpl_->white_balance_raw_xtrans_pipeline = [pimpl_->device newComputePipelineStateWithFunction:wb_raw_xtrans_func error:&error];
            if (!pimpl_->white_balance_raw_xtrans_pipeline) {
                std::cerr << "❌ Failed to create white balance RAW X-Trans pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ White balance RAW X-Trans pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ White balance RAW X-Trans function 'apply_white_balance_xtrans' not found in library" << std::endl;
        }
        
        // LibRaw Convert to RGB
        id<MTLFunction> libraw_convert_func = [pimpl_->library newFunctionWithName:@"libraw_convert_to_rgb"];
        if (libraw_convert_func) {
            pimpl_->libraw_convert_to_rgb_pipeline = [pimpl_->device newComputePipelineStateWithFunction:libraw_convert_func error:&error];
            if (!pimpl_->libraw_convert_to_rgb_pipeline) {
                std::cerr << "❌ Failed to create LibRaw convert to RGB pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ LibRaw convert to RGB pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ LibRaw convert to RGB function 'libraw_convert_to_rgb' not found in library" << std::endl;
        }
        
        // LibRaw Color Convert with Matrix Selection
        id<MTLFunction> libraw_matrix_select_func = [pimpl_->library newFunctionWithName:@"libraw_color_convert_with_matrix_selection"];
        if (libraw_matrix_select_func) {
            pimpl_->libraw_color_convert_matrix_selection_pipeline = [pimpl_->device newComputePipelineStateWithFunction:libraw_matrix_select_func error:&error];
            if (!pimpl_->libraw_color_convert_matrix_selection_pipeline) {
                std::cerr << "❌ Failed to create LibRaw matrix selection pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
                std::cout << "✅ LibRaw matrix selection pipeline created successfully" << std::endl;
            }
        } else {
            std::cerr << "❌ LibRaw matrix selection function 'libraw_color_convert_with_matrix_selection' not found in library" << std::endl;
        }
        
        // New Separated Kernel Pipelines
        
        // Matrix Transform Kernels
        id<MTLFunction> matrix_3x3_func = [pimpl_->library newFunctionWithName:@"apply_3x3_matrix_transform"];
        if (matrix_3x3_func) {
            pimpl_->apply_3x3_matrix_transform_pipeline = [pimpl_->device newComputePipelineStateWithFunction:matrix_3x3_func error:&error];
            if (pimpl_->apply_3x3_matrix_transform_pipeline) {
                std::cout << "✅ 3x3 Matrix transform pipeline created successfully" << std::endl;
            }
        }
        
        id<MTLFunction> matrix_3x4_func = [pimpl_->library newFunctionWithName:@"apply_3x4_matrix_transform"];
        if (matrix_3x4_func) {
            pimpl_->apply_3x4_matrix_transform_pipeline = [pimpl_->device newComputePipelineStateWithFunction:matrix_3x4_func error:&error];
            if (pimpl_->apply_3x4_matrix_transform_pipeline) {
                std::cout << "✅ 3x4 Matrix transform pipeline created successfully" << std::endl;
            }
        }
        
        id<MTLFunction> multiply_matrices_func = [pimpl_->library newFunctionWithName:@"multiply_matrices_3x3_3x4"];
        if (multiply_matrices_func) {
            pimpl_->multiply_matrices_3x3_3x4_pipeline = [pimpl_->device newComputePipelineStateWithFunction:multiply_matrices_func error:&error];
            if (pimpl_->multiply_matrices_3x3_3x4_pipeline) {
                std::cout << "✅ Matrix multiplication pipeline created successfully" << std::endl;
            }
        }
        
        // Gamma Correction Kernels
        id<MTLFunction> gamma_encode_func = [pimpl_->library newFunctionWithName:@"apply_gamma_correction_encode"];
        if (gamma_encode_func) {
            pimpl_->gamma_correction_encode_pipeline = [pimpl_->device newComputePipelineStateWithFunction:gamma_encode_func error:&error];
            if (pimpl_->gamma_correction_encode_pipeline) {
                std::cout << "✅ Gamma correction encode pipeline created successfully" << std::endl;
            }
        }
        
        id<MTLFunction> gamma_decode_func = [pimpl_->library newFunctionWithName:@"apply_gamma_correction_decode"];
        if (gamma_decode_func) {
            pimpl_->gamma_correction_decode_pipeline = [pimpl_->device newComputePipelineStateWithFunction:gamma_decode_func error:&error];
            if (pimpl_->gamma_correction_decode_pipeline) {
                std::cout << "✅ Gamma correction decode pipeline created successfully" << std::endl;
            }
        }
        
        id<MTLFunction> gamma_inplace_func = [pimpl_->library newFunctionWithName:@"apply_gamma_correction_encode_inplace"];
        if (gamma_inplace_func) {
            pimpl_->gamma_correction_encode_inplace_pipeline = [pimpl_->device newComputePipelineStateWithFunction:gamma_inplace_func error:&error];
            if (pimpl_->gamma_correction_encode_inplace_pipeline) {
                std::cout << "✅ Gamma correction in-place pipeline created successfully" << std::endl;
            }
        }
        
        // Color Space Matrix Kernels
        id<MTLFunction> select_matrix_func = [pimpl_->library newFunctionWithName:@"select_color_space_matrix"];
        if (select_matrix_func) {
            pimpl_->select_color_space_matrix_pipeline = [pimpl_->device newComputePipelineStateWithFunction:select_matrix_func error:&error];
            if (pimpl_->select_color_space_matrix_pipeline) {
                std::cout << "✅ Color space matrix selection pipeline created successfully" << std::endl;
            }
        }
        
        id<MTLFunction> compute_out_cam_func = [pimpl_->library newFunctionWithName:@"compute_libraw_out_cam_matrix"];
        if (compute_out_cam_func) {
            pimpl_->compute_libraw_out_cam_matrix_pipeline = [pimpl_->device newComputePipelineStateWithFunction:compute_out_cam_func error:&error];
            if (pimpl_->compute_libraw_out_cam_matrix_pipeline) {
                std::cout << "✅ LibRaw out_cam matrix computation pipeline created successfully" << std::endl;
            }
        }
        
        id<MTLFunction> apply_color_space_func = [pimpl_->library newFunctionWithName:@"apply_selected_color_space_matrix"];
        if (apply_color_space_func) {
            pimpl_->apply_selected_color_space_matrix_pipeline = [pimpl_->device newComputePipelineStateWithFunction:apply_color_space_func error:&error];
            if (pimpl_->apply_selected_color_space_matrix_pipeline) {
                std::cout << "✅ Selected color space matrix application pipeline created successfully" << std::endl;
            }
        }
        
        // Pipeline Orchestrator Kernels
        id<MTLFunction> complete_pipeline_func = [pimpl_->library newFunctionWithName:@"libraw_complete_pipeline_rgb_only"];
        if (complete_pipeline_func) {
            pimpl_->libraw_complete_pipeline_rgb_only_pipeline = [pimpl_->device newComputePipelineStateWithFunction:complete_pipeline_func error:&error];
            if (pimpl_->libraw_complete_pipeline_rgb_only_pipeline) {
                std::cout << "✅ LibRaw complete RGB pipeline created successfully" << std::endl;
            }
        }
        
        id<MTLFunction> color_conversion_step_func = [pimpl_->library newFunctionWithName:@"libraw_color_conversion_step"];
        if (color_conversion_step_func) {
            pimpl_->libraw_color_conversion_step_pipeline = [pimpl_->device newComputePipelineStateWithFunction:color_conversion_step_func error:&error];
            if (pimpl_->libraw_color_conversion_step_pipeline) {
                std::cout << "✅ LibRaw color conversion step pipeline created successfully" << std::endl;
            }
        }
        
        id<MTLFunction> gamma_step_func = [pimpl_->library newFunctionWithName:@"libraw_gamma_correction_step"];
        if (gamma_step_func) {
            pimpl_->libraw_gamma_correction_step_pipeline = [pimpl_->device newComputePipelineStateWithFunction:gamma_step_func error:&error];
            if (pimpl_->libraw_gamma_correction_step_pipeline) {
                std::cout << "✅ LibRaw gamma correction step pipeline created successfully" << std::endl;
            }
        }
        
        int pipeline_count = 
            (pimpl_->bayer_linear_pipeline ? 1 : 0) +
            (pimpl_->bayer_vng_pipeline ? 1 : 0) +
            (pimpl_->bayer_ahd_pipeline ? 1 : 0) +
            (pimpl_->bayer_dcb_pipeline ? 1 : 0) +
            (pimpl_->bayer_amaze_pipeline ? 1 : 0) +
            (pimpl_->xtrans_3pass_pipeline ? 1 : 0) +
            (pimpl_->xtrans_1pass_pipeline ? 1 : 0) +
            (pimpl_->color_space_convert_pipeline ? 1 : 0) +
            (pimpl_->camera_to_output_convert_pipeline ? 1 : 0) +
            (pimpl_->white_balance_and_color_convert_pipeline ? 1 : 0) +
            (pimpl_->white_balance_raw_bayer_pipeline ? 1 : 0) +
            (pimpl_->white_balance_raw_xtrans_pipeline ? 1 : 0) +
            (pimpl_->libraw_convert_to_rgb_pipeline ? 1 : 0) +
            (pimpl_->libraw_color_convert_matrix_selection_pipeline ? 1 : 0) +
            (pimpl_->apply_3x3_matrix_transform_pipeline ? 1 : 0) +
            (pimpl_->apply_3x4_matrix_transform_pipeline ? 1 : 0) +
            (pimpl_->multiply_matrices_3x3_3x4_pipeline ? 1 : 0) +
            (pimpl_->gamma_correction_encode_pipeline ? 1 : 0) +
            (pimpl_->gamma_correction_decode_pipeline ? 1 : 0) +
            (pimpl_->gamma_correction_encode_inplace_pipeline ? 1 : 0) +
            (pimpl_->select_color_space_matrix_pipeline ? 1 : 0) +
            (pimpl_->compute_libraw_out_cam_matrix_pipeline ? 1 : 0) +
            (pimpl_->apply_selected_color_space_matrix_pipeline ? 1 : 0) +
            (pimpl_->libraw_complete_pipeline_rgb_only_pipeline ? 1 : 0) +
            (pimpl_->libraw_color_conversion_step_pipeline ? 1 : 0) +
            (pimpl_->libraw_gamma_correction_step_pipeline ? 1 : 0);
        
        std::cout << "✅ Created " << pipeline_count << " Metal compute pipelines" << std::endl;
        
        return true;
    }
#else
    return false;
#endif
}

bool GPUAccelerator::is_available() const {
    return pimpl_->initialized;
}

std::string GPUAccelerator::get_device_info() const {
    return pimpl_->device_info;
}

// Demosaicing implementations
bool GPUAccelerator::demosaic_bayer_linear(const ImageBuffer& raw_buffer,
                                         ImageBuffer& rgb_buffer,
                                         const ProcessingParams& params) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->bayer_linear_pipeline) {
        return false;
    }
    
    return execute_bayer_kernel(pimpl_->bayer_linear_pipeline, raw_buffer, rgb_buffer, params);
#else
    return false;
#endif
}

bool GPUAccelerator::demosaic_bayer_vng(const ImageBuffer& raw_buffer,
                                       ImageBuffer& rgb_buffer,
                                       const ProcessingParams& params) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->bayer_vng_pipeline) {
        return false;
    }
    
    return execute_bayer_kernel(pimpl_->bayer_vng_pipeline, raw_buffer, rgb_buffer, params);
#else
    return false;
#endif
}

bool GPUAccelerator::demosaic_bayer_ahd(const ImageBuffer& raw_buffer,
                                       ImageBuffer& rgb_buffer, 
                                       const ProcessingParams& params) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->bayer_ahd_pipeline) {
        return false;
    }
    
    return execute_bayer_kernel(pimpl_->bayer_ahd_pipeline, raw_buffer, rgb_buffer, params);
#else
    return false;
#endif
}

bool GPUAccelerator::demosaic_bayer_dcb(const ImageBuffer& raw_buffer,
                                       ImageBuffer& rgb_buffer,
                                       const ProcessingParams& params) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->bayer_dcb_pipeline) {
        return false;
    }
    
    return execute_bayer_kernel(pimpl_->bayer_dcb_pipeline, raw_buffer, rgb_buffer, params);
#else
    return false;
#endif
}

bool GPUAccelerator::demosaic_bayer_amaze(const ImageBuffer& raw_buffer,
                                         ImageBuffer& rgb_buffer,
                                         const ProcessingParams& params) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->bayer_amaze_pipeline) {
        return false;
    }
    
    return execute_bayer_kernel(pimpl_->bayer_amaze_pipeline, raw_buffer, rgb_buffer, params);
#else
    return false;
#endif
}

bool GPUAccelerator::demosaic_xtrans_3pass(const ImageBuffer& raw_buffer,
                                          ImageBuffer& rgb_buffer,
                                          const ProcessingParams& params) {
#ifdef __OBJC__
    if (!pimpl_->initialized) {
        std::cerr << "❌ GPU accelerator not initialized" << std::endl;
        return false;
    }
    
    if (!pimpl_->xtrans_3pass_pipeline) {
        std::cerr << "❌ X-Trans 3-pass pipeline not available" << std::endl;
        return false;
    }
    
    std::cout << "✅ X-Trans 3-pass pipeline ready, executing kernel..." << std::endl;
    return execute_xtrans_kernel(pimpl_->xtrans_3pass_pipeline, raw_buffer, rgb_buffer, params);
#else
    return false;
#endif
}

bool GPUAccelerator::demosaic_xtrans_1pass(const ImageBuffer& raw_buffer,
                                          ImageBuffer& rgb_buffer,
                                          const ProcessingParams& params) {
#ifdef __OBJC__
    if (!pimpl_->xtrans_1pass_pipeline) {
        std::cerr << "❌ X-Trans 1-pass pipeline not available" << std::endl;
        return false;
    }
    
    std::cout << "🎯 Using GPU X-Trans 1-pass demosaic" << std::endl;
    return execute_xtrans_kernel(pimpl_->xtrans_1pass_pipeline, raw_buffer, rgb_buffer, params);
#else
    return false;
#endif
}

#ifdef __OBJC__
bool GPUAccelerator::execute_bayer_kernel(id<MTLComputePipelineState> pipeline,
                                         const ImageBuffer& raw_buffer,
                                         ImageBuffer& rgb_buffer,
                                         const ProcessingParams& params) {
    @autoreleasepool {
        // Calculate buffer sizes
        size_t raw_size = raw_buffer.width * raw_buffer.height * raw_buffer.channels * 2;
        size_t rgb_size = rgb_buffer.width * rgb_buffer.height * rgb_buffer.channels * 2;
        
        // Create buffers - Use Apple Silicon unified memory optimization
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithBytes:raw_buffer.image
                                                                      length:raw_size
                                                                     options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        
        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithLength:rgb_size
                                                                     options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        
        // Create parameters buffer
        struct BayerParams {
            uint32_t width;
            uint32_t height;
            uint32_t filters;
            float bright;
            float gamma_power;
            uint32_t use_camera_wb;
        } bayer_params = {
            static_cast<uint32_t>(raw_buffer.width),
            static_cast<uint32_t>(raw_buffer.height),
            0x61616161u, // filters - Standard RGGB Bayer pattern for proper fcol() function
            params.bright,
            params.gamma_power, // Use actual gamma from parameters
            params.use_camera_wb ? 1u : 0u
        };
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&bayer_params
                                                                  length:sizeof(bayer_params)
                                                                 options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        
        // Create command buffer
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        // Calculate thread group size - Apple Silicon optimized
        MTLSize thread_group_size = MTLSizeMake(32, 32, 1);  // Optimized for unified memory
        MTLSize grid_size = MTLSizeMake(raw_buffer.width, raw_buffer.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Check for errors
        if (command_buffer.error) {
            NSLog(@"❌ Metal command buffer error: %@", command_buffer.error.localizedDescription);
            return false;
        }
        
        MTLCommandBufferStatus status = command_buffer.status;
        if (status != MTLCommandBufferStatusCompleted) {
            NSLog(@"❌ Metal command buffer status: %ld (expected: %ld)", (long)status, (long)MTLCommandBufferStatusCompleted);
            return false;
        }
        
        // Copy result back
        memcpy(rgb_buffer.image, rgb_metal_buffer.contents, rgb_size);
        
        return true;
    }
}

bool GPUAccelerator::execute_xtrans_kernel(id<MTLComputePipelineState> pipeline,
                                          const ImageBuffer& raw_buffer,
                                          ImageBuffer& rgb_buffer,
                                          const ProcessingParams& params) {
    @autoreleasepool {
        std::cout << "🔧 X-Trans kernel execution starting..." << std::endl;
        std::cout << "  Raw: " << raw_buffer.width << "x" << raw_buffer.height 
                  << " (" << raw_buffer.channels << " channels, 2 bytes/ch)" << std::endl;
        std::cout << "  RGB: " << rgb_buffer.width << "x" << rgb_buffer.height 
                  << " (" << rgb_buffer.channels << " channels, 2 bytes/ch)" << std::endl;
        // Calculate buffer sizes
        size_t raw_size = raw_buffer.width * raw_buffer.height * raw_buffer.channels * 2;
        size_t rgb_size = rgb_buffer.width * rgb_buffer.height * rgb_buffer.channels * 2;
        
        std::cout << "  Raw buffer size: " << raw_size << " bytes" << std::endl;
        std::cout << "  RGB buffer size: " << rgb_size << " bytes" << std::endl;
        
        // Debug: Check raw input data range
        if (raw_buffer.image && raw_size > 0) {
            uint16_t* raw_ptr = (uint16_t*)raw_buffer.image;
            uint16_t min_val = raw_ptr[0];
            uint16_t max_val = raw_ptr[0];
            size_t num_pixels = raw_size / sizeof(uint16_t);
            for (size_t i = 0; i < std::min(num_pixels, size_t(1000)); i++) {
                min_val = std::min(min_val, raw_ptr[i]);
                max_val = std::max(max_val, raw_ptr[i]);
            }
            std::cout << "  🔍 Raw input range: " << min_val << " - " << max_val << std::endl;
        }
        
        // Similar to Bayer but with X-Trans specific parameters - Use unified memory optimization
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithBytes:raw_buffer.image
                                                                      length:raw_size
                                                                     options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        
        if (!raw_metal_buffer) {
            std::cerr << "❌ Failed to create raw Metal buffer" << std::endl;
            return false;
        }
        
        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithLength:rgb_size
                                                                     options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        
        if (!rgb_metal_buffer) {
            std::cerr << "❌ Failed to create RGB Metal buffer" << std::endl;
            return false;
        }
        
        std::cout << "  ✅ Metal buffers created successfully" << std::endl;
        
        struct XTransParams {
            uint32_t width;
            uint32_t height;
            uint32_t xtrans[6][6];   // X-Trans pattern 6x6
            float bright;
            float gamma_power;
            uint32_t use_camera_wb;
        } xtrans_params = {
            static_cast<uint32_t>(raw_buffer.width),
            static_cast<uint32_t>(raw_buffer.height),
            {
                {0, 1, 0, 1, 2, 1},  // Standard Fujifilm X-Trans pattern
                {2, 0, 2, 0, 1, 0},
                {1, 2, 1, 2, 0, 2},
                {1, 0, 1, 0, 2, 0},
                {2, 1, 2, 1, 0, 1},
                {0, 2, 0, 2, 1, 2}
            },
            params.bright,
            2.222f, // gamma_power のデフォルト値
            params.use_camera_wb ? 1u : 0u
        };
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&xtrans_params
                                                                  length:sizeof(xtrans_params)
                                                                 options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        if (!command_buffer) {
            std::cerr << "❌ Failed to create command buffer" << std::endl;
            return false;
        }
        
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) {
            std::cerr << "❌ Failed to create compute command encoder" << std::endl;
            return false;
        }
        
        std::cout << "  ✅ Command buffer and encoder created" << std::endl;
        
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        MTLSize thread_group_size = MTLSizeMake(32, 32, 1);  // Apple Silicon optimized
        MTLSize grid_size = MTLSizeMake(raw_buffer.width, raw_buffer.height, 1);
        
        std::cout << "  Dispatching threads: grid=" << grid_size.width << "x" << grid_size.height 
                  << ", group=" << thread_group_size.width << "x" << thread_group_size.height << std::endl;
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        std::cout << "  ✅ Threads dispatched, committing..." << std::endl;
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        std::cout << "  ✅ Command buffer completed" << std::endl;
        
        // Check for errors
        if (command_buffer.error) {
            NSLog(@"❌ X-Trans Metal command buffer error: %@", command_buffer.error.localizedDescription);
            return false;
        }
        
        MTLCommandBufferStatus status = command_buffer.status;
        if (status != MTLCommandBufferStatusCompleted) {
            NSLog(@"❌ X-Trans Metal command buffer status: %ld (expected: %ld)", (long)status, (long)MTLCommandBufferStatusCompleted);
            return false;
        }
        
        memcpy(rgb_buffer.image, rgb_metal_buffer.contents, rgb_size);
        
        return true;
    }
}

// Color space conversion implementations - LibRaw compatible
bool GPUAccelerator::convert_color_space(const ImageBuffer& rgb_input,
                                        ImageBuffer& rgb_output,
                                        int output_color_space,
                                        const float rgb_cam[3][4],
                                        float gamma_power,
                                        float gamma_slope,
                                        bool apply_gamma) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->color_space_convert_pipeline) {
        return false;
    }
    
    @autoreleasepool {
        // Validate color space ID
        if (output_color_space < 0 || output_color_space > 7) {
            std::cerr << "❌ Invalid color space ID: " << output_color_space << " (valid range: 0-7)" << std::endl;
            return false;
        }
        
        std::cout << "🎯 GPU Color Space Conversion: " << output_color_space 
                  << " (gamma=" << gamma_power << ", apply=" << apply_gamma << ")" << std::endl;
        
        // Create buffers
        size_t rgb_size = rgb_input.width * rgb_input.height * 3 * sizeof(uint16_t);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:rgb_input.image
                                                               length:rgb_size
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:rgb_size
                                                                 options:MTLResourceStorageModeShared];
        
        if (!input_buffer || !output_buffer) {
            std::cerr << "❌ Failed to create Metal buffers for color space conversion" << std::endl;
            return false;
        }
        
        // Create LibRaw-compatible parameters buffer  
        struct {
            float out_cam[3][4];     // Combined transformation matrix
            uint32_t output_width;
            uint32_t output_height;
            uint32_t color_space_id;
            float gamma_power;
            float gamma_slope;
            uint32_t apply_gamma;
            uint32_t padding;
        } params = {};
        
        params.output_width = rgb_input.width;
        params.output_height = rgb_input.height;
        params.color_space_id = output_color_space;
        params.gamma_power = gamma_power;
        params.gamma_slope = gamma_slope;
        params.apply_gamma = apply_gamma ? 1 : 0;
        
        // Copy rgb_cam matrix to out_cam (LibRaw compatible format)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                if (j < 3) {
                    params.out_cam[i][j] = rgb_cam[i][j];
                } else {
                    params.out_cam[i][j] = 0.0f; // 4th column not used for 3-channel RGB
                }
            }
        }
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                               length:sizeof(params)
                                                              options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->camera_to_output_convert_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        // Dispatch threads
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Check for errors
        if (command_buffer.error) {
            NSLog(@"❌ Color space conversion error: %@", command_buffer.error.localizedDescription);
            return false;
        }
        
        // Copy result back
        memcpy(rgb_output.image, output_buffer.contents, rgb_size);
        
        std::cout << "✅ GPU Color Space Conversion completed" << std::endl;
        return true;
    }
#else
    return false;
#endif
}

bool GPUAccelerator::apply_white_balance_and_color_conversion(const ImageBuffer& rgb_input,
                                                            ImageBuffer& rgb_output,
                                                            const float wb_multipliers[4],
                                                            int output_color_space,
                                                            float gamma_power,
                                                            float gamma_slope,
                                                            bool apply_gamma) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->white_balance_and_color_convert_pipeline) {
        return false;
    }
    
    @autoreleasepool {
        // Validate color space ID
        if (output_color_space < 0 || output_color_space > 7) {
            std::cerr << "❌ Invalid color space ID: " << output_color_space << std::endl;
            return false;
        }
        
        std::cout << "🎯 GPU White Balance + Color Conversion: " << output_color_space << std::endl;
        
        // Create buffers
        size_t rgb_size = rgb_input.width * rgb_input.height * 3 * sizeof(uint16_t);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:rgb_input.image
                                                               length:rgb_size
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:rgb_size
                                                                 options:MTLResourceStorageModeShared];
        
        // Create WB multipliers buffer
        float wb_params[4] = {wb_multipliers[0], wb_multipliers[1], wb_multipliers[2], wb_multipliers[3]};
        id<MTLBuffer> wb_buffer = [pimpl_->device newBufferWithBytes:&wb_params
                                                            length:sizeof(wb_params)
                                                           options:MTLResourceStorageModeShared];
        
        // Create parameters buffer
        struct {
            float matrix[3][3];
            uint32_t output_width;
            uint32_t output_height;
            uint32_t color_space_id;
            float gamma_power;
            float gamma_slope;
            uint32_t apply_gamma;
            uint32_t padding;
        } params = {};
        
        params.output_width = rgb_input.width;
        params.output_height = rgb_input.height;
        params.color_space_id = output_color_space;
        params.gamma_power = gamma_power;
        params.gamma_slope = gamma_slope;
        params.apply_gamma = apply_gamma ? 1 : 0;
        
        // Matrix selection happens in shader
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                params.matrix[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                               length:sizeof(params)
                                                              options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->white_balance_and_color_convert_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:wb_buffer offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        
        // Dispatch threads
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Check for errors
        if (command_buffer.error) {
            NSLog(@"❌ White balance + color conversion error: %@", command_buffer.error.localizedDescription);
            return false;
        }
        
        // Copy result back
        memcpy(rgb_output.image, output_buffer.contents, rgb_size);
        
        std::cout << "✅ GPU White Balance + Color Conversion completed" << std::endl;
        return true;
    }
#else
    return false;
#endif
}

// =====================================================================
// LibRaw Separated Pipeline Implementation
// =====================================================================

// Step 1: White balance for RAW Bayer data
bool GPUAccelerator::apply_white_balance_raw_bayer(const ImageBuffer& raw_input,
                                                  ImageBuffer& raw_output,
                                                  const float scale_mul[4],
                                                  const float pre_mul[4],
                                                  uint32_t filters,
                                                  float bright) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->white_balance_raw_bayer_pipeline) {
        return false;
    }
    
    @autoreleasepool {
        std::cout << "🎯 GPU White Balance RAW Bayer" << std::endl;
        
        // Create buffers
        size_t raw_size = raw_input.width * raw_input.height * 2;  // ushort = 2 bytes
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:raw_input.image
                                                               length:raw_size
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:raw_size
                                                                 options:MTLResourceStorageModeShared];
        
        // Create parameters
        struct {
            float pre_mul[4];
            float scale_mul[4];
            uint32_t width;
            uint32_t height;
            uint32_t filters;
            float bright;
            uint32_t use_camera_wb;
            uint32_t padding[2];
        } params = {};
        
        for (int i = 0; i < 4; i++) {
            params.pre_mul[i] = pre_mul[i];
            params.scale_mul[i] = scale_mul[i];
        }
        params.width = raw_input.width;
        params.height = raw_input.height;
        params.filters = filters;
        params.bright = bright;
        params.use_camera_wb = 1;
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                               length:sizeof(params)
                                                              options:MTLResourceStorageModeShared];
        
        // Execute kernel
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->white_balance_raw_bayer_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
        MTLSize grid_size = MTLSizeMake(raw_input.width, raw_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.error) {
            NSLog(@"❌ White balance RAW error: %@", command_buffer.error.localizedDescription);
            return false;
        }
        
        // Copy result
        memcpy(raw_output.image, output_buffer.contents, raw_size);
        
        std::cout << "✅ GPU White Balance RAW Bayer completed" << std::endl;
        return true;
    }
#else
    return false;
#endif
}

// Step 1: White balance for RAW X-Trans data
bool GPUAccelerator::apply_white_balance_raw_xtrans(const ImageBuffer& raw_input,
                                                   ImageBuffer& raw_output,
                                                   const float scale_mul[4],
                                                   const float pre_mul[4],
                                                   float bright) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->white_balance_raw_xtrans_pipeline) {
        return false;
    }
    
    @autoreleasepool {
        std::cout << "🎯 GPU White Balance RAW X-Trans" << std::endl;
        
        size_t raw_size = raw_input.width * raw_input.height * 2;  // ushort = 2 bytes
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:raw_input.image
                                                               length:raw_size
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:raw_size
                                                                 options:MTLResourceStorageModeShared];
        
        struct {
            float pre_mul[4];
            float scale_mul[4];
            uint32_t width;
            uint32_t height;
            uint32_t filters;
            float bright;
            uint32_t use_camera_wb;
            uint32_t padding[2];
        } params = {};
        
        for (int i = 0; i < 4; i++) {
            params.pre_mul[i] = pre_mul[i];
            params.scale_mul[i] = scale_mul[i];
        }
        params.width = raw_input.width;
        params.height = raw_input.height;
        params.filters = 0; // X-Trans doesn't use simple filters
        params.bright = bright;
        params.use_camera_wb = 1;
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                               length:sizeof(params)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->white_balance_raw_xtrans_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
        MTLSize grid_size = MTLSizeMake(raw_input.width, raw_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.error) {
            NSLog(@"❌ White balance X-Trans RAW error: %@", command_buffer.error.localizedDescription);
            return false;
        }
        
        memcpy(raw_output.image, output_buffer.contents, raw_size);
        
        std::cout << "✅ GPU White Balance RAW X-Trans completed" << std::endl;
        return true;
    }
#else
    return false;
#endif
}

// Step 3: LibRaw convert_to_rgb
bool GPUAccelerator::libraw_convert_to_rgb(const ImageBuffer& rgb_input,
                                          ImageBuffer& rgb_output,
                                          const float out_cam[3][4],
                                          int output_color,
                                          bool raw_color,
                                          float gamma_power,
                                          float gamma_slope,
                                          bool apply_gamma) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->libraw_convert_to_rgb_pipeline) {
        return false;
    }
    
    @autoreleasepool {
        std::cout << "🎯 GPU LibRaw Convert to RGB: color=" << output_color 
                  << " raw=" << raw_color << " gamma=" << apply_gamma << std::endl;
        
        size_t rgb_size = rgb_input.width * rgb_input.height * 3 * sizeof(uint16_t);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:rgb_input.image
                                                               length:rgb_size
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:rgb_size
                                                                 options:MTLResourceStorageModeShared];
        
        // LibRaw compatible parameters
        struct {
            float out_cam[3][4];
            uint32_t width;
            uint32_t height;
            uint32_t output_color;
            float gamma_power;
            float gamma_slope;
            uint32_t apply_gamma;
            uint32_t raw_color;
            uint32_t padding[2];
        } params = {};
        
        // Copy out_cam matrix
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                params.out_cam[i][j] = out_cam[i][j];
            }
        }
        
        params.width = rgb_input.width;
        params.height = rgb_input.height;
        params.output_color = output_color;
        params.gamma_power = gamma_power;
        params.gamma_slope = gamma_slope;
        params.apply_gamma = apply_gamma ? 1 : 0;
        params.raw_color = raw_color ? 1 : 0;
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                               length:sizeof(params)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->libraw_convert_to_rgb_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.error) {
            NSLog(@"❌ LibRaw convert to RGB error: %@", command_buffer.error.localizedDescription);
            return false;
        }
        
        memcpy(rgb_output.image, output_buffer.contents, rgb_size);
        
        std::cout << "✅ GPU LibRaw Convert to RGB completed" << std::endl;
        return true;
    }
#else
    return false;
#endif
}

// Alternative: Color conversion with matrix selection
bool GPUAccelerator::libraw_color_convert_with_matrix_selection(const ImageBuffer& rgb_input,
                                                               ImageBuffer& rgb_output,
                                                               const float rgb_cam[3][4],
                                                               int output_color,
                                                               bool raw_color,
                                                               float gamma_power,
                                                               float gamma_slope,
                                                               bool apply_gamma) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->libraw_color_convert_matrix_selection_pipeline) {
        return false;
    }
    
    @autoreleasepool {
        std::cout << "🎯 GPU LibRaw Matrix Selection Convert: color=" << output_color << std::endl;
        
        size_t rgb_size = rgb_input.width * rgb_input.height * 3 * sizeof(uint16_t);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:rgb_input.image
                                                               length:rgb_size
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:rgb_size
                                                                 options:MTLResourceStorageModeShared];
        
        // rgb_cam matrix buffer
        id<MTLBuffer> rgb_cam_buffer = [pimpl_->device newBufferWithBytes:rgb_cam
                                                                 length:sizeof(float) * 3 * 4
                                                                options:MTLResourceStorageModeShared];
        
        struct {
            float out_cam[3][4]; // Not used in this version
            uint32_t width;
            uint32_t height;
            uint32_t output_color;
            float gamma_power;
            float gamma_slope;
            uint32_t apply_gamma;
            uint32_t raw_color;
            uint32_t padding[2];
        } params = {};
        
        params.width = rgb_input.width;
        params.height = rgb_input.height;
        params.output_color = output_color;
        params.gamma_power = gamma_power;
        params.gamma_slope = gamma_slope;
        params.apply_gamma = apply_gamma ? 1 : 0;
        params.raw_color = raw_color ? 1 : 0;
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                               length:sizeof(params)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->libraw_color_convert_matrix_selection_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:rgb_cam_buffer offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.error) {
            NSLog(@"❌ LibRaw matrix selection convert error: %@", command_buffer.error.localizedDescription);
            return false;
        }
        
        memcpy(rgb_output.image, output_buffer.contents, rgb_size);
        
        std::cout << "✅ GPU LibRaw Matrix Selection Convert completed" << std::endl;
        return true;
    }
#else
    return false;
#endif
}

bool GPUAccelerator::apply_white_balance_to_raw(const ImageBuffer& raw_buffer,
                                               const float wb_multipliers[4],
                                               unsigned int filters) {
#ifdef METAL_ACCELERATION_AVAILABLE
    if (!pimpl_ || !pimpl_->device || !pimpl_->white_balance_raw_bayer_pipeline) {
        std::cout << "⚠️  Metal White Balance RAW not available, falling back to CPU" << std::endl;
        return false;
    }
    
    std::cout << "🎯 Using GPU White Balance on RAW data" << std::endl;
    
    id<MTLBuffer> raw_gpu_buffer = [pimpl_->device newBufferWithBytes:raw_buffer.image
                                                       length:raw_buffer.width * raw_buffer.height * 2
                                                      options:MTLResourceStorageModeShared];
    
    // Create parameters buffer
    struct WhiteBalanceRawParams {
        float pre_mul[4];
        float scale_mul[4];
        uint32_t width;
        uint32_t height;
        uint32_t filters;
        float bright;
        uint32_t use_camera_wb;
        uint32_t padding;
    } params;
    
    // Copy white balance multipliers
    for (int i = 0; i < 4; i++) {
        params.scale_mul[i] = wb_multipliers[i];
        params.pre_mul[i] = wb_multipliers[i];
    }
    params.width = raw_buffer.width;
    params.height = raw_buffer.height;
    params.filters = filters;
    params.bright = 1.0f;
    params.use_camera_wb = 1;
    params.padding = 0;
    
    id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                       length:sizeof(params)
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pimpl_->white_balance_raw_bayer_pipeline];
    [encoder setBuffer:raw_gpu_buffer offset:0 atIndex:0];  // Input
    [encoder setBuffer:raw_gpu_buffer offset:0 atIndex:1];  // Output (in-place)
    [encoder setBuffer:params_buffer offset:0 atIndex:2];
    
    MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
    MTLSize grid_size = MTLSizeMake(raw_buffer.width, raw_buffer.height, 1);
    
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
    [encoder endEncoding];
    
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    
    if (command_buffer.error) {
        NSLog(@"❌ GPU White Balance RAW error: %@", command_buffer.error.localizedDescription);
        return false;
    }
    
    // Copy result back (in-place modification, so raw_buffer.image gets the result)
    memcpy(raw_buffer.image, raw_gpu_buffer.contents, raw_buffer.width * raw_buffer.height * 2);
    
    std::cout << "✅ GPU White balance applied to RAW data" << std::endl;
    return true;
#else
    return false;
#endif
}
#endif

} // namespace libraw_enhanced