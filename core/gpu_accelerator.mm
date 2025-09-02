//
// gpu_accelerator.mm
// LibRaw Enhanced - True GPU Metal Accelerator Implementation
//

#include "gpu_accelerator.h"
#include "accelerator.h"
#include "metal/shader_types.h"
#include "metal/shader_common.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <simd/simd.h>
using namespace simd;
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
    
    // Bayer Demosaic Pipelines
    id<MTLComputePipelineState> bayer_linear_pipeline;
    id<MTLComputePipelineState> bayer_amaze_pipeline;
    id<MTLComputePipelineState> bayer_border_pipeline;

    // X-Trans Pipelines
    id<MTLComputePipelineState> xtrans_1pass_pipeline;
    id<MTLComputePipelineState> xtrans_3pass_pipeline;
    id<MTLComputePipelineState> xtrans_border_pipeline;
    
    // ImageBufferFloat processing pipelines
    id<MTLComputePipelineState> apply_white_balance_float_pipeline;
    id<MTLComputePipelineState> convert_color_space_float_pipeline;
    id<MTLComputePipelineState> gamma_correct_float_pipeline;
    
#endif
    
    bool initialized;
    std::string device_info;
    
    Impl() : initialized(false) {
#ifdef __OBJC__
        // Initialize all pipeline states to nil
        bayer_linear_pipeline = nil;
        bayer_amaze_pipeline = nil;
        bayer_border_pipeline = nil;
        xtrans_1pass_pipeline = nil;
        xtrans_3pass_pipeline = nil;
        xtrans_border_pipeline = nil;
        apply_white_balance_float_pipeline = nil;
        convert_color_space_float_pipeline = nil;
        gamma_correct_float_pipeline = nil;
#endif
    }
};

GPUAccelerator::GPUAccelerator() : pimpl_(std::make_unique<Impl>()) {}
GPUAccelerator::~GPUAccelerator() = default;

bool GPUAccelerator::initialize() {
    
#ifdef __OBJC__
    @autoreleasepool {
        std::cout << "[DEBUG] Initializing GPU..." << std::endl;
        pimpl_->device = MTLCreateSystemDefaultDevice();
        if (!pimpl_->device) {
            // このログが出れば、Metal自体が使えない環境
            std::cout << "[DEBUG] FAILED: MTLCreateSystemDefaultDevice() returned nil." << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Metal device created: " << [[pimpl_->device name] UTF8String] << std::endl;
        
        pimpl_->command_queue = [pimpl_->device newCommandQueue];
        if (!pimpl_->command_queue) {
            std::cout << "[DEBUG] FAILED: newCommandQueue() returned nil." << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Command queue created." << std::endl;
        
        if (!load_shaders()) {
            // このログが出れば、シェーダーの読み込みかコンパイルに失敗している
            std::cout << "[DEBUG] FAILED: load_shaders() returned false." << std::endl;
            return false;
        }
        
        g_metal_device = pimpl_->device;
        // ...
        std::cout << "[DEBUG] GPU Initialization SUCCESSFUL." << std::endl;
        return true;
    }
#else
    return false;
#endif
}

bool GPUAccelerator::load_shaders() {
#ifdef __OBJC__
    @autoreleasepool {
        // バンドル内のデフォルトライブラリを取得する
        pimpl_->library = [pimpl_->device newDefaultLibrary];
        
        if (!pimpl_->library) {
            std::cout << "[DEBUG] .metallib not found or failed to load. Trying to compile from source..." << std::endl;
            if (!compile_individual_shaders()) {
                // このログが出れば、ソースからのコンパイルに失敗
                std::cout << "[DEBUG] FAILED: compile_individual_shaders() returned false." << std::endl;
                return false;
            }
        }
        
        if (!pimpl_->library) {
            // このログが出れば、最終的にライブラリが作成できなかった
            std::cout << "[DEBUG] FAILED: Shader library is still nil after all attempts." << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Shader library loaded/compiled successfully." << std::endl;
        
        std::cout << "[DEBUG] Creating compute pipelines..." << std::endl;
        if (!create_compute_pipelines()) {
            std::cout << "[DEBUG] ❌ create_compute_pipelines() FAILED" << std::endl;
            return false;
        }
        std::cout << "[DEBUG] ✅ create_compute_pipelines() SUCCESS" << std::endl;
        
        pimpl_->initialized = true;
        pimpl_->device_info = std::string([pimpl_->device.name UTF8String]) + " (GPU)";
        
        return true;
    }
#else
    return false;
#endif
}

std::string GPUAccelerator::load_shader_file(const std::string& filename) {
    std::vector<std::string> possible_paths = {
        std::string("core/metal/") + filename,
        std::string("../core/metal/") + filename,
        std::string("../../core/metal/") + filename,
    };
    
    for (const auto& full_path : possible_paths) {
        std::ifstream file(full_path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }
    }
    
    return "";
}


bool GPUAccelerator::compile_individual_shaders() {
#ifdef __OBJC__
    @autoreleasepool {
        NSError* error = nil;
        
        // シェーダーファイルを読み込んで結合
        std::string combined_source;
        
        // ヘッダーファイルを最初に追加
        std::string types_header = load_shader_file("shader_types.h");
        std::string common_header = load_shader_file("shader_common.h");
        
        if (types_header.empty() || common_header.empty()) {
            std::cout << "[DEBUG] Failed to load shader headers." << std::endl;
            return false;
        }
        
        combined_source += types_header + "\n";
        combined_source += common_header + "\n";
        
        // シェーダーファイルを追加
        std::vector<std::string> shader_files = {
            "demosaic_bayer_linear.metal",
            "demosaic_bayer_amaze.metal",
            "demosaic_bayer_border.metal",
            "demosaic_xtrans_1pass.metal",
            "demosaic_xtrans_3pass.metal",
            "demosaic_xtrans_border.metal",
            "apply_white_balance.metal",
            "convert_color_space.metal",
            "gamma_correction.metal"
        };
        
        for (const auto& filename : shader_files) {
            std::string source = load_shader_file(filename);
            if (source.empty()) {
                std::cout << "[DEBUG] Failed to load shader: " << filename << std::endl;
                continue;
            }
            
            // インクルード文を削除
            size_t pos;
            while ((pos = source.find("#include \"shader_types.h\"")) != std::string::npos) {
                source.erase(pos, strlen("#include \"shader_types.h\""));
            }
            while ((pos = source.find("#include \"shader_common.h\"")) != std::string::npos) {
                source.erase(pos, strlen("#include \"shader_common.h\""));
            }
            
            combined_source += "// --- " + filename + " ---\n";
            combined_source += source + "\n\n";
        }
        
        // メタルソースをコンパイル
        NSString* nsSource = [NSString stringWithUTF8String:combined_source.c_str()];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_3;
        
        pimpl_->library = [pimpl_->device newLibraryWithSource:nsSource
                                                      options:options
                                                        error:&error];
        
        if (error) {
            std::cout << "[DEBUG] Metal shader compilation failed: " 
                      << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        if (!pimpl_->library) {
            std::cout << "[DEBUG] Failed to create Metal library." << std::endl;
            return false;
        }
        
        return true;
    }
#else
    return false;
#endif
}

bool GPUAccelerator::create_compute_pipelines() {
#ifdef __OBJC__
    @autoreleasepool {
        NSError* error = nil;
        
        auto createPipeline = [&](NSString* functionName) -> id<MTLComputePipelineState> {
            id<MTLFunction> function = [pimpl_->library newFunctionWithName:functionName];
            if (!function) {
                // どの関数が見つからなかったかを出力
                std::cerr << "Failed to find Metal function: " << [functionName UTF8String] << std::endl;
                return nil;
            }
            id<MTLComputePipelineState> pipeline = [pimpl_->device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) {
                // パイプライン作成時にエラーが発生した場合、その内容を出力
                std::cerr << "Failed to create pipeline state for function: " << [functionName UTF8String]
                        << ", error: " << [[error localizedDescription] UTF8String] << std::endl;
            }
            return pipeline;
        };

        // Bayer Pipelines
        pimpl_->bayer_linear_pipeline = createPipeline(@"demosaic_bayer_linear");
        pimpl_->bayer_amaze_pipeline = createPipeline(@"demosaic_bayer_amaze");
        pimpl_->bayer_border_pipeline = createPipeline(@"demosaic_bayer_border");

        // X-Trans Pipelines
        pimpl_->xtrans_1pass_pipeline = createPipeline(@"demosaic_xtrans_1pass");
        pimpl_->xtrans_3pass_pipeline = createPipeline(@"demosaic_xtrans_3pass");
        pimpl_->xtrans_border_pipeline = createPipeline(@"demosaic_xtrans_border");

        // Post-processing Pipelines
        pimpl_->apply_white_balance_float_pipeline = createPipeline(@"apply_white_balance");
        pimpl_->convert_color_space_float_pipeline = createPipeline(@"convert_color_space");
        pimpl_->gamma_correct_float_pipeline = createPipeline(@"gamma_correct");

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

// --- Demosaic Implementations ---

//===================================================================
// Demosaic Bayer Liner
//===================================================================

bool GPUAccelerator::demosaic_bayer_linear(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, uint16_t maximum_value) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->bayer_linear_pipeline) return false;
    
    @autoreleasepool {
        const auto width = raw_buffer.width, height = raw_buffer.height, pixel_count = width * height;
        
        // Prepare raw data
        std::vector<uint16_t> raw_gpu_data(pixel_count);
        for (size_t i = 0; i < pixel_count; i++) {
            size_t r = i / width, c = i % width;
            uint32_t color_idx = fcol_bayer(r, c, filters);
            raw_gpu_data[i] = raw_buffer.image[i][color_idx];
        }

        // Create Metal buffers
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithBytes:raw_gpu_data.data() 
                                                            length:pixel_count * sizeof(uint16_t) 
                                                            options:MTLResourceStorageModeShared];
        
        //id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithLength:pixel_count * 3 * sizeof(float)
        //                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0] 
                                                            length:pixel_count * 3 * sizeof(float) 
                                                            options:MTLResourceStorageModeShared 
                                                            deallocator:nil];
        
        BayerParams params = {
            (uint32_t)width,
            (uint32_t)height,
            1,
            (float)maximum_value,
            filters,
        };
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params 
                                                            length:sizeof(params) 
                                                            options:MTLResourceStorageModeShared];
        
        // Execute pipeline
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        MTLSize grid_size = MTLSizeMake(width, height, 1);
        MTLSize threadgroup_size = MTLSizeMake(16, 16, 1);

        [encoder setComputePipelineState:pimpl_->bayer_linear_pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:pimpl_->bayer_border_pipeline];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:params_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;
        
        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Demosaic Bayer AMaZE
//===================================================================

bool GPUAccelerator::demosaic_bayer_amaze(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, const float (&cam_mul)[4], uint16_t maximum_value) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->bayer_amaze_pipeline) return false;
    
    @autoreleasepool {
        const auto width = raw_buffer.width, height = raw_buffer.height, pixel_count = width * height;
        
        std::vector<ushort4> raw_rgbg_data(pixel_count);
        for (size_t i = 0; i < pixel_count; i++) {
            raw_rgbg_data[i] = { raw_buffer.image[i][0], raw_buffer.image[i][1], raw_buffer.image[i][2], raw_buffer.image[i][3] };
        }

        const size_t AMAZE_TS_CONST = 160;
        const size_t TILE_PIXELS = AMAZE_TS_CONST * AMAZE_TS_CONST;
        const size_t TILE_PIXELS_HALF = AMAZE_TS_CONST * (AMAZE_TS_CONST);
        
        const int TS_STEP = AMAZE_TS_CONST - 32;
        int tiles_x = ((int)width + 16 + TS_STEP - 1) / TS_STEP;
        int tiles_y = ((int)height + 16 + TS_STEP - 1) / TS_STEP;
        size_t total_tiles = tiles_x * tiles_y;
        
        // 中間バッファを total_tiles 分のサイズで確保
        size_t total_tile_pixels = total_tiles * TILE_PIXELS;
        size_t total_tile_pixels_half = total_tiles * TILE_PIXELS_HALF;

        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithBytes:raw_rgbg_data.data() length:pixel_count * sizeof(ushort4) options:MTLResourceStorageModeShared];
        
        //id<MTLBuffer> rgb_output_buffer = [pimpl_->device newBufferWithLength:pixel_count * 3 * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0] 
                                                            length:pixel_count * 3 * sizeof(float) 
                                                            options:MTLResourceStorageModeShared 
                                                            deallocator:nil];
        
        // Basic processing buffers
        id<MTLBuffer> rgbgreen_buf   = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> delhvsqsum_buf = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dirwts0_buf    = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dirwts1_buf    = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> vcd_buf        = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> hcd_buf        = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];  // デバッグのためSharedに変更
        id<MTLBuffer> vcdalt_buf     = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> hcdalt_buf     = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> cddiffsq_buf   = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dgintv_buf     = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dginth_buf     = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];
                
        // Note: rbm=vcd, rbp=hcd, rbint=dirwts0, pmwt=dirwts1 (buffer aliasing in Metal shader)
        
        // Existing half-size buffers
        id<MTLBuffer> hvwt_buf       = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dgrbsq1m_buf   = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dgrbsq1p_buf   = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> nyquist_buf    = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(unsigned char) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> nyquist2_buf   = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(unsigned char) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> nyqutest_buf   = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(unsigned char) options:MTLResourceStorageModePrivate];

        // CPU-equivalent processing buffers (half-size)
        id<MTLBuffer> delp_buf       = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> delm_buf       = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(float) options:MTLResourceStorageModePrivate];
        
        // CPU版完全同等: Dgrb0は専用バッファ、Dgrb1はhcdバッファを再利用
        id<MTLBuffer> Dgrb0_buf      = [pimpl_->device newBufferWithLength:total_tile_pixels_half * sizeof(float) options:MTLResourceStorageModePrivate]; // G-R difference
        // Note: Dgrb1はhcdバッファを再利用するため、専用バッファは作成しない（CPU版 Line 1450: float* Dgrb1 = hcd;）

        id<MTLBuffer> cfa            = [pimpl_->device newBufferWithLength:total_tile_pixels * sizeof(float) options:MTLResourceStorageModePrivate];

        // Calculate dynamic clipping thresholds from cam_mul (same as CPU implementation)
        float max_mul = std::max({cam_mul[0], cam_mul[2]});
        float min_mul = std::min({cam_mul[0], cam_mul[2]});
        float initialGain = (min_mul > 1e-6f) ? (max_mul / min_mul) : 1.0f;
        float clip_pt = 1.0f / initialGain;
        float clip_pt8 = 0.8f / initialGain;

        BayerParams params = { 
            (uint32_t)width, 
            (uint32_t)height, 
            4,
            (float)maximum_value,
            filters,
            {(uint32_t)tiles_x, (uint32_t)tiles_y},
            clip_pt,
            clip_pt8
        };        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        MTLSize grid = MTLSizeMake(tiles_x, tiles_y, 1);
        MTLSize group = MTLSizeMake(4, 4, 1); // スレッドグループサイズを最適化        

        [encoder setComputePipelineState:pimpl_->bayer_amaze_pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        [encoder setBuffer:rgbgreen_buf offset:0 atIndex:3];
        [encoder setBuffer:delhvsqsum_buf offset:0 atIndex:4];
        [encoder setBuffer:dirwts0_buf offset:0 atIndex:5];
        [encoder setBuffer:dirwts1_buf offset:0 atIndex:6];
        [encoder setBuffer:vcd_buf offset:0 atIndex:7];
        [encoder setBuffer:hcd_buf offset:0 atIndex:8];
        [encoder setBuffer:vcdalt_buf offset:0 atIndex:9];
        [encoder setBuffer:hcdalt_buf offset:0 atIndex:10];
        [encoder setBuffer:cddiffsq_buf offset:0 atIndex:11];
        [encoder setBuffer:hvwt_buf offset:0 atIndex:12];
        [encoder setBuffer:dgintv_buf offset:0 atIndex:13];
        [encoder setBuffer:dginth_buf offset:0 atIndex:14];
        [encoder setBuffer:dgrbsq1m_buf offset:0 atIndex:15];
        [encoder setBuffer:dgrbsq1p_buf offset:0 atIndex:16];
        [encoder setBuffer:nyquist_buf offset:0 atIndex:17];
        [encoder setBuffer:nyquist2_buf offset:0 atIndex:18];        
        [encoder setBuffer:nyqutest_buf offset:0 atIndex:19];        
        [encoder setBuffer:delp_buf offset:0 atIndex:20];
        [encoder setBuffer:delm_buf offset:0 atIndex:21];
        [encoder setBuffer:Dgrb0_buf offset:0 atIndex:22]; // G-R difference buffer
        [encoder setBuffer:cfa offset:0 atIndex:23];
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Note: Dgrb1はhcd_buf (index 8) を再利用（CPU版同等: float* Dgrb1 = hcd;）
        // Note: rbm/rbp/pmwt/rbint buffers now use aliasing in Metal shader

        [encoder setComputePipelineState:pimpl_->bayer_border_pipeline];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:params_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];    

        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;
        
        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Demosaic X-Trans 1pass
//===================================================================

bool GPUAccelerator::demosaic_xtrans_1pass(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, const char (&xtrans)[6][6], const float (&color_matrix)[3][4], uint16_t maximum_value) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->xtrans_1pass_pipeline || !pimpl_->xtrans_border_pipeline) return false;
    
    @autoreleasepool {
        const auto width = raw_buffer.width, height = raw_buffer.height, pixel_count = width * height;
        
        // Prepare raw data
        std::vector<uint16_t> raw_gpu_data(pixel_count);
        for (size_t i = 0; i < pixel_count; ++i) {
            size_t r = i / width, c = i % width;
            raw_gpu_data[i] = raw_buffer.image[i][fcol_xtrans(r, c, xtrans)];
        }
        
        // Create Metal buffers
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithBytes:raw_gpu_data.data() 
                                                            length:pixel_count * sizeof(uint16_t) 
                                                            options:MTLResourceStorageModeShared];
        
        //id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithLength:pixel_count * 3 * sizeof(float)
        //                                                  options:MTLResourceStorageModeShared];                                                                    
        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0] 
                                                            length:pixel_count * 3 * sizeof(float) 
                                                            options:MTLResourceStorageModeShared 
                                                            deallocator:nil];
        
        // Prepare XTrans parameters
        XTransParams params = {
            (uint32_t)width,
            (uint32_t)height,
            1u,
            (float)maximum_value
        };
        std::memcpy(params.xtrans, xtrans, sizeof(params.xtrans));
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                            length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        // Execute pipelines
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];

        MTLSize grid_size = MTLSizeMake(width, height, 1);
        MTLSize threadgroup_size = MTLSizeMake(16, 16, 1);

        // First do border interpolation
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pimpl_->xtrans_border_pipeline];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Then do 1-pass demosaic
        [encoder setComputePipelineState:pimpl_->xtrans_1pass_pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;
        
        // Copy results
        //memcpy(&rgb_buffer.image[0][0], [rgb_metal_buffer contents], pixel_count * 3 * sizeof(float));
        
        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Demosaic X-Trans 3pass
//===================================================================

bool GPUAccelerator::demosaic_xtrans_3pass(const ImageBuffer& raw_buffer,
                                          ImageBufferFloat& rgb_buffer,
                                          const char (&xtrans)[6][6],
                                          const float (&color_matrix)[3][4],
                                          uint16_t maximum_value) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->xtrans_3pass_pipeline) return false;
    
    @autoreleasepool {
        const auto width = raw_buffer.width, height = raw_buffer.height, pixel_count = width * height;

        constexpr int ts = 114;
        constexpr int passes = 3;
        constexpr int ndir = 4 << (passes > 1);
        
        // 生データ準備
        std::vector<uint16_t> raw_gpu_data(pixel_count);
        for (size_t i = 0; i < pixel_count; ++i) {
            size_t r = i / width, c = i % width;
            raw_gpu_data[i] = raw_buffer.image[i][fcol_xtrans(r, c, xtrans)];
        }
        
        // Metalバッファ作成
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithBytes:raw_gpu_data.data() 
                                                            length:pixel_count * sizeof(uint16_t) 
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0] 
                                                            length:pixel_count * 3 * sizeof(float) 
                                                            options:MTLResourceStorageModeShared 
                                                            deallocator:nil];
        
        // ワークスペースバッファ
        size_t tile_count_x = (width + (ts - 16) - 1) / (ts - 16);
        size_t tile_count_y = (height + (ts - 16) - 1) / (ts - 16);
        size_t tile_count = tile_count_x * tile_count_y;
                
        // allhexデータ計算
        short allhex_data[2][3][3][8];
        uint16_t sgrow = 0, sgcol = 0;
        
        auto isgreen = [&](int row, int col) -> bool {
            return (xtrans[row % 3][col % 3] & 1) != 0;
        };
        
        constexpr short orth[12] = { 1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1 };
        constexpr short patt[2][16] = {
            { 0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0 },
            { 0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1 }
        };
        
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                int gint = isgreen(row, col) ? 1 : 0;
                
                for (int ng = 0, d = 0; d < 10; d += 2) {
                    if (isgreen(row + orth[d] + 6, col + orth[d + 2] + 6)) {
                        ng = 0;
                    } else {
                        ng++;
                    }
                    
                    if (ng == 4) {
                        sgrow = row;
                        sgcol = col;
                    }
                    
                    if (ng == gint + 1) {
                        for (int c = 0; c < 8; c++) {
                            int v = orth[d] * patt[gint][c * 2] + orth[d + 1] * patt[gint][c * 2 + 1];
                            int h = orth[d + 2] * patt[gint][c * 2] + orth[d + 3] * patt[gint][c * 2 + 1];
                            allhex_data[0][row][col][c ^ (gint * 2 & d)] = h + v * width;
                            allhex_data[1][row][col][c ^ (gint * 2 & d)] = h + v * ts;
                        }
                    }
                }
            }
        }
        
        id<MTLBuffer> allhex_buffer = [pimpl_->device newBufferWithBytes:allhex_data
                                                                  length:sizeof(allhex_data)
                                                                 options:MTLResourceStorageModeShared];
        
        vector_uint2 sg_coords = {sgrow, sgcol};
        id<MTLBuffer> sg_coords_buffer = [pimpl_->device newBufferWithBytes:&sg_coords
                                                                     length:sizeof(sg_coords)
                                                                    options:MTLResourceStorageModeShared];
        
        // XYZ_CAMマトリクス準備
        float xyz_cam[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                xyz_cam[i][j] = color_matrix[i][j];
            }
        }
        id<MTLBuffer> xyz_cam_buffer = [pimpl_->device newBufferWithBytes:xyz_cam
                                                                   length:sizeof(xyz_cam)
                                                                  options:MTLResourceStorageModeShared];
        
        // cbrt LUT準備
        constexpr int table_size = 1<<16;
        std::vector<float> cbrt_lut_vec(table_size);
        for (int i = 0; i < table_size; i++) {
            double r = i / static_cast<double>(table_size - 1);
            cbrt_lut_vec[i] = static_cast<float>(r > (216.0/24389.0) ? std::cbrt(r) : (24389.0/27.0 * r + 16.0) / 116.0);
        }
        id<MTLBuffer> cbrt_lut_buffer = [pimpl_->device newBufferWithBytes:cbrt_lut_vec.data()
                                                                    length:cbrt_lut_vec.size() * sizeof(float)
                                                                   options:MTLResourceStorageModeShared];
        
        // XTransパラメータ準備
        XTransParams params = {
            (uint32_t)width,
            (uint32_t)height,
            8u,
            (float)maximum_value,
            {},  // xtrans will be copied below
            0    // use_cielab = 0 (YPbPr mode by default)
        };
        std::memcpy(params.xtrans, xtrans, sizeof(params.xtrans));
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                                  length:sizeof(params)
                                                                 options:MTLResourceStorageModeShared];

        id<MTLBuffer> tile_data_buffer = [pimpl_->device newBufferWithLength:tile_count * sizeof(XTransTileData) options:MTLResourceStorageModePrivate];

        // スレッドグループサイズ計算
        MTLSize grid_size = MTLSizeMake(tile_count_x, tile_count_y, 1);
        MTLSize threadgroup_size = MTLSizeMake(4, 4, 1);
        
        // パイプライン実行
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->xtrans_3pass_pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:cbrt_lut_buffer offset:0 atIndex:2];
        [encoder setBuffer:allhex_buffer offset:0 atIndex:3];
        [encoder setBuffer:sg_coords_buffer offset:0 atIndex:4];
        [encoder setBuffer:xyz_cam_buffer offset:0 atIndex:5];
        [encoder setBuffer:tile_data_buffer offset:0 atIndex:6];        
        [encoder setBuffer:params_buffer offset:0 atIndex:7];
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        
        // ボーダー補間
        [encoder setComputePipelineState:pimpl_->xtrans_border_pipeline];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:threadgroup_size];
      
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        return command_buffer.status == MTLCommandBufferStatusCompleted;
    }
#else
    return false;
#endif
}

//===================================================================
// Apply White Balance
//===================================================================

// ImageBufferFloat processing methods
bool GPUAccelerator::apply_white_balance(const ImageBufferFloat& rgb_input, ImageBufferFloat& rgb_output, const float wb_multipliers[4]) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->apply_white_balance_float_pipeline) return false;
    
    @autoreleasepool {
        size_t pixel_count = rgb_input.width * rgb_input.height;
        size_t buffer_size = pixel_count * 3 * sizeof(float);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:&rgb_input.image[0][0]
                                                                 length:buffer_size
                                                                options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:buffer_size
                                                                  options:MTLResourceStorageModeShared];
        
        WhiteBalanceParams wb_params = {
            static_cast<uint32_t>(rgb_input.width),
            static_cast<uint32_t>(rgb_input.height),
            {wb_multipliers[0], wb_multipliers[1], wb_multipliers[2], wb_multipliers[3]}
        };
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&wb_params
                                                                  length:sizeof(wb_params)
                                                                 options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->apply_white_balance_float_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        MTLSize thread_group_size = MTLSizeMake(32, 32, 1);
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;
        
        memcpy(&rgb_output.image[0][0], [output_buffer contents], buffer_size);
        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Convert Color Space
//===================================================================

bool GPUAccelerator::convert_color_space(const ImageBufferFloat& rgb_input, ImageBufferFloat& rgb_output, const float transform[3][4]) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->convert_color_space_float_pipeline) return false;
    
    @autoreleasepool {
        size_t pixel_count = rgb_input.width * rgb_input.height;
        size_t buffer_size = pixel_count * 3 * sizeof(float);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:&rgb_input.image[0][0]
                                                                 length:buffer_size
                                                                options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:buffer_size
                                                                  options:MTLResourceStorageModeShared];
        
        ColorSpaceParams cs_params = {
            static_cast<uint32_t>(rgb_input.width),
            static_cast<uint32_t>(rgb_input.height),
            {}
        };
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                cs_params.transform[i][j] = transform[i][j];
            }
        }
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&cs_params
                                                                  length:sizeof(cs_params)
                                                                 options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->convert_color_space_float_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        MTLSize thread_group_size = MTLSizeMake(32, 32, 1);
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;
        
        memcpy(&rgb_output.image[0][0], [output_buffer contents], buffer_size);
        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Gamma Correct
//===================================================================

bool GPUAccelerator::gamma_correct(const ImageBufferFloat& rgb_input, ImageBufferFloat& rgb_output, float gamma_power, float gamma_slope, int output_color_space) {
#ifdef __OBJC__
    if (!pimpl_->initialized || !pimpl_->gamma_correct_float_pipeline) return false;
    
    @autoreleasepool {
        size_t pixel_count = rgb_input.width * rgb_input.height;
        size_t buffer_size = pixel_count * 3 * sizeof(float);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytes:&rgb_input.image[0][0]
                                                                 length:buffer_size
                                                                options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithLength:buffer_size
                                                                  options:MTLResourceStorageModeShared];
        
        GammaParams gamma_params = {
            static_cast<uint32_t>(rgb_input.width),
            static_cast<uint32_t>(rgb_input.height),
            gamma_power,
            gamma_slope,
            static_cast<uint32_t>(output_color_space)
        };
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&gamma_params
                                                                  length:sizeof(gamma_params)
                                                                 options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pimpl_->gamma_correct_float_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        
        MTLSize thread_group_size = MTLSizeMake(32, 32, 1);
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;
        
        memcpy(&rgb_output.image[0][0], [output_buffer contents], buffer_size);
        return true;
    }
#else
    return false;
#endif
}

} // namespace libraw_enhanced