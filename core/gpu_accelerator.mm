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
#include <unordered_map>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <simd/simd.h>
using namespace simd;
#endif

namespace libraw_enhanced {

class GPUAccelerator::Impl {
public:
    Impl() : initialized(false) {}

    bool initialized;
    std::string device_info = "Apple Silicon GPU";

#ifdef __OBJC__
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;
#endif
};

GPUAccelerator::GPUAccelerator() : pimpl_(std::make_unique<Impl>()) {}
GPUAccelerator::~GPUAccelerator() = default;

bool GPUAccelerator::initialize() {    
#ifdef __OBJC__
    @autoreleasepool {
        std::cout << "🚀 Initializing GPU..." << std::endl;
        pimpl_->device = MTLCreateSystemDefaultDevice();
        if (!pimpl_->device) {
            std::cout << "❌ FAILED: MTLCreateSystemDefaultDevice() returned nil." << std::endl;
            return false;
        }
        std::cout << "📋 Metal device created: " << [[pimpl_->device name] UTF8String] << std::endl;
        
        pimpl_->command_queue = [pimpl_->device newCommandQueue];
        if (!pimpl_->command_queue) {
            std::cout << "❌ FAILED: newCommandQueue() returned nil." << std::endl;
            return false;
        }
        std::cout << "📋 Command queue created." << std::endl;
        
        pimpl_->initialized = true;
        std::cout << "✅ GPU Initialization SUCCESSFUL." << std::endl;
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

//===================================================================
// Demosaic Bayer Liner
//===================================================================

bool GPUAccelerator::demosaic_bayer_linear(const ImageBufferFloat& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, float maximum_value) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    
    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> bayer_linear_pipeline = get_pipeline("demosaic_bayer_linear");
    id<MTLComputePipelineState> bayer_border_pipeline = get_pipeline("demosaic_bayer_border");
    if (!bayer_linear_pipeline || !bayer_border_pipeline) {
        std::cerr << "❌ Failed to get bayer linear pipelines" << std::endl;
        return false;
    }
    
    @autoreleasepool {
        const auto width = raw_buffer.width, height = raw_buffer.height, pixel_count = width * height;
        
        // Prepare raw data
        std::vector<float> raw_gpu_data(pixel_count);
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < pixel_count; i++) {
            size_t r = i / width, c = i % width;
            uint32_t color_idx = fcol_bayer(r, c, filters);
            raw_gpu_data[i] = raw_buffer.image[i][color_idx];
        }

        // Create Metal buffers
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:raw_gpu_data.data() 
                                                            length:pixel_count * sizeof(float) 
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0] 
                                                            length:pixel_count * 3 * sizeof(float) 
                                                            options:MTLResourceStorageModeShared 
                                                            deallocator:nil];
        
        DemosaicBayerParams params = {
            (uint32_t)width,
            (uint32_t)height,
            1,
            maximum_value,
            filters,
        };
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytesNoCopy:&params 
                                                            length:sizeof(params) 
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        // Execute pipeline
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        MTLSize grid_size = MTLSizeMake(width, height, 1);
        MTLSize threadgroup_size = MTLSizeMake(16, 16, 1);

        [encoder setComputePipelineState:bayer_linear_pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:bayer_border_pipeline];
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

bool GPUAccelerator::demosaic_bayer_amaze(const ImageBufferFloat& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, const float (&cam_mul)[4], float maximum_value) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    
    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> bayer_amaze_pipeline = get_pipeline("demosaic_bayer_amaze");
    id<MTLComputePipelineState> bayer_border_pipeline = get_pipeline("demosaic_bayer_border");
    if (!bayer_amaze_pipeline || !bayer_border_pipeline) {
        std::cerr << "❌ Failed to get bayer amaze pipelines" << std::endl;
        return false;
    }
    
    @autoreleasepool {
        const auto width = raw_buffer.width, height = raw_buffer.height, pixel_count = width * height;

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

        // Owned Metal buffers — avoids newBufferWithBytesNoCopy page-alignment
        // requirement.  Fill raw input directly into the Metal Shared buffer to
        // skip an intermediate std::vector copy.
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithLength:pixel_count * sizeof(float4)
                                                            options:MTLResourceStorageModeShared];
        if (!raw_metal_buffer) return false;
        {
            float4* raw_ptr = (float4*)[raw_metal_buffer contents];
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < pixel_count; i++) {
                raw_ptr[i] = { raw_buffer.image[i][0], raw_buffer.image[i][1],
                               raw_buffer.image[i][2], raw_buffer.image[i][1] };
            }
        }

        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0]
                                                            length:pixel_count * 3 * sizeof(float)
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!rgb_metal_buffer) return false;

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

        DemosaicBayerParams params = { 
            (uint32_t)width, 
            (uint32_t)height, 
            4,
            maximum_value,
            filters,
            {(uint32_t)tiles_x, (uint32_t)tiles_y},
            clip_pt,
            clip_pt8
        };        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                            length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];
        if (!params_buffer) return false;

        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        MTLSize grid = MTLSizeMake(tiles_x, tiles_y, 1);
        MTLSize group = MTLSizeMake(4, 4, 1);

        [encoder setComputePipelineState:bayer_amaze_pipeline];
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

        [encoder setComputePipelineState:bayer_border_pipeline];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:params_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;

        // GPU wrote directly into rgb_buffer.image via newBufferWithBytesNoCopy — no memcpy needed.

        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Demosaic X-Trans 1pass
//===================================================================

bool GPUAccelerator::demosaic_xtrans_1pass(const ImageBufferFloat& raw_buffer, ImageBufferFloat& rgb_buffer, const char (&xtrans)[6][6], const float (&color_matrix)[3][4], float maximum_value) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    
    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> xtrans_1pass_pipeline = get_pipeline("demosaic_xtrans_1pass");
    id<MTLComputePipelineState> xtrans_border_pipeline = get_pipeline("demosaic_xtrans_border");
    if (!xtrans_1pass_pipeline || !xtrans_border_pipeline) {
        std::cerr << "❌ Failed to get X-Trans 1pass pipelines" << std::endl;
        return false;
    }
    
    @autoreleasepool {
        const auto width = raw_buffer.width, height = raw_buffer.height, pixel_count = width * height;

        // Owned Metal buffers — no page-alignment dependency.
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithLength:pixel_count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        if (!raw_metal_buffer) return false;
        {
            float* raw_ptr = (float*)[raw_metal_buffer contents];
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < pixel_count; ++i) {
                size_t r = i / width, c = i % width;
                raw_ptr[i] = raw_buffer.image[i][fcol_xtrans(r, c, xtrans)];
            }
        }

        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0]
                                                            length:pixel_count * 3 * sizeof(float)
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!rgb_metal_buffer) return false;

        // Prepare XTrans parameters
        DemosaicXTransParams params = {
            (uint32_t)width,
            (uint32_t)height,
            1u,
            maximum_value
        };
        std::memcpy(params.xtrans, xtrans, sizeof(params.xtrans));
        
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                            length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        // Execute pipelines
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        MTLSize grid_size = MTLSizeMake(width, height, 1);
        MTLSize threadgroup_size = MTLSizeMake(16, 16, 1);

        // First do border interpolation
        [encoder setComputePipelineState:xtrans_border_pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Then do 1-pass demosaic
        [encoder setComputePipelineState:xtrans_1pass_pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];        
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;

        // GPU wrote directly into rgb_buffer.image via newBufferWithBytesNoCopy — no memcpy needed.

        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Demosaic X-Trans 3pass
//===================================================================

bool GPUAccelerator::demosaic_xtrans_3pass(const ImageBufferFloat& raw_buffer,
                                          ImageBufferFloat& rgb_buffer,
                                          const char (&xtrans)[6][6],
                                          const float (&color_matrix)[3][4],
                                          float maximum_value) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    
    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> xtrans_3pass_pipeline = get_pipeline("demosaic_xtrans_3pass");
    id<MTLComputePipelineState> xtrans_border_pipeline = get_pipeline("demosaic_xtrans_border");
    if (!xtrans_3pass_pipeline || !xtrans_border_pipeline) {
        std::cerr << "❌ Failed to get X-Trans 3pass pipelines" << std::endl;
        return false;
    }
    
    @autoreleasepool {
        const auto width = raw_buffer.width, height = raw_buffer.height, pixel_count = width * height;

        constexpr int ts = XTRANS_3PASS_TS;

        // Owned Metal buffers — no page-alignment dependency.
        id<MTLBuffer> raw_metal_buffer = [pimpl_->device newBufferWithLength:pixel_count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        if (!raw_metal_buffer) return false;
        {
            float* raw_ptr = (float*)[raw_metal_buffer contents];
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < pixel_count; ++i) {
                size_t r = i / width, c = i % width;
                raw_ptr[i] = raw_buffer.image[i][fcol_xtrans(r, c, xtrans)];
            }
        }

        id<MTLBuffer> rgb_metal_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0]
                                                            length:pixel_count * 3 * sizeof(float)
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!rgb_metal_buffer) return false;

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
        
        // NOTE: OMP removed intentionally.  This 3×3 loop (9 iterations) fills
        // sgrow/sgcol with the single matching (row,col) pair.  Parallelising it
        // would be a data race on sgrow/sgcol (multiple threads writing the same
        // shared variables without synchronisation).  The loop is tiny so there
        // is no performance benefit anyway.
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
        
        // Use newBufferWithBytes for all small/stack-allocated data — these are
        // never page-aligned so newBufferWithBytesNoCopy would always return nil.
        id<MTLBuffer> allhex_buffer = [pimpl_->device newBufferWithBytes:allhex_data
                                                            length:sizeof(allhex_data)
                                                            options:MTLResourceStorageModeShared];
        if (!allhex_buffer) return false;

        vector_uint2 sg_coords = {sgrow, sgcol};
        id<MTLBuffer> sg_coords_buffer = [pimpl_->device newBufferWithBytes:&sg_coords
                                                            length:sizeof(sg_coords)
                                                            options:MTLResourceStorageModeShared];
        if (!sg_coords_buffer) return false;

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
        if (!xyz_cam_buffer) return false;

        // cbrt LUT準備 — fill owned Metal buffer directly to skip vector copy
        constexpr int table_size = 1<<16;
        id<MTLBuffer> cbrt_lut_buffer = [pimpl_->device newBufferWithLength:table_size * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        if (!cbrt_lut_buffer) return false;
        {
            float* lut_ptr = (float*)[cbrt_lut_buffer contents];
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (int i = 0; i < table_size; i++) {
                double r = i / static_cast<double>(table_size - 1);
                lut_ptr[i] = static_cast<float>(r > (216.0/24389.0) ? std::cbrt(r) : (24389.0/27.0 * r + 16.0) / 116.0);
            }
        }

        // XTransパラメータ準備
        DemosaicXTransParams params = {
            (uint32_t)width,
            (uint32_t)height,
            8u,
            maximum_value,
            {},  // xtrans will be copied below
            0    // use_cielab = 0 (YPbPr mode by default)
        };
        std::memcpy(params.xtrans, xtrans, sizeof(params.xtrans));

        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                            length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];
        if (!params_buffer) return false;

        id<MTLBuffer> tile_data_buffer = [pimpl_->device newBufferWithLength:tile_count * sizeof(XTrans3passTile)
                                                            options:MTLResourceStorageModePrivate];

        // パイプライン実行 - 新しい遅延ロード方式使用
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        // スレッドグループサイズ計算
        MTLSize grid_size = MTLSizeMake(tile_count_x, tile_count_y, 1);
        MTLSize threadgroup_size = MTLSizeMake(4, 4, 1);
 
        [encoder setComputePipelineState:xtrans_3pass_pipeline];
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
        
        [encoder setComputePipelineState:xtrans_border_pipeline];
        [encoder setBuffer:raw_metal_buffer offset:0 atIndex:0];
        [encoder setBuffer:rgb_metal_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;

        // GPU wrote directly into rgb_buffer.image via newBufferWithBytesNoCopy — no memcpy needed.

        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Apply White Balance
//===================================================================

bool GPUAccelerator::apply_white_balance(const ImageBuffer& raw_buffer,
                        ImageBufferFloat& rgb_buffer,
                        const float wb_multipliers[4],
                        uint32_t filters,
                        const char xtrans[6][6]) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    
    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> apply_white_balance_pipeline = get_pipeline((filters == FILTERS_XTRANS)? "apply_white_balance_xtrans" : "apply_white_balance_bayer");
    if (!apply_white_balance_pipeline) {
        std::cerr << "❌ Failed to get apply white balance pipeline" << std::endl;
        return false;
    }
    
    @autoreleasepool {
        size_t pixel_count = raw_buffer.width * raw_buffer.height;
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytesNoCopy:&raw_buffer.image[0][0]
                                                                length:pixel_count * 4 * sizeof(uint16_t)
                                                                options:MTLResourceStorageModeShared
                                                                deallocator:nil];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_buffer.image[0][0]
                                                                length:pixel_count * 3 * sizeof(float)
                                                                options:MTLResourceStorageModeShared
                                                                deallocator:nil];
        
        ApplyWhiteBalanceParams params = {
            static_cast<uint32_t>(rgb_buffer.width),
            static_cast<uint32_t>(rgb_buffer.height),
            {wb_multipliers[0], wb_multipliers[1], wb_multipliers[2], wb_multipliers[3]},
            filters,
        };
        memcpy(params.xtrans, xtrans, sizeof(params.xtrans));
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytesNoCopy:&params
                                                                length:sizeof(params)
                                                                options:MTLResourceStorageModeShared
                                                                deallocator:nil];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        MTLSize grid_size = MTLSizeMake(rgb_buffer.width, rgb_buffer.height, 1);
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);

        [encoder setComputePipelineState:apply_white_balance_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
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
// Convert Color Space
//===================================================================

bool GPUAccelerator::convert_color_space(const ImageBufferFloat& rgb_input,
                                        ImageBufferFloat& rgb_output,
                                        const float transform[3][4]) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    
    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> convert_color_space_pipeline = get_pipeline("convert_color_space");
    if (!convert_color_space_pipeline) {
        std::cerr << "❌ Failed to get convert color space pipeline" << std::endl;
        return false;
    }
     
    @autoreleasepool {
        size_t pixel_count = rgb_input.width * rgb_input.height;
        size_t buffer_size = pixel_count * 3 * sizeof(float);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_input.image[0][0]
                                                            length:buffer_size
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_output.image[0][0]
                                                            length:buffer_size
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        ConvertColorSpaceParams params = {
            static_cast<uint32_t>(rgb_input.width),
            static_cast<uint32_t>(rgb_input.height),
            {
                {transform[0][0], transform[0][1], transform[0][2], transform[0][3]},
                {transform[1][0], transform[1][1], transform[1][2], transform[1][3]},
                {transform[2][0], transform[2][1], transform[2][2], transform[2][3]},
            },
        };
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytesNoCopy:&params
                                                            length:sizeof(params)
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);

        [encoder setComputePipelineState:convert_color_space_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
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
// Gamma Correct
//===================================================================

bool GPUAccelerator::gamma_correct(const ImageBufferFloat& rgb_input,
                                    ImageBufferFloat& rgb_output,
                                    float gamma_power,
                                    float gamma_slope,
                                    int output_color_space) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    
    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> gamma_correct_pipeline = get_pipeline("gamma_correct");
    if (!gamma_correct_pipeline) {
        std::cerr << "❌ Failed to get gamma correct pipeline" << std::endl;
        return false;
    }
    
    @autoreleasepool {
        size_t pixel_count = rgb_input.width * rgb_input.height;
        size_t buffer_size = pixel_count * 3 * sizeof(float);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_input.image[0][0]
                                                            length:buffer_size
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_output.image[0][0]
                                                            length:buffer_size
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        GammaCorrectParams params = {
            static_cast<uint32_t>(rgb_input.width),
            static_cast<uint32_t>(rgb_input.height),
            gamma_power,
            gamma_slope,
            static_cast<uint32_t>(output_color_space)
        };
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytesNoCopy:&params
                                                            length:sizeof(params)
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);

        [encoder setComputePipelineState:gamma_correct_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
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
// トーンマッピング
//===================================================================

bool GPUAccelerator::tone_mapping(const ImageBufferFloat& rgb_input,
                                ImageBufferFloat& rgb_output,
                                float after_scale) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    
    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> tone_mapping_pipeline = get_pipeline("tone_mapping");
    if (!tone_mapping_pipeline) {
        std::cerr << "❌ Failed to get tone mapping pipeline" << std::endl;
        return false;
    }
    
    @autoreleasepool {
        size_t pixel_count = rgb_input.width * rgb_input.height;
        size_t buffer_size = pixel_count * 3 * sizeof(float);
        
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_input.image[0][0]
                                                            length:buffer_size
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_output.image[0][0]
                                                            length:buffer_size
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        ToneMappingParams params = {
            static_cast<uint32_t>(rgb_input.width),
            static_cast<uint32_t>(rgb_input.height),
            after_scale
        };
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytesNoCopy:&params
                                                            length:sizeof(params)
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        id<MTLCommandBuffer> command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);

        [encoder setComputePipelineState:tone_mapping_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];        
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
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
// マイクロコントラスト
//===================================================================

bool GPUAccelerator::enhance_micro_contrast(const ImageBufferFloat& rgb_input,
                                            ImageBufferFloat& rgb_output,
                                            float threshold,
                                            float strength,
                                            float target_contrast) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;

    // 遅延ローディングでパイプライン取得
    id<MTLComputePipelineState> enhance_micro_contrast_pipeline = get_pipeline("enhance_micro_contrast");
    if (!enhance_micro_contrast_pipeline) {
        std::cerr << "❌ Failed to get enhance micro contrast pipeline" << std::endl;
        return false;
    }

    @autoreleasepool {
        size_t pixel_count = rgb_input.width * rgb_input.height;
        size_t rgb_bytes   = pixel_count * 3 * sizeof(float);

        // Both buffers wrap caller-owned page-aligned memory via newBufferWithBytesNoCopy.
        id<MTLBuffer> input_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_input.image[0][0]
                                                            length:rgb_bytes
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!input_buffer) return false;

        id<MTLBuffer> I_buffer = [pimpl_->device newBufferWithLength:pixel_count * 4 * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        if (!I_buffer) return false;
        vector_float3* I = (vector_float3 *)[I_buffer contents];

        // Shared (not Private) so CPU can read back after GPU Gaussian blurs to
        // compute max_local_std correctly — the Metal shader's original approach
        // of computing a global max with unprotected writes and threadgroup_barrier
        // was a data race (threadgroup_barrier only syncs within one threadgroup).
        id<MTLBuffer> local_mean_buffer = [pimpl_->device newBufferWithLength:pixel_count * 4 * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        if (!local_mean_buffer) return false;

        id<MTLBuffer> local_var_buffer = [pimpl_->device newBufferWithLength:pixel_count * 4 * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        if (!local_var_buffer) return false;

        id<MTLBuffer> output_buffer = [pimpl_->device newBufferWithBytesNoCopy:&rgb_output.image[0][0]
                                                            length:rgb_bytes
                                                            options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!output_buffer) return false;

        // max_local_std will be filled in by CPU after GPU Gaussian blurs complete.
        EnhanceMicroContrastParams params = {
            static_cast<uint32_t>(rgb_input.width),
            static_cast<uint32_t>(rgb_input.height),
            threshold,
            strength,
            target_contrast,
            0.f  // max_local_std placeholder; overwritten below
        };
        id<MTLBuffer> params_buffer = [pimpl_->device newBufferWithBytes:&params
                                                            length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];
        if (!params_buffer) return false;

        MTLTextureDescriptor *texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                            width:NSUInteger(rgb_input.width)
                                                            height:NSUInteger(rgb_input.height)
                                                            mipmapped:NO];
        texDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        id<MTLTexture> I_texture = [I_buffer newTextureWithDescriptor:texDesc
                                                            offset:0
                                                            bytesPerRow:NSUInteger(rgb_input.width * 4 * sizeof(float))];
        id<MTLTexture> local_mean_texture = [local_mean_buffer newTextureWithDescriptor:texDesc
                                                            offset:0
                                                            bytesPerRow:NSUInteger(rgb_input.width * 4 * sizeof(float))];
        id<MTLTexture> local_var_texture = [local_var_buffer newTextureWithDescriptor:texDesc
                                                            offset:0
                                                            bytesPerRow:NSUInteger(rgb_input.width * 4 * sizeof(float))];

        MPSImageGaussianBlur *gaussianBlur = [[MPSImageGaussianBlur alloc] initWithDevice:pimpl_->device sigma:5.f / 3.7f];

        id<MTLCommandBuffer> command_buffer;

        MTLSize grid_size = MTLSizeMake(rgb_input.width, rgb_input.height, 1);
        MTLSize thread_group_size = MTLSizeMake(16, 16, 1);

        // 局所平均の計算
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (uint32_t idx = 0; idx < pixel_count; ++idx) {
            I[idx] = {rgb_input.image[idx][0], rgb_input.image[idx][1], rgb_input.image[idx][2]};
        }
        command_buffer = [pimpl_->command_queue commandBuffer];
        [gaussianBlur encodeToCommandBuffer:command_buffer sourceTexture:I_texture destinationTexture:local_mean_texture];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;

        // 局所標準偏差の計算（前半）
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (uint32_t idx = 0; idx < pixel_count; ++idx) {
            I[idx] = I[idx] * I[idx];
        }
        command_buffer = [pimpl_->command_queue commandBuffer];
        [gaussianBlur encodeToCommandBuffer:command_buffer sourceTexture:I_texture destinationTexture:local_var_texture];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;

        // ---------------------------------------------------------------
        // CPU computes max_local_std from GPU-produced Shared buffers.
        //
        // Both local_mean_buffer and local_var_buffer are RGBA32Float
        // (4 floats/pixel) written by MPS Gaussian blur.
        // variance[i] = blur(I²)[i] - blur(I)[i]²
        // local_std[i] = sqrt(max(variance[i], 0))
        // max_local_std = max over all pixels of max(std.r, std.g, std.b)
        //
        // waitUntilCompleted above ensures the GPU writes are visible here.
        // ---------------------------------------------------------------
        {
            const simd_float4* mean_ptr = (const simd_float4*)[local_mean_buffer contents];
            const simd_float4* var_ptr  = (const simd_float4*)[local_var_buffer  contents];
            float max_local_std = 0.f;
#ifdef _OPENMP
#pragma omp parallel for reduction(max: max_local_std)
#endif
            for (size_t i = 0; i < pixel_count; ++i) {
                simd_float4 v  = var_ptr[i] - mean_ptr[i] * mean_ptr[i];
                simd_float4 clamped = simd_max(v, simd_make_float4(0.f));
                simd_float4 s  = simd_make_float4(std::sqrt(clamped.x), std::sqrt(clamped.y),
                                                  std::sqrt(clamped.z), 0.f);
                float m = std::max({s.x, s.y, s.z});
                max_local_std = std::max(max_local_std, m);
            }
            // Write the correct value into the Shared params_buffer that the
            // GPU kernel will read at buffer(4).
            ((EnhanceMicroContrastParams*)[params_buffer contents])->max_local_std = max_local_std;
        }

        command_buffer = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> post_encoder = [command_buffer computeCommandEncoder];
        [post_encoder setComputePipelineState:enhance_micro_contrast_pipeline];
        [post_encoder setBuffer:input_buffer  offset:0 atIndex:0];
        [post_encoder setBuffer:local_mean_buffer offset:0 atIndex:1];
        [post_encoder setBuffer:local_var_buffer  offset:0 atIndex:2];
        [post_encoder setBuffer:output_buffer offset:0 atIndex:3];
        [post_encoder setBuffer:params_buffer offset:0 atIndex:4];
        [post_encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [post_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;

        // GPU wrote directly into rgb_output.image via newBufferWithBytesNoCopy — no memcpy needed.

        return true;
    }
#else
    return false;
#endif
}

//===================================================================
// Axial chromatic-aberration cleanup (cross-channel guided filter on
// R, B with G as guide, edge-gated to bright high-gradient regions).
//
// MPS handles:
//   * MPSImageBox            — every box-filter pass (8 calls)
//   * MPSImageSobel          — single-pass gradient magnitude on G
// Custom Metal kernels handle the algorithm-specific element-wise work
// (plane prep, (a, b) regression, apply, blend).
//===================================================================
bool GPUAccelerator::ca_axial_cleanup(const ImageBufferFloat& rgb_input,
                                       ImageBufferFloat& rgb_output,
                                       int   radius,
                                       float epsilon,
                                       float strength) {
#ifdef __OBJC__
    if (!pimpl_->initialized) return false;
    if (!rgb_input.is_valid()) return false;
    if (radius < 1) radius = 1;
    strength = std::clamp(strength, 0.f, 1.f);

    const NSUInteger W = rgb_input.width;
    const NSUInteger H = rgb_input.height;
    const size_t N      = static_cast<size_t>(W) * static_cast<size_t>(H);
    const size_t rgb_bytes = N * 3 * sizeof(float);

    // ---------------------------------------------------------------
    // CPU pre-pass: compute max(G) for normalisation.  Letting the
    // GPU do this would require a CPU↔GPU sync that wipes out the
    // savings of running on the GPU at all.
    // ---------------------------------------------------------------
    float max_g = 1e-6f;
    {
        const float (*src)[3] = rgb_input.image;
#ifdef _OPENMP
        #pragma omp parallel for reduction(max : max_g)
#endif
        for (size_t i = 0; i < N; i++) {
            const float g = src[i][1];
            if (g > max_g) max_g = g;
        }
    }
    const float norm_ref = std::max(max_g, 1e-6f);
    const float inv_norm = 1.f / norm_ref;

    // ---------------------------------------------------------------
    // Pipelines (cached lazily).
    // ---------------------------------------------------------------
    id<MTLComputePipelineState> p_prepare    = get_pipeline("axial_ca", "axial_ca_prepare");
    id<MTLComputePipelineState> p_compute_ab = get_pipeline("axial_ca", "axial_ca_compute_ab");
    id<MTLComputePipelineState> p_apply      = get_pipeline("axial_ca", "axial_ca_apply");
    id<MTLComputePipelineState> p_blend      = get_pipeline("axial_ca", "axial_ca_blend");
    id<MTLComputePipelineState> p_grad_g     = get_pipeline("axial_ca", "axial_ca_grad_g");
    if (!p_prepare || !p_compute_ab || !p_apply || !p_blend || !p_grad_g) {
        std::cerr << "❌ axial_ca pipeline(s) missing" << std::endl;
        return false;
    }

    @autoreleasepool {
        id<MTLDevice> device = pimpl_->device;
        id<MTLCommandQueue> queue = pimpl_->command_queue;

        // Helper: allocate an R32Float texture-backed buffer big enough
        // for the image.  We use one Shared MTLBuffer per plane so we
        // can wrap it as an R32Float MTLTexture without copies.
        const size_t plane_bytes = N * sizeof(float);

        // Storage for plane buffers — we keep them in an NSMutableArray so
        // they stay live throughout the GPU work (texture-backing buffers
        // can otherwise be released as locals).
        NSMutableArray* plane_buffers = [NSMutableArray array];
        auto make_plane_texture = [&](void) -> id<MTLTexture> {
            id<MTLBuffer> buf = [device newBufferWithLength:plane_bytes
                                                    options:MTLResourceStorageModeShared];
            if (!buf) return nil;
            [plane_buffers addObject:buf];
            MTLTextureDescriptor* d = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                              width:W height:H mipmapped:NO];
            d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            return [buf newTextureWithDescriptor:d
                                            offset:0
                                       bytesPerRow:W * sizeof(float)];
        };

        // Plane textures.
        id<MTLTexture> tx_R   = make_plane_texture();
        id<MTLTexture> tx_G   = make_plane_texture();
        id<MTLTexture> tx_B   = make_plane_texture();
        id<MTLTexture> tx_GG  = make_plane_texture();
        id<MTLTexture> tx_RG  = make_plane_texture();
        id<MTLTexture> tx_BG  = make_plane_texture();
        id<MTLTexture> tx_mR  = make_plane_texture();
        id<MTLTexture> tx_mG  = make_plane_texture();
        id<MTLTexture> tx_mB  = make_plane_texture();
        id<MTLTexture> tx_mGG = make_plane_texture();
        id<MTLTexture> tx_mRG = make_plane_texture();
        id<MTLTexture> tx_mBG = make_plane_texture();
        id<MTLTexture> tx_aR  = make_plane_texture();
        id<MTLTexture> tx_bR  = make_plane_texture();
        id<MTLTexture> tx_aB  = make_plane_texture();
        id<MTLTexture> tx_bB  = make_plane_texture();
        id<MTLTexture> tx_maR = make_plane_texture();
        id<MTLTexture> tx_mbR = make_plane_texture();
        id<MTLTexture> tx_maB = make_plane_texture();
        id<MTLTexture> tx_mbB = make_plane_texture();
        id<MTLTexture> tx_Rfilt        = make_plane_texture();
        id<MTLTexture> tx_Bfilt        = make_plane_texture();
        id<MTLTexture> tx_grad         = make_plane_texture();
        id<MTLTexture> tx_grad_smooth  = make_plane_texture();

        // Source + destination as RGBA32Float textures over the caller-owned
        // page-aligned RGB float[3] buffers.  Note: ImageBufferFloat stores
        // 3 floats per pixel back-to-back, so we wrap as an interleaved
        // RGB32Float buffer.  Apple Metal doesn't have RGB32Float — but we
        // store/load via vector_float4 with the .a channel ignored, which
        // requires the buffer to have 4 floats per pixel.  Our existing
        // ImageBufferFloat is 3 floats/pixel.
        //
        // Easiest path: copy to a 4-channel staging buffer for GPU work and
        // back at the end.  This is what enhance_micro_contrast already
        // does internally.
        id<MTLBuffer> buf_src4 = [device newBufferWithLength:N * 4 * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (!buf_src4) return false;
        {
            vector_float3* dstv = (vector_float3*)[buf_src4 contents];
            const float (*src)[3] = rgb_input.image;
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < N; i++) {
                dstv[i] = {src[i][0], src[i][1], src[i][2]};
            }
        }
        id<MTLBuffer> buf_dst4 = [device newBufferWithLength:N * 4 * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (!buf_dst4) return false;
        MTLTextureDescriptor* texDesc4 = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                          width:W height:H mipmapped:NO];
        texDesc4.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        id<MTLTexture> tx_src = [buf_src4 newTextureWithDescriptor:texDesc4
                                                              offset:0
                                                         bytesPerRow:W * 4 * sizeof(float)];
        id<MTLTexture> tx_dst = [buf_dst4 newTextureWithDescriptor:texDesc4
                                                              offset:0
                                                         bytesPerRow:W * 4 * sizeof(float)];
        if (!tx_src || !tx_dst) return false;

        // Param buffers.
        AxialCaPrepareParams prep_params = {
            static_cast<uint32_t>(W), static_cast<uint32_t>(H), inv_norm};
        AxialCaAbParams ab_params = {
            static_cast<uint32_t>(W), static_cast<uint32_t>(H), epsilon};
        vector_uint2 dims = {static_cast<uint32_t>(W),
                             static_cast<uint32_t>(H)};
        // Edge gating uses the same smoothstep range as the CPU reference
        // (0.01, 0.06) so the GPU output matches bit-for-bit modulo
        // floating-point rounding.
        AxialCaBlendParams blend_params = {
            static_cast<uint32_t>(W), static_cast<uint32_t>(H),
            strength, norm_ref, 0.01f, 0.06f};

        // MPS box filter is used for all box-mean passes (guided-filter
        // box averages and the smoothed-gradient pass).
        MPSImageBox* box = [[MPSImageBox alloc] initWithDevice:device
                                                   kernelWidth:(2 * radius + 1)
                                                  kernelHeight:(2 * radius + 1)];

        const MTLSize threadgroup = MTLSizeMake(16, 16, 1);
        const MTLSize grid        = MTLSizeMake(W, H, 1);

        id<MTLCommandBuffer> cb = [queue commandBuffer];

        // ---------------------------------------------------------------
        // 1) prepare: normalise & split into R/G/B/GG/RG/BG planes.
        // ---------------------------------------------------------------
        {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_prepare];
            [e setTexture:tx_src atIndex:0];
            [e setTexture:tx_R   atIndex:1];
            [e setTexture:tx_G   atIndex:2];
            [e setTexture:tx_B   atIndex:3];
            [e setTexture:tx_GG  atIndex:4];
            [e setTexture:tx_RG  atIndex:5];
            [e setTexture:tx_BG  atIndex:6];
            [e setBytes:&prep_params length:sizeof(prep_params) atIndex:0];
            [e dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [e endEncoding];
        }

        // ---------------------------------------------------------------
        // 2) Six box-filter passes for the guided-filter regression
        //    inputs.  MPS encodes them directly into the same command
        //    buffer.
        // ---------------------------------------------------------------
        [box encodeToCommandBuffer:cb sourceTexture:tx_G  destinationTexture:tx_mG];
        [box encodeToCommandBuffer:cb sourceTexture:tx_GG destinationTexture:tx_mGG];
        [box encodeToCommandBuffer:cb sourceTexture:tx_R  destinationTexture:tx_mR];
        [box encodeToCommandBuffer:cb sourceTexture:tx_RG destinationTexture:tx_mRG];
        [box encodeToCommandBuffer:cb sourceTexture:tx_B  destinationTexture:tx_mB];
        [box encodeToCommandBuffer:cb sourceTexture:tx_BG destinationTexture:tx_mBG];

        // ---------------------------------------------------------------
        // 3) Compute (a, b) regression coefficients for R and B.
        // ---------------------------------------------------------------
        auto run_compute_ab = [&](id<MTLTexture> mp, id<MTLTexture> mIp,
                                   id<MTLTexture> a_out, id<MTLTexture> b_out) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_compute_ab];
            [e setTexture:tx_mG  atIndex:0];
            [e setTexture:tx_mGG atIndex:1];
            [e setTexture:mp     atIndex:2];
            [e setTexture:mIp    atIndex:3];
            [e setTexture:a_out  atIndex:4];
            [e setTexture:b_out  atIndex:5];
            [e setBytes:&ab_params length:sizeof(ab_params) atIndex:0];
            [e dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [e endEncoding];
        };
        run_compute_ab(tx_mR, tx_mRG, tx_aR, tx_bR);
        run_compute_ab(tx_mB, tx_mBG, tx_aB, tx_bB);

        // ---------------------------------------------------------------
        // 4) Box-smooth the (a, b) coefficient maps.
        // ---------------------------------------------------------------
        [box encodeToCommandBuffer:cb sourceTexture:tx_aR destinationTexture:tx_maR];
        [box encodeToCommandBuffer:cb sourceTexture:tx_bR destinationTexture:tx_mbR];
        [box encodeToCommandBuffer:cb sourceTexture:tx_aB destinationTexture:tx_maB];
        [box encodeToCommandBuffer:cb sourceTexture:tx_bB destinationTexture:tx_mbB];

        // ---------------------------------------------------------------
        // 5) Apply: R_filt = mean_aR · G + mean_bR  (same for B).
        // ---------------------------------------------------------------
        auto run_apply = [&](id<MTLTexture> ma, id<MTLTexture> mb,
                              id<MTLTexture> q_out) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_apply];
            [e setTexture:tx_G atIndex:0];
            [e setTexture:ma   atIndex:1];
            [e setTexture:mb   atIndex:2];
            [e setTexture:q_out atIndex:3];
            [e setBytes:&dims length:sizeof(dims) atIndex:0];
            [e dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [e endEncoding];
        };
        run_apply(tx_maR, tx_mbR, tx_Rfilt);
        run_apply(tx_maB, tx_mbB, tx_Bfilt);

        // ---------------------------------------------------------------
        // 6) Edge map: custom 2-tap finite-difference gradient on G
        //    (matches the CPU reference exactly).  Then MPS box-smooth
        //    to match the guided-filter window reach.
        // ---------------------------------------------------------------
        {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_grad_g];
            [e setTexture:tx_G    atIndex:0];
            [e setTexture:tx_grad atIndex:1];
            [e setBytes:&dims length:sizeof(dims) atIndex:0];
            [e dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [e endEncoding];
        }
        [box encodeToCommandBuffer:cb sourceTexture:tx_grad destinationTexture:tx_grad_smooth];

        // ---------------------------------------------------------------
        // 7) Final edge-gated blend.  Writes the de-normalised result
        //    straight to the destination RGBA32Float texture.
        // ---------------------------------------------------------------
        {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_blend];
            [e setTexture:tx_src         atIndex:0];
            [e setTexture:tx_R           atIndex:1];
            [e setTexture:tx_B           atIndex:2];
            [e setTexture:tx_Rfilt       atIndex:3];
            [e setTexture:tx_Bfilt       atIndex:4];
            [e setTexture:tx_grad_smooth atIndex:5];
            [e setTexture:tx_dst         atIndex:6];
            [e setBytes:&blend_params length:sizeof(blend_params) atIndex:0];
            [e dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [e endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "❌ axial_ca GPU command buffer failed" << std::endl;
            return false;
        }

        // Copy the 4-channel staging buffer back into the caller's 3-channel
        // ImageBufferFloat.
        {
            const vector_float3* srcv = (const vector_float3*)[buf_dst4 contents];
            float (*dst)[3] = rgb_output.image;
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < N; i++) {
                dst[i][0] = srcv[i].x;
                dst[i][1] = srcv[i].y;
                dst[i][2] = srcv[i].z;
            }
        }

        std::cout << "📷 [GPU] CA axial cleanup (radius=" << radius
                  << " eps=" << epsilon << " strength=" << strength
                  << " norm=" << norm_ref << ")" << std::endl;
        return true;
    }
#else
    (void)rgb_input; (void)rgb_output; (void)radius; (void)epsilon; (void)strength;
    return false;
#endif
}

//===================================================================
// 遅延ロード + キャッシュ方式の新しいシェーダー管理
//===================================================================

#ifdef __OBJC__
id<MTLComputePipelineState> GPUAccelerator::get_pipeline(const std::string& shader_name, std::string func_name) {
    @autoreleasepool {
        // パイプラインキャッシュから探して原始的に実装
        static std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;

        if (func_name.empty()) {
            func_name = shader_name;
        }
        
        auto it = pipeline_cache.find(func_name);
        if (it != pipeline_cache.end() && it->second != nil) {
            std::cout << "[DEBUG] Using cached pipeline: " << shader_name << std::endl;
            return it->second;
        }
        
        std::cout << "[DEBUG] Compiling pipeline on-demand: " << shader_name << std::endl;
        
        // メモリキャッシュのみ使用、直接コンパイル実行
        id<MTLLibrary> library = compile_and_cache_shader(shader_name);
        
        if (!library) {
            std::cerr << "[ERROR] Failed to compile shader: " << shader_name << std::endl;
            return nil;
        }
        
        // 3. ライブラリから関数取得
        NSString* function_name = [NSString stringWithUTF8String:func_name.c_str()];
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        
        if (!function) {
            std::cerr << "[ERROR] Function not found in shader: " << shader_name << std::endl;
            return nil;
        }
        
        // 4. パイプライン作成
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [pimpl_->device newComputePipelineStateWithFunction:function error:&error];
        
        if (!pipeline || error) {
            std::cerr << "[ERROR] Failed to create pipeline for: " << shader_name;
            if (error) {
                std::cerr << ", error: " << [[error localizedDescription] UTF8String];
            }
            std::cerr << std::endl;
            return nil;
        }
        
        // 5. キャッシュに保存
        pipeline_cache[func_name] = pipeline;
        std::cout << "[DEBUG] Pipeline compiled and cached: " << shader_name << std::endl;
        
        return pipeline;
    }
}

// ファイルキャッシュ機能削除済み - メモリキャッシュのみ使用
id<MTLLibrary> GPUAccelerator::compile_and_cache_shader(const std::string& shader_name) {
    @autoreleasepool {
        // 1. シェーダーファイル読み込み
        std::string shader_source = load_shader_file(shader_name + ".metal");
        if (shader_source.empty()) {
            std::cerr << "[ERROR] Failed to load shader source: " << shader_name << std::endl;
            return nil;
        }
        
        // 2. ヘッダーファイルを結合
        std::string types_header = load_shader_file("shader_types.h");
        std::string common_header = load_shader_file("shader_common.h");
        std::string constants_header = load_shader_file("constants.h");
        
        if (types_header.empty() || common_header.empty() || constants_header.empty()) {
            std::cerr << "[ERROR] Failed to load shader headers for: " << shader_name << std::endl;
            return nil;
        }
        
        // 3. #include文を削除してヘッダー結合
        size_t pos;
        while ((pos = shader_source.find("#include \"shader_types.h\"")) != std::string::npos) {
            shader_source.erase(pos, strlen("#include \"shader_types.h\""));
        }
        while ((pos = shader_source.find("#include \"shader_common.h\"")) != std::string::npos) {
            shader_source.erase(pos, strlen("#include \"shader_common.h\""));
        }
        while ((pos = shader_source.find("#include \"constants.h\"")) != std::string::npos) {
            shader_source.erase(pos, strlen("#include \"constants.h\""));
        }
        while ((pos = shader_source.find("#include \"constants.h\"")) != std::string::npos) {
            shader_source.erase(pos, strlen("#include \"constants.h\""));
        }
        
        std::string combined_source = types_header + "\n" + common_header + "\n" + constants_header + "\n" + shader_source;
        
        // 4. Metalコンパイル（最適化オプション付き）
        NSString* ns_source = [NSString stringWithUTF8String:combined_source.c_str()];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        
        // MTLLanguageVersionの互換性対応
        if (@available(macOS 11.0, *)) {
            options.languageVersion = MTLLanguageVersion2_3;
        } else {
            options.languageVersion = MTLLanguageVersion2_0; // macOS 10.15+対応
        }
        
        // 最適化オプション設定
        options.fastMathEnabled = YES;  // 高速数学演算
        
        // プリプロセッサ定義で最適化マクロを追加
        NSMutableDictionary* preprocessorMacros = [[NSMutableDictionary alloc] init];
        preprocessorMacros[@"METAL_OPTIMIZED"] = @"1";
        preprocessorMacros[@"FAST_MATH"] = @"1";
        preprocessorMacros[@"APPLE_M1_OPTIMIZED"] = @"1";
        options.preprocessorMacros = preprocessorMacros;
        
        NSError* error = nil;
        id<MTLLibrary> library = [pimpl_->device newLibraryWithSource:ns_source options:options error:&error];
        
        if (error || !library) {
            std::cerr << "[ERROR] Failed to compile shader: " << shader_name;
            if (error) {
                std::cerr << ", error: " << [[error localizedDescription] UTF8String];
            }
            std::cerr << std::endl;
            return nil;
        }
        
        // Note: ファイルキャッシュは未実装、メモリキャッシュのみ使用
        std::cout << "[DEBUG] Shader compiled (memory cache only): " << shader_name << std::endl;
        
        return library;
    }
}

std::string GPUAccelerator::load_shader_file(const std::string& filename) {
    std::vector<std::string> possible_paths = {
        std::string("metal/") + filename,
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

#endif // __OBJC__

} // namespace libraw_enhanced