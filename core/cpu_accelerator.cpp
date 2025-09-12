#include "cpu_accelerator.h"
#include "accelerator.h"
#include "constants.h"
#include "metal/shader_common.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <memory>
#include <vector>
#include <array>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#endif

namespace libraw_enhanced {
                         
CPUAccelerator::CPUAccelerator() {}
CPUAccelerator::~CPUAccelerator() = default;

bool CPUAccelerator::initialize() {
    device_name_ = "Apple Silicon CPU";
    initialized_ = true;
    std::cout << "⚡ CPU vectorization initialized on: " << device_name_ << std::endl;
    return true;
}

bool CPUAccelerator::is_available() const { return initialized_; }
void CPUAccelerator::release_resources() { initialized_ = false; }

//===================================================================
// ホワイトバランス
//===================================================================

bool CPUAccelerator::apply_white_balance(const ImageBuffer& raw_buffer,
                                        ImageBufferFloat& rgb_buffer,
                                        const float wb_multipliers[4],
                                        uint32_t filters,
                                        const char xtrans[6][6]) {

    if (!initialized_) return false;

    // 引数チェック
    if (!rgb_buffer.is_valid() || !rgb_buffer.is_valid()) {
        std::cerr << "❌ Invalid or mismatched buffers for white balance" << std::endl;
        return false;
    }
    
    if (filters == FILTERS_XTRANS) {
        apply_wb_xtrans(raw_buffer, rgb_buffer, wb_multipliers, xtrans);
    } else {
        apply_wb_bayer(raw_buffer, rgb_buffer, wb_multipliers, filters);
    }

    return true;
}

// Bayer CFA white balance implementation
void CPUAccelerator::apply_wb_bayer(
    const ImageBuffer& raw_buffer,
    ImageBufferFloat& rgb_buffer,
    const float wb_multipliers[4],
    uint32_t filters
) {
    const size_t total_pixels = raw_buffer.width * raw_buffer.height;
    
    // Process each pixel in the raw buffer
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++) {
        int row = pixel_idx / raw_buffer.width;
        int col = pixel_idx % raw_buffer.width;
        
        // Get color channel for this pixel position using LibRaw's fcol logic
        int color_channel = (filters >> ((((row) << 1 & 14) | ((col) & 1)) << 1)) & 3;
        
        // Apply white balance multiplier to the native color channel
        uint16_t original_value = raw_buffer.image[pixel_idx][color_channel];
        float adjusted_value = original_value * wb_multipliers[color_channel];

        if (color_channel == 3) {
            rgb_buffer.image[pixel_idx][1] = adjusted_value;
        } else {
            rgb_buffer.image[pixel_idx][color_channel] = adjusted_value;
        }
    }
    
    std::cout << "✅ Bayer WB process completed" << std::endl;
}

// X-Trans CFA white balance implementation
void CPUAccelerator::apply_wb_xtrans(
    const ImageBuffer& raw_buffer,
    ImageBufferFloat& rgb_buffer,
    const float wb_multipliers[4],
    const char xtrans[6][6]
) {
    const size_t total_pixels = raw_buffer.width * raw_buffer.height;
    
    // Process each pixel in the raw buffer
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++) {
        int row = pixel_idx / raw_buffer.width;
        int col = pixel_idx % raw_buffer.width;
        
        // Get color channel using X-Trans pattern
        int color_channel = xtrans[row % 6][col % 6];
        
        // Apply white balance multiplier to the native color channel
        uint16_t original_value = raw_buffer.image[pixel_idx][color_channel];
        float adjusted_value = original_value * wb_multipliers[color_channel];
                    
        rgb_buffer.image[pixel_idx][color_channel] = adjusted_value;
    }
    
    std::cout << "✅ X-Trans WB process completed" << std::endl;
}

//===================================================================
// Pre-interpolation
//===================================================================

bool CPUAccelerator::pre_interpolate(ImageBufferFloat& rgb_buffer, uint32_t filters, const char (&xtrans)[6][6], bool half_size) {
    if (!rgb_buffer.is_valid()) return false;
    float (*image)[3] = rgb_buffer.image;
    size_t width = rgb_buffer.width, height = rgb_buffer.height;
    std::cout << "📋 Pre-interpolation: " << width << "x" << height << " (filters=" << filters << ", half_size=" << half_size << ")" << std::endl;

    if (half_size && filters == FILTERS_XTRANS) {
        size_t row, col;
//      for (row = 0; row < 3; ++row) for (col = 1; col < 4; ++col) if (!(image[row * width + col][0] | image[row * width + col][2])) goto break_outer;
        for (row = 0; row < 3; ++row) for (col = 1; col < 4; ++col) if (image[row * width + col][0] != 0.f && image[row * width + col][2] != 0.f) goto break_outer;
        break_outer:
        for (; row < height; row += 3)
            for (size_t col_start = (col - 1) % 3 + 1; col_start < width - 1; col_start += 3)
                for (int c = 0; c < 3; c += 2)
                    image[row * width + col_start][c] = (image[row * width + col_start - 1][c] + image[row * width + col_start + 1][c]) * 0.5f;
    }
    return true;
}

//===================================================================
// Demosaic function for linear interpolation
//===================================================================

bool CPUAccelerator::demosaic_bayer_linear(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, uint16_t maximum_value) {
    if (!initialized_) return false;

    const size_t width = raw_buffer.width, height = raw_buffer.height;

    // まず RAW データをコピー（各ピクセルのネイティブカラーのみ）
    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col++) {
            size_t idx = row * width + col;
            int color = fcol_bayer(row, col, filters);
            
            // RAW値を取得して正規化
            float raw_val = (float)raw_buffer.image[idx][color] / (float)maximum_value;
            
            // 初期化：全チャンネルを0にセット
            rgb_buffer.image[idx][0] = 0.0f;
            rgb_buffer.image[idx][1] = 0.0f;
            rgb_buffer.image[idx][2] = 0.0f;
            
            // ネイティブカラーをセット
            rgb_buffer.image[idx][color] = raw_val;
        }
    }

    // シンプルなバイリニア補間
    for (size_t row = 1; row < height - 1; row++) {
        for (size_t col = 1; col < width - 1; col++) {
            size_t idx = row * width + col;
            int color = fcol_bayer(row, col, filters);
            
            // 各ピクセルで欠けている色を補間
            for (int c = 0; c < 3; c++) {
                if (c == color) continue; // ネイティブカラーはスキップ
                
                float sum = 0.0f;
                int count = 0;
                
                // 上下左右の隣接ピクセルをチェック
                for (int dy = -1; dy <= 1; dy += 2) {
                    for (int dx = -1; dx <= 1; dx += 2) {
                        int ny = row + dy;
                        int nx = col + dx;
                        if (ny >= 0 && ny < (int)height && nx >= 0 && nx < (int)width) {
                            int neighbor_color = fcol_bayer(ny, nx, filters);
                            if (neighbor_color == c) {
                                size_t neighbor_idx = ny * width + nx;
                                sum += (float)raw_buffer.image[neighbor_idx][c] / (float)maximum_value;
                                count++;
                            }
                        }
                    }
                }
                
                // 対角線上にない場合は、直交方向もチェック
                if (count == 0) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dy == 0 && dx == 0) continue;
                            int ny = row + dy;
                            int nx = col + dx;
                            if (ny >= 0 && ny < (int)height && nx >= 0 && nx < (int)width) {
                                int neighbor_color = fcol_bayer(ny, nx, filters);
                                if (neighbor_color == c) {
                                    size_t neighbor_idx = ny * width + nx;
                                    sum += (float)raw_buffer.image[neighbor_idx][c] / (float)maximum_value;
                                    count++;
                                }
                            }
                        }
                    }
                }
                
                // 平均値を設定
                if (count > 0) {
                    rgb_buffer.image[idx][c] = sum / count;
                }
            }
        }
    }

    // 境界補間 - ENABLED for fair CPU vs GPU comparison
    border_interpolate(rgb_buffer, filters, 1);
    
    return true;
}


//===================================================================
// Demosaic function for AAHD (Adaptive AHD) demosaic
//===================================================================

typedef uint16_t ushort3[3];
typedef int int3[3];

// Direction macros (LibRaw exact)
#define Pnw (-1 - nr_width)
#define Pn (-nr_width)
#define Pne (+1 - nr_width)
#define Pw (-1)
#define Pe (+1)
#define Psw (-1 + nr_width)
#define Ps (+nr_width)
#define Pse (+1 + nr_width)

class AAHD_Processor {
private:
    int nr_height, nr_width;
    static const int nr_margin = 4;
    static const int Thot = 4;
    static const int Tdead = 4;
    static const int OverFraction = 8;
    
    ushort3 *rgb_ahd[2];
    int3 *yuv[2];
    char *ndir, *homo[2];
    uint16_t channel_maximum[3], channels_max;
    uint16_t channel_minimum[3];
    
    // YUV coefficients (LibRaw exact - Rec. 2020)
    static const float yuv_coeff[3][3];
    static float gammaLUT[0x10000];
    float yuv_cam[3][3];
    
    const ImageBuffer& raw_buffer_;
    ImageBufferFloat& rgb_buffer_;
    uint32_t filters_;
    const uint16_t maximum_value_;
    
    enum {
        HVSH = 1,
        HOR = 2,
        VER = 4,
        HORSH = HOR | HVSH,
        VERSH = VER | HVSH,
        HOT = 8
    };
    
    static inline float calc_dist(int c1, int c2) {
        return c1 > c2 ? (float)c1 / c2 : (float)c2 / c1;
    }
    
    int Y(ushort3 &rgb) {
        return yuv_cam[0][0] * rgb[0] + yuv_cam[0][1] * rgb[1] + yuv_cam[0][2] * rgb[2];
    }
    
    int U(ushort3 &rgb) {
        return yuv_cam[1][0] * rgb[0] + yuv_cam[1][1] * rgb[1] + yuv_cam[1][2] * rgb[2];
    }
    
    int V(ushort3 &rgb) {
        return yuv_cam[2][0] * rgb[0] + yuv_cam[2][1] * rgb[1] + yuv_cam[2][2] * rgb[2];
    }
    
    inline int nr_offset(int row, int col) {
        return (row * nr_width + col);
    }
    
public:
    AAHD_Processor(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, uint16_t maximum_value) 
        : raw_buffer_(raw_buffer), rgb_buffer_(rgb_buffer), filters_(filters), maximum_value_(maximum_value) {
        
        nr_height = raw_buffer.height + nr_margin * 2;
        nr_width = raw_buffer.width + nr_margin * 2;
        
        // Allocate all AAHD buffers in one block (LibRaw exact)
        rgb_ahd[0] = (ushort3 *)calloc(nr_height * nr_width,
                                     (sizeof(ushort3) * 2 + sizeof(int3) * 2 + 3));
        if (!rgb_ahd[0]) {
            throw std::bad_alloc();
        }
        
        rgb_ahd[1] = rgb_ahd[0] + nr_height * nr_width;
        yuv[0] = (int3 *)(rgb_ahd[1] + nr_height * nr_width);
        yuv[1] = yuv[0] + nr_height * nr_width;
        ndir = (char *)(yuv[1] + nr_height * nr_width);
        homo[0] = ndir + nr_height * nr_width;
        homo[1] = homo[0] + nr_height * nr_width;
        
        // Initialize channel statistics
        channel_maximum[0] = channel_maximum[1] = channel_maximum[2] = 0;
        channel_minimum[0] = raw_buffer.image[0][0];
        channel_minimum[1] = raw_buffer.image[0][1];
        channel_minimum[2] = raw_buffer.image[0][2];
        
        // Initialize YUV transformation matrix
        // Using identity rgb_cam for now - in real LibRaw this comes from color correction
        float rgb_cam[3][4] = {
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f}
        };
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                yuv_cam[i][j] = 0;
                for (int k = 0; k < 3; ++k) {
                    yuv_cam[i][j] += yuv_coeff[i][k] * rgb_cam[k][j];
                }
            }
        }
        
        // Initialize gamma LUT (LibRaw exact)
        if (gammaLUT[0] < -0.1f) {
            for (int i = 0; i < 0x10000; i++) {
                float r = (float)i / 0x10000;
                gammaLUT[i] = 0x10000 * (r < 0.0181 ? 4.5f * r : 1.0993f * pow(r, 0.45f) - 0.0993f);
            }
        }
        
        // Copy raw data to AAHD buffers with margin (LibRaw exact)
        size_t iwidth = raw_buffer.width;
        size_t iheight = raw_buffer.height;
        
        for (size_t i = 0; i < iheight; ++i) {
            int col_cache[48];
            for (int j = 0; j < 48; ++j) {
                int c = fcol_bayer(static_cast<int>(i), j, filters_);
                if (c == 3) c = 1; // Map G2 to G1
                col_cache[j] = c;
            }
            
            int moff = nr_offset(static_cast<int>(i) + nr_margin, nr_margin);
            for (size_t j = 0; j < iwidth; ++j, ++moff) {
                int c = col_cache[j % 48];
                
                uint16_t d = 0;
                uint16_t *pixel = raw_buffer.image[i * iwidth + j];
                
                for (int ch = 0; ch < 4; ch++) {
                    if (pixel[ch] != 0) {
                        d = pixel[ch];
                        break;
                    }
                }
                
                if (i < 3 && j < 3) {
                    std::cout << "Pixel(" << i << "," << j << ") color=" << c << " RAW_value=" << d;
                    std::cout << " RGBG=[" << pixel[0] << "," << pixel[1] << "," << pixel[2] << "," << pixel[3] << "]" << std::endl;
                }
                
                if (d != 0) {
                    if (channel_maximum[c] < d) channel_maximum[c] = d;
                    if (channel_minimum[c] > d) channel_minimum[c] = d;
                    rgb_ahd[1][moff][c] = rgb_ahd[0][moff][c] = d;
                }
            }
        }
        
        channels_max = std::max({channel_maximum[0], channel_maximum[1], channel_maximum[2]});
        
        std::cout << "✅ AAHD_Processor initialized: " << nr_width << "x" << nr_height 
                  << " (margin=" << nr_margin << ")" << std::endl;
    }
    
    ~AAHD_Processor() {
        free(rgb_ahd[0]);
    }
    
    void hide_hots();
    void make_ahd_greens();
    void make_ahd_gline(int i);
    void make_ahd_rb();
    void make_ahd_rb_hv(int i);
    void make_ahd_rb_last(int i);
    void evaluate_ahd();
    void combine_image();
    void refine_hv_dirs();
    void refine_hv_dirs(int i, int js);
    void refine_ihv_dirs(int i);
};

// Static member definitions
const float AAHD_Processor::yuv_coeff[3][3] = {
    {+0.2627f, +0.6780f, +0.0593f},
    {-0.13963f, -0.36037f, +0.5f},
    {+0.5034f, -0.4629f, -0.0405f}
};

float AAHD_Processor::gammaLUT[0x10000] = {-1.f};

void AAHD_Processor::hide_hots() {
    
    int iwidth = raw_buffer_.width;
    for (size_t i = 0; i < raw_buffer_.height; ++i) {
        int js = fcol_bayer(static_cast<int>(i), 0, filters_) & 1;
        int kc = fcol_bayer(static_cast<int>(i), js, filters_);
        
        int moff = nr_offset(static_cast<int>(i) + nr_margin, nr_margin + js);
        for (int j = js; j < iwidth; j += 2, moff += 2) {
            ushort3 *rgb = &rgb_ahd[0][moff];
            int c = rgb[0][kc];
            
            if ((c > rgb[2 * Pe][kc] && c > rgb[2 * Pw][kc] && c > rgb[2 * Pn][kc] &&
                 c > rgb[2 * Ps][kc] && c > rgb[Pe][1] && c > rgb[Pw][1] &&
                 c > rgb[Pn][1] && c > rgb[Ps][1]) ||
                (c < rgb[2 * Pe][kc] && c < rgb[2 * Pw][kc] && c < rgb[2 * Pn][kc] &&
                 c < rgb[2 * Ps][kc] && c < rgb[Pe][1] && c < rgb[Pw][1] &&
                 c < rgb[Pn][1] && c < rgb[Ps][1])) {
                
                int chot = c >> Thot;
                int cdead = c << Tdead;
                int avg = 0;
                
                for (int k = -2; k < 3; k += 2) {
                    for (int m = -2; m < 3; m += 2) {
                        if (m == 0 && k == 0) continue;
                        avg += rgb[nr_offset(k, m)][kc];
                    }
                }
                avg /= 8;
                
                if (chot > avg || cdead < avg) {
                    ndir[moff] |= HOT;
                    
                    int dh = std::abs(rgb[2 * Pw][kc] - rgb[2 * Pe][kc]) +
                             std::abs(rgb[Pw][1] - rgb[Pe][1]) +
                             std::abs(rgb[Pw][1] - rgb[Pe][1] + rgb[2 * Pe][kc] - rgb[2 * Pw][kc]);
                    int dv = std::abs(rgb[2 * Pn][kc] - rgb[2 * Ps][kc]) +
                             std::abs(rgb[Pn][1] - rgb[Ps][1]) +
                             std::abs(rgb[Pn][1] - rgb[Ps][1] + rgb[2 * Ps][kc] - rgb[2 * Pn][kc]);
                    
                    int d = (dv > dh) ? Pw : Pn;
                    
                    rgb_ahd[1][moff][kc] = rgb[0][kc] = (rgb[+2 * d][kc] + rgb[-2 * d][kc]) / 2;
                }
            }
        }
    }
}

void AAHD_Processor::make_ahd_greens() {
    for (size_t i = 0; i < raw_buffer_.height; ++i) {
        make_ahd_gline(static_cast<int>(i));
    }
}

void AAHD_Processor::make_ahd_gline(int i) {
    int js = fcol_bayer(i, 0, filters_) & 1;
    int kc = fcol_bayer(i, js, filters_);
    int hvdir[2] = {Pe, Ps};
    
    for (int d = 0; d < 2; ++d) {
        int moff = nr_offset(i + nr_margin, nr_margin + js);
        for (size_t j = js; j < raw_buffer_.width; j += 2, moff += 2) {
            ushort3 *cnr = &rgb_ahd[d][moff];
            
            int h1 = 2 * cnr[-hvdir[d]][1] - int(cnr[-2 * hvdir[d]][kc] + cnr[0][kc]);
            int h2 = 2 * cnr[+hvdir[d]][1] - int(cnr[+2 * hvdir[d]][kc] + cnr[0][kc]);
            int eg = cnr[0][kc] + (h1 + h2) / 4;
            
            int min_val = std::min(cnr[-hvdir[d]][1], cnr[+hvdir[d]][1]);
            int max_val = std::max(cnr[-hvdir[d]][1], cnr[+hvdir[d]][1]);
            min_val -= min_val / OverFraction;
            max_val += max_val / OverFraction;
            
            if (eg < min_val) eg = min_val - sqrtf(min_val - eg);
            else if (eg > max_val) eg = max_val + sqrtf(eg - max_val);
            
            if (eg > channel_maximum[1]) eg = channel_maximum[1];
            else if (eg < channel_minimum[1]) eg = channel_minimum[1];
            
            cnr[0][1] = eg;
        }
    }
}

void AAHD_Processor::make_ahd_rb() {
    for (size_t i = 0; i < raw_buffer_.height; ++i) {
        make_ahd_rb_hv(static_cast<int>(i));
    }
    for (size_t i = 0; i < raw_buffer_.height; ++i) {
        make_ahd_rb_last(static_cast<int>(i));
    }
}

void AAHD_Processor::make_ahd_rb_hv(int i) {
    int iwidth = raw_buffer_.width;
    int js = fcol_bayer(i, 0, filters_) & 1;
    int kc = fcol_bayer(i, js, filters_);
    js ^= 1;
    int hvdir[2] = {Pe, Ps};
    
    for (int j = js; j < iwidth; j += 2) {
        int moff = nr_offset(i + nr_margin, j + nr_margin);
        for (int d = 0; d < 2; ++d) {
            ushort3 *cnr = &rgb_ahd[d][moff];
            int c = kc ^ (d << 1);
            int eg = cnr[0][1] + (cnr[-hvdir[d]][c] - cnr[-hvdir[d]][1] + cnr[+hvdir[d]][c] - cnr[+hvdir[d]][1]) / 2;
            
            if (eg > channel_maximum[c]) eg = channel_maximum[c];
            else if (eg < channel_minimum[c]) eg = channel_minimum[c];
            cnr[0][c] = eg;
        }
    }
}

void AAHD_Processor::make_ahd_rb_last(int i) {
    int iwidth = raw_buffer_.width;
    int js = fcol_bayer(i, 0, filters_) & 1;
    int kc = fcol_bayer(i, js, filters_);
    int dirs[2][3] = {{Pnw, Pn, Pne}, {Pnw, Pw, Psw}};
    int moff = nr_offset(i + nr_margin, nr_margin);
    
    for (int j = 0; j < iwidth; ++j) {
        for (int d = 0; d < 2; ++d) {
            ushort3 *cnr = &rgb_ahd[d][moff + j];
            int c = kc ^ 2;
            if ((j & 1) != js) c ^= d << 1;
            
            int bh = 0, bk = 0, bgd = 0;
            for (int k = 0; k < 3; ++k) {
                for (int h = 0; h < 3; ++h) {
                    int gd = std::abs(2 * cnr[0][1] - (cnr[+dirs[d][k]][1] + cnr[-dirs[d][h]][1])) +
                             std::abs(cnr[+dirs[d][k]][c] - cnr[-dirs[d][h]][c]) / 4 +
                             std::abs(cnr[+dirs[d][k]][c] - cnr[+dirs[d][k]][1] + cnr[-dirs[d][h]][1] - cnr[-dirs[d][h]][c]) / 4;
                    if (bgd == 0 || gd < bgd) {
                        bgd = gd;
                        bh = h;
                        bk = k;
                    }
                }
            }
            int eg = cnr[0][1] + (cnr[+dirs[d][bk]][c] - cnr[+dirs[d][bk]][1] + cnr[-dirs[d][bh]][c] - cnr[-dirs[d][bh]][1]) / 2;
            if (eg > channel_maximum[c]) eg = channel_maximum[c];
            else if (eg < channel_minimum[c]) eg = channel_minimum[c];
            cnr[0][c] = eg;
        }
    }
}

void AAHD_Processor::evaluate_ahd() {
    int hvdir[4] = {Pw, Pe, Pn, Ps};
    
    for (int d = 0; d < 2; ++d) {
        for (int i = 0; i < nr_width * nr_height; ++i) {
            ushort3 rgb;
            for (int c = 0; c < 3; ++c) rgb[c] = gammaLUT[rgb_ahd[d][i][c]];
            yuv[d][i][0] = Y(rgb);
            yuv[d][i][1] = U(rgb);
            yuv[d][i][2] = V(rgb);
        }
    }
    
    for (size_t i = 0; i < raw_buffer_.height; ++i) {
        int moff = nr_offset(static_cast<int>(i) + nr_margin, nr_margin);
        for (size_t j = 0; j < raw_buffer_.width; ++j, ++moff) {
            float ydiff[2][4];
            int uvdiff[2][4];
            
            for (int d = 0; d < 2; ++d) {
                for (int k = 0; k < 4; k++) {
                    ydiff[d][k] = std::abs(yuv[d][moff][0] - yuv[d][moff + hvdir[k]][0]);
                    uvdiff[d][k] = SQR(yuv[d][moff][1] - yuv[d][moff + hvdir[k]][1]) +
                                   SQR(yuv[d][moff][2] - yuv[d][moff + hvdir[k]][2]);
                }
            }
            float toth = ydiff[0][0] + ydiff[0][1] + (uvdiff[0][0] + uvdiff[0][1]) / 1000.0f;
            float totv = ydiff[1][2] + ydiff[1][3] + (uvdiff[1][2] + uvdiff[1][3]) / 1000.0f;
            
            ndir[moff] = 0;
            if (totv < toth * 1.2f) ndir[moff] |= VER;
            if (toth < totv * 1.2f) ndir[moff] |= HOR;
            if (ndir[moff] == 0) ndir[moff] = HOR | VER;
        }
    }
}

void AAHD_Processor::refine_hv_dirs() {
    for (size_t i = 0; i < raw_buffer_.height; ++i) {
        refine_ihv_dirs(static_cast<int>(i));
    }
}

void AAHD_Processor::refine_ihv_dirs(int i) {
    for (int js = 0; js < 2; ++js) {
        refine_hv_dirs(i, js);
    }
}

void AAHD_Processor::refine_hv_dirs(int i, int js) {
    int moff = nr_offset(i + nr_margin, nr_margin + js);
    for (size_t j = js; j < raw_buffer_.width; j += 2, moff += 2) {
        int nv = ((ndir[moff + Pn] & VER) + (ndir[moff + Ps] & VER) +
                  (ndir[moff + Pw] & VER) + (ndir[moff + Pe] & VER)) / VER;
        int nh = ((ndir[moff + Pn] & HOR) + (ndir[moff + Ps] & HOR) +
                  (ndir[moff + Pw] & HOR) + (ndir[moff + Pe] & HOR)) / HOR;
        bool codir = (ndir[moff] & VER) ? ((ndir[moff + Pn] & VER) || (ndir[moff + Ps] & VER))
                                        : ((ndir[moff + Pw] & HOR) || (ndir[moff + Pe] & HOR));
        if ((ndir[moff] & VER) && (nh > 2 && !codir)) ndir[moff] = HOR;
        if ((ndir[moff] & HOR) && (nv > 2 && !codir)) ndir[moff] = VER;
    }
}

void AAHD_Processor::combine_image() {
    for (size_t i = 0; i < raw_buffer_.height; ++i) {
        for (size_t j = 0; j < raw_buffer_.width; ++j) {
            int moff = nr_offset(static_cast<int>(i) + nr_margin, static_cast<int>(j) + nr_margin);
            size_t img_off = i * raw_buffer_.width + j;
            int d = (ndir[moff] & VER) ? 1 : 0;
            
            rgb_buffer_.image[img_off][0] = (float)rgb_ahd[d][moff][0] / maximum_value_;
            rgb_buffer_.image[img_off][1] = (float)rgb_ahd[d][moff][1] / maximum_value_;
            rgb_buffer_.image[img_off][2] = (float)rgb_ahd[d][moff][2] / maximum_value_;
        }
    }
}

bool CPUAccelerator::demosaic_bayer_aahd(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, uint16_t maximum_value) {

    if (!initialized_) return false;
        
    AAHD_Processor aahd_proc(raw_buffer, rgb_buffer, filters, maximum_value);
    aahd_proc.hide_hots();
    aahd_proc.make_ahd_greens(); 
    aahd_proc.make_ahd_rb();
    aahd_proc.evaluate_ahd();
    aahd_proc.refine_hv_dirs();
    aahd_proc.combine_image();
    
    // ✨デモザイク後境界補間✨: 最終処理で境界を修正
    border_interpolate(rgb_buffer, filters, 2);
    
    return true;
}

//===================================================================
// DCB (Demosaic Color Balance) processor
//===================================================================

namespace { 

class DCB_Processor {
private:
    int width, height;
    uint32_t filters;
    float (*image)[4];
    uint16_t maximum_val_;

    float CLIP_FLOAT(float val) const {
        return val > (float)maximum_val_ ? (float)maximum_val_ : (val < 0.0f ? 0.0f : val);
    }

public:
    uint16_t CLIP_16BIT(float val) const {
        if (val < 0.0f) return 0;
        if (val > (float)maximum_val_) return maximum_val_;
        return static_cast<uint16_t>(val + 0.5f);
    }

    DCB_Processor(float (*img_buffer)[4], int w, int h, uint32_t f, uint16_t max_val)
        : width(w), height(h), filters(f), image(img_buffer), maximum_val_(max_val) {}

    // (dcb_* 関連の全関数は、コンパイラの自動ベクトル化を妨げないよう、
    //  ループ内での関数呼び出しを避け、シンプルな記述に整理されています)
    void dcb_ver(float (*image3)[3]);
    void dcb_hor(float (*image2)[3]);
    void dcb_color();
    void dcb_color2(float (*image2)[3]);
    void dcb_color3(float (*image3)[3]);
    void dcb_decide(float (*image2)[3], float (*image3)[3]);
    void dcb_copy_to_buffer(float (*image2)[3]);
    void dcb_restore_from_buffer(float (*image2)[3]);
    void dcb_pp();
    void dcb_nyquist();
    void dcb_color_full();
    void dcb_map();
    void dcb_correction();
    void dcb_correction2();
    void dcb_refinement();
    void run_dcb(int iterations, bool dcb_enhance);
};


void DCB_Processor::dcb_ver(float (*image3)[3]) {
    for (int row = 2; row < height - 2; row++) {
        for (int col = 2 + (fcol_bayer(row, 2, filters) & 1), indx = row * width + col; col < width - 2; col += 2, indx += 2) {
            image3[indx][1] = CLIP_FLOAT((image[indx + width][1] + image[indx - width][1]) / 2.0f);
        }
    }
}

void DCB_Processor::dcb_hor(float (*image2)[3]) {
    for (int row = 2; row < height - 2; row++) {
        for (int col = 2 + (fcol_bayer(row, 2, filters) & 1), indx = row * width + col; col < width - 2; col += 2, indx += 2) {
            image2[indx][1] = CLIP_FLOAT((image[indx + 1][1] + image[indx - 1][1]) / 2.0f);
        }
    }
}

void DCB_Processor::dcb_color() {
    int u = width;
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1 + (fcol_bayer(row, 1, filters) & 1), indx = row * u + col; col < u - 1; col += 2, indx += 2) {
            int c = 2 - fcol_bayer(row, col, filters);
            image[indx][c] = CLIP_FLOAT((4 * image[indx][1] - image[indx + u + 1][1] - image[indx + u - 1][1] - image[indx - u + 1][1] - image[indx - u - 1][1] + 
                                         image[indx + u + 1][c] + image[indx + u - 1][c] + image[indx - u + 1][c] + image[indx - u - 1][c]) / 4.0f);
        }
    }
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1 + (fcol_bayer(row, 2, filters) & 1), indx = row * u + col; col < width - 1; col += 2, indx += 2) {
            int c = fcol_bayer(row, col + 1, filters);
            int d = 2 - c;
            image[indx][c] = CLIP_FLOAT((2 * image[indx][1] - image[indx + 1][1] - image[indx - 1][1] + image[indx + 1][c] + image[indx - 1][c]) / 2.0f);
            image[indx][d] = CLIP_FLOAT((2 * image[indx][1] - image[indx + u][1] - image[indx - u][1] + image[indx + u][d] + image[indx - u][d]) / 2.0f);
        }
    }
}

void DCB_Processor::dcb_color2(float (*image2)[3]) {
    int u = width;
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1 + (fcol_bayer(row, 1, filters) & 1), indx = row * u + col; col < u - 1; col += 2, indx += 2) {
            int c = 2 - fcol_bayer(row, col, filters);
            image2[indx][c] = CLIP_FLOAT((4 * image2[indx][1] - image2[indx + u + 1][1] - image2[indx + u - 1][1] - image2[indx - u + 1][1] - image2[indx - u - 1][1] + 
                                          image[indx + u + 1][c] + image[indx + u - 1][c] + image[indx - u + 1][c] + image[indx - u - 1][c]) / 4.0f);
        }
    }
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1 + (fcol_bayer(row, 2, filters) & 1), indx = row * u + col; col < width - 1; col += 2, indx += 2) {
            int c = fcol_bayer(row, col + 1, filters);
            int d = 2 - c;
            image2[indx][c] = CLIP_FLOAT((image[indx + 1][c] + image[indx - 1][c]) / 2.0f);
            image2[indx][d] = CLIP_FLOAT((2 * image2[indx][1] - image2[indx + u][1] - image2[indx - u][1] + image[indx + u][d] + image[indx - u][d]) / 2.0f);
        }
    }
}

void DCB_Processor::dcb_color3(float (*image3)[3]) {
    int u = width;
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1 + (fcol_bayer(row, 1, filters) & 1), indx = row * u + col; col < u - 1; col += 2, indx += 2) {
            int c = 2 - fcol_bayer(row, col, filters);
            image3[indx][c] = CLIP_FLOAT((4 * image3[indx][1] - image3[indx + u + 1][1] - image3[indx + u - 1][1] - image3[indx - u + 1][1] - image3[indx - u - 1][1] +
                                          image[indx + u + 1][c] + image[indx + u - 1][c] + image[indx - u + 1][c] + image[indx - u - 1][c]) / 4.0f);
        }
    }
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1 + (fcol_bayer(row, 2, filters) & 1), indx = row * u + col; col < width - 1; col += 2, indx += 2) {
            int c = fcol_bayer(row, col + 1, filters);
            int d = 2 - c;
            image3[indx][c] = CLIP_FLOAT((2 * image3[indx][1] - image3[indx + 1][1] - image3[indx - 1][1] + image[indx + 1][c] + image[indx - 1][c]) / 2.0f);
            image3[indx][d] = CLIP_FLOAT((image[indx + u][d] + image[indx - u][d]) / 2.0f);
        }
    }
}

void DCB_Processor::dcb_decide(float (*image2)[3], float (*image3)[3]) {
    int v = 2 * width;
    for (int row = 2; row < height - 2; row++) {
        for (int col = 2 + (fcol_bayer(row, 2, filters) & 1), indx = row * width + col; col < width - 2; col += 2, indx += 2) {
            int c = fcol_bayer(row, col, filters);
            int d = std::abs(c - 2);
            float current = std::max({image[indx + v][c], image[indx - v][c], image[indx - 2][c], image[indx + 2][c]}) -
                            std::min({image[indx + v][c], image[indx - v][c], image[indx - 2][c], image[indx + 2][c]}) +
                            std::max({image[indx + 1 + width][d], image[indx + 1 - width][d], image[indx - 1 + width][d], image[indx - 1 - width][d]}) -
                            std::min({image[indx + 1 + width][d], image[indx + 1 - width][d], image[indx - 1 + width][d], image[indx - 1 - width][d]});
            float current2 = std::max({image2[indx + v][d], image2[indx - v][d], image2[indx - 2][d], image2[indx + 2][d]}) -
                             std::min({image2[indx + v][d], image2[indx - v][d], image2[indx - 2][d], image2[indx + 2][d]}) +
                             std::max({image2[indx + 1 + width][c], image2[indx + 1 - width][c], image2[indx - 1 + width][c], image2[indx - 1 - width][c]}) -
                             std::min({image2[indx + 1 + width][c], image2[indx + 1 - width][c], image2[indx - 1 + width][c], image2[indx - 1 - width][c]});
            float current3 = std::max({image3[indx + v][d], image3[indx - v][d], image3[indx - 2][d], image3[indx + 2][d]}) -
                             std::min({image3[indx + v][d], image3[indx - v][d], image3[indx - 2][d], image3[indx + 2][d]}) +
                             std::max({image3[indx + 1 + width][c], image3[indx + 1 - width][c], image3[indx - 1 + width][c], image3[indx - 1 - width][c]}) -
                             std::min({image3[indx + 1 + width][c], image3[indx + 1 - width][c], image3[indx - 1 + width][c], image3[indx - 1 - width][c]});
            image[indx][1] = (std::abs(current - current2) < std::abs(current - current3)) ? image2[indx][1] : image3[indx][1];
        }
    }
}

void DCB_Processor::dcb_copy_to_buffer(float (*image2)[3]) {
    for (int i = 0; i < height * width; i++) {
        image2[i][0] = image[i][0];
        image2[i][2] = image[i][2];
    }
}

void DCB_Processor::dcb_restore_from_buffer(float (*image2)[3]) {
    for (int i = 0; i < height * width; i++) {
        image[i][0] = image2[i][0];
        image[i][2] = image2[i][2];
    }
}

void DCB_Processor::dcb_pp() {
    int u = width;
    for (int row = 2; row < height - 2; row++) {
        for (int col = 2, indx = row * u + col; col < width - 2; col++, indx++) {
            float r1 = (image[indx - 1][0] + image[indx + 1][0] + image[indx - u][0] + image[indx + u][0] + image[indx - u - 1][0] + image[indx + u + 1][0] + image[indx - u + 1][0] + image[indx + u - 1][0]) / 8.0f;
            float g1 = (image[indx - 1][1] + image[indx + 1][1] + image[indx - u][1] + image[indx + u][1] + image[indx - u - 1][1] + image[indx + u + 1][1] + image[indx - u + 1][1] + image[indx + u - 1][1]) / 8.0f;
            float b1 = (image[indx - 1][2] + image[indx + 1][2] + image[indx - u][2] + image[indx + u][2] + image[indx - u - 1][2] + image[indx + u + 1][2] + image[indx - u + 1][2] + image[indx + u - 1][2]) / 8.0f;
            image[indx][0] = CLIP_FLOAT(r1 + (image[indx][1] - g1));
            image[indx][2] = CLIP_FLOAT(b1 + (image[indx][1] - g1));
        }
    }
}

void DCB_Processor::dcb_nyquist() {
    int v = 2 * width;
    for (int row = 2; row < height - 2; row++) {
        for (int col = 2 + (fcol_bayer(row, 2, filters) & 1), indx = row * width + col; col < width - 2; col += 2, indx += 2) {
            int c = fcol_bayer(row, col, filters);
            image[indx][1] = CLIP_FLOAT((image[indx + v][1] + image[indx - v][1] + image[indx - 2][1] + image[indx + 2][1]) / 4.0f +
                                        image[indx][c] - (image[indx + v][c] + image[indx - v][c] + image[indx - 2][c] + image[indx + 2][c]) / 4.0f);
        }
    }
}

void DCB_Processor::dcb_color_full() {
    int u = width, w = 3 * u;
    std::vector<float> chroma_vec(static_cast<size_t>(u) * height * 2);
    float (*chroma)[2] = reinterpret_cast<float(*)[2]>(chroma_vec.data());

    for (int row = 1; row < height - 1; row++) {
        for (int col = 1 + (fcol_bayer(row, 1, filters) & 1), indx = row * u + col; col < u - 1; col += 2, indx += 2) {
            int c = fcol_bayer(row, col, filters);
            chroma[indx][c / 2] = image[indx][c] - image[indx][1];
        }
    }
    for (int row = 3; row < height - 3; row++) {
        for (int col = 3 + (fcol_bayer(row, 1, filters) & 1), indx = row * u + col; col < u - 3; col += 2, indx += 2) {
            int c = 1 - fcol_bayer(row, col, filters) / 2;
            float f[4], g[4];
            f[0] = 1.0f / (1.0f + std::abs(chroma[indx - u - 1][c] - chroma[indx + u + 1][c]) + std::abs(chroma[indx - u - 1][c] - chroma[indx - w - 3][c]) + std::abs(chroma[indx + u + 1][c] - chroma[indx - w - 3][c]));
            f[1] = 1.0f / (1.0f + std::abs(chroma[indx - u + 1][c] - chroma[indx + u - 1][c]) + std::abs(chroma[indx - u + 1][c] - chroma[indx - w + 3][c]) + std::abs(chroma[indx + u - 1][c] - chroma[indx - w + 3][c]));
            f[2] = 1.0f / (1.0f + std::abs(chroma[indx + u - 1][c] - chroma[indx - u + 1][c]) + std::abs(chroma[indx + u - 1][c] - chroma[indx + w + 3][c]) + std::abs(chroma[indx - u + 1][c] - chroma[indx + w - 3][c]));
            f[3] = 1.0f / (1.0f + std::abs(chroma[indx + u + 1][c] - chroma[indx - u - 1][c]) + std::abs(chroma[indx + u + 1][c] - chroma[indx + w - 3][c]) + std::abs(chroma[indx - u - 1][c] - chroma[indx + w + 3][c]));
            g[0] = 1.325f * chroma[indx - u - 1][c] - 0.175f * chroma[indx - w - 3][c] - 0.075f * chroma[indx - w - 1][c] - 0.075f * chroma[indx - u - 3][c];
            g[1] = 1.325f * chroma[indx - u + 1][c] - 0.175f * chroma[indx - w + 3][c] - 0.075f * chroma[indx - w + 1][c] - 0.075f * chroma[indx - u + 3][c];
            g[2] = 1.325f * chroma[indx + u - 1][c] - 0.175f * chroma[indx + w - 3][c] - 0.075f * chroma[indx + w - 1][c] - 0.075f * chroma[indx + u - 3][c];
            g[3] = 1.325f * chroma[indx + u + 1][c] - 0.175f * chroma[indx + w + 3][c] - 0.075f * chroma[indx + w + 1][c] - 0.075f * chroma[indx + u + 3][c];
            chroma[indx][c] = (f[0] * g[0] + f[1] * g[1] + f[2] * g[2] + f[3] * g[3]) / (f[0] + f[1] + f[2] + f[3]);
        }
    }
    for (int row = 3; row < height - 3; row++) {
        for (int col = 3 + (fcol_bayer(row, 2, filters) & 1), indx = row * u + col; col < u - 3; col += 2, indx += 2) {
            int c = fcol_bayer(row, col + 1, filters) / 2;
            for (int d = 0; d <= 1; c = 1 - c, d++) {
                float f[4], g[4];
                f[0] = 1.0f / (1.0f + std::abs(chroma[indx - u][c] - chroma[indx + u][c]) + std::abs(chroma[indx - u][c] - chroma[indx - w][c]) + std::abs(chroma[indx + u][c] - chroma[indx - w][c]));
                f[1] = 1.0f / (1.0f + std::abs(chroma[indx + 1][c] - chroma[indx - 1][c]) + std::abs(chroma[indx + 1][c] - chroma[indx + 3][c]) + std::abs(chroma[indx - 1][c] - chroma[indx + 3][c]));
                f[2] = 1.0f / (1.0f + std::abs(chroma[indx - 1][c] - chroma[indx + 1][c]) + std::abs(chroma[indx - 1][c] - chroma[indx - 3][c]) + std::abs(chroma[indx + 1][c] - chroma[indx - 3][c]));
                f[3] = 1.0f / (1.0f + std::abs(chroma[indx + u][c] - chroma[indx - u][c]) + std::abs(chroma[indx + u][c] - chroma[indx + w][c]) + std::abs(chroma[indx - u][c] - chroma[indx + w][c]));
                g[0] = 0.875f * chroma[indx - u][c] + 0.125f * chroma[indx - w][c];
                g[1] = 0.875f * chroma[indx + 1][c] + 0.125f * chroma[indx + 3][c];
                g[2] = 0.875f * chroma[indx - 1][c] + 0.125f * chroma[indx - 3][c];
                g[3] = 0.875f * chroma[indx + u][c] + 0.125f * chroma[indx + w][c];
                chroma[indx][c] = (f[0] * g[0] + f[1] * g[1] + f[2] * g[2] + f[3] * g[3]) / (f[0] + f[1] + f[2] + f[3]);
            }
        }
    }
    for (int row = 6; row < height - 6; row++) {
        for (int col = 6, indx = row * u + col; col < width - 6; col++, indx++) {
            image[indx][0] = CLIP_FLOAT(chroma[indx][0] + image[indx][1]);
            image[indx][2] = CLIP_FLOAT(chroma[indx][1] + image[indx][1]);
            float g1 = std::min({image[indx + 1 + u][0], image[indx + 1 - u][0], image[indx - 1 + u][0], image[indx - 1 - u][0], image[indx - 1][0], image[indx + 1][0], image[indx - u][0], image[indx + u][0]});
            float g2 = std::max({image[indx + 1 + u][0], image[indx + 1 - u][0], image[indx - 1 + u][0], image[indx - 1 - u][0], image[indx - 1][0], image[indx + 1][0], image[indx - u][0], image[indx + u][0]});
            image[indx][0] = ulim_generic(image[indx][0], g2, g1);
            g1 = std::min({image[indx + 1 + u][2], image[indx + 1 - u][2], image[indx - 1 + u][2], image[indx - 1 - u][2], image[indx - 1][2], image[indx + 1][2], image[indx - u][2], image[indx + u][2]});
            g2 = std::max({image[indx + 1 + u][2], image[indx + 1 - u][2], image[indx - 1 + u][2], image[indx - 1 - u][2], image[indx - 1][2], image[indx + 1][2], image[indx - u][2], image[indx + u][2]});
            image[indx][2] = ulim_generic(image[indx][2], g2, g1);
        }
    }
}

void DCB_Processor::dcb_map() {
    int u = width;
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1, indx = row * u + col; col < width - 1; col++, indx++) {
            if (image[indx][1] > (image[indx - 1][1] + image[indx + 1][1] + image[indx - u][1] + image[indx + u][1]) / 4.0f)
                image[indx][3] = ((std::min(image[indx - 1][1], image[indx + 1][1]) + image[indx - 1][1] + image[indx + 1][1]) < (std::min(image[indx - u][1], image[indx + u][1]) + image[indx - u][1] + image[indx + u][1]));
            else
                image[indx][3] = ((std::max(image[indx - 1][1], image[indx + 1][1]) + image[indx - 1][1] + image[indx + 1][1]) > (std::max(image[indx - u][1], image[indx + u][1]) + image[indx - u][1] + image[indx + u][1]));
        }
    }
}

void DCB_Processor::dcb_correction() {
    int v = 2 * width;
    for (int row = 2; row < height - 2; row++) {
        for (int col = 2 + (fcol_bayer(row, 2, filters) & 1), indx = row * width + col; col < width - 2; col += 2, indx += 2) {
            int current = 4 * image[indx][3] + 2 * (image[indx + width][3] + image[indx - width][3] + image[indx + 1][3] + image[indx - 1][3]) +
                          image[indx + v][3] + image[indx - v][3] + image[indx + 2][3] + image[indx - 2][3];
            image[indx][1] = ((16 - current) * (image[indx - 1][1] + image[indx + 1][1]) / 2.0f + current * (image[indx - width][1] + image[indx + width][1]) / 2.0f) / 16.0f;
        }
    }
}

void DCB_Processor::dcb_correction2() {
    int v = 2 * width;
    for (int row = 4; row < height - 4; row++) {
        for (int col = 4 + (fcol_bayer(row, 2, filters) & 1), indx = row * width + col; col < width - 4; col += 2, indx += 2) {
            int c = fcol_bayer(row, col, filters);
            int current = 4 * image[indx][3] + 2 * (image[indx + width][3] + image[indx - width][3] + image[indx + 1][3] + image[indx - 1][3]) +
                          image[indx + v][3] + image[indx - v][3] + image[indx + 2][3] + image[indx - 2][3];
            image[indx][1] = CLIP_FLOAT(((16 - current) * ((image[indx - 1][1] + image[indx + 1][1]) / 2.0f + image[indx][c] - (image[indx + 2][c] + image[indx - 2][c]) / 2.0f) +
                                         current * ((image[indx - width][1] + image[indx + width][1]) / 2.0f + image[indx][c] - (image[indx + v][c] + image[indx - v][c]) / 2.0f)) / 16.0f);
        }
    }
}

void DCB_Processor::dcb_refinement() {
    int u = width, v = 2 * u, w = 3 * u;
    for (int row = 4; row < height - 4; row++) {
        for (int col = 4 + (fcol_bayer(row, 2, filters) & 1), indx = row * u + col; col < u - 4; col += 2, indx += 2) {
            int c = fcol_bayer(row, col, filters);

            // ✨ バグ修正: image[indx + 2][3] を2回足していたのを image[indx - 2][3] に修正
            int current = 4 * image[indx][3] + 2 * (image[indx + u][3] + image[indx - u][3] + image[indx + 1][3] + image[indx - 1][3]) +
                          image[indx + v][3] + image[indx - v][3] + image[indx + 2][3] + image[indx - 2][3];

            if (image[indx][c] > 1) {
                float f[5];
                f[0] = (image[indx - u][1] + image[indx + u][1]) / (2.0f * image[indx][c]);
                f[1] = (image[indx - v][c] > 0) ? 2.0f * image[indx - u][1] / (image[indx - v][c] + image[indx][c]) : f[0];
                f[2] = (image[indx - v][c] > 0) ? (image[indx - u][1] + image[indx - w][1]) / (2.0f * image[indx - v][c]) : f[0];
                f[3] = (image[indx + v][c] > 0) ? 2.0f * image[indx + u][1] / (image[indx + v][c] + image[indx][c]) : f[0];
                f[4] = (image[indx + v][c] > 0) ? (image[indx + u][1] + image[indx + w][1]) / (2.0f * image[indx + v][c]) : f[0];
                float g1 = (5.0f * f[0] + 3.0f * f[1] + f[2] + 3.0f * f[3] + f[4]) / 13.0f;

                f[0] = (image[indx - 1][1] + image[indx + 1][1]) / (2.0f * image[indx][c]);
                f[1] = (image[indx - 2][c] > 0) ? 2.0f * image[indx - 1][1] / (image[indx - 2][c] + image[indx][c]) : f[0];
                f[2] = (image[indx - 2][c] > 0) ? (image[indx - 1][1] + image[indx - 3][1]) / (2.0f * image[indx - 2][c]) : f[0];
                f[3] = (image[indx + 2][c] > 0) ? 2.0f * image[indx + 1][1] / (image[indx + 2][c] + image[indx][c]) : f[0];
                f[4] = (image[indx + 2][c] > 0) ? (image[indx + 1][1] + image[indx + 3][1]) / (2.0f * image[indx + 2][c]) : f[0];
                float g2 = (5.0f * f[0] + 3.0f * f[1] + f[2] + 3.0f * f[3] + f[4]) / 13.0f;
                
                image[indx][1] = CLIP_FLOAT((image[indx][c]) * (current * g1 + (16 - current) * g2) / 16.0f);
            } else {
                image[indx][1] = image[indx][c];
            }

            float g1 = std::min({image[indx + 1 + u][1], image[indx + 1 - u][1], image[indx - 1 + u][1], image[indx - 1 - u][1], image[indx - 1][1], image[indx + 1][1], image[indx - u][1], image[indx + u][1]});
            float g2 = std::max({image[indx + 1 + u][1], image[indx + 1 - u][1], image[indx - 1 + u][1], image[indx - 1 - u][1], image[indx - 1][1], image[indx + 1][1], image[indx - u][1], image[indx + u][1]});
            image[indx][1] = ulim_generic(image[indx][1], g2, g1);
        }
    }
}

void DCB_Processor::run_dcb(int iterations, bool dcb_enhance) {
    size_t buffer_size = static_cast<size_t>(width) * height;
    std::vector<float> image2_vec(buffer_size * 3), image3_vec(buffer_size * 3);
    float (*image2)[3] = reinterpret_cast<float(*)[3]>(image2_vec.data());
    float (*image3)[3] = reinterpret_cast<float(*)[3]>(image3_vec.data());
    
    dcb_hor(image2);
    dcb_color2(image2);
    dcb_ver(image3);
    dcb_color3(image3);
    dcb_decide(image2, image3);
    image3_vec.clear();
    image3_vec.shrink_to_fit();
    dcb_copy_to_buffer(image2);
    
    for (int i = 1; i <= iterations; i++) {
        dcb_nyquist();
        dcb_nyquist();
        dcb_nyquist();
        dcb_map();
        dcb_correction();
    }
    
    dcb_color();
    dcb_pp();
    dcb_map();
    dcb_correction2();
    dcb_map();
    dcb_correction();
    dcb_map();
    dcb_correction();
    dcb_map();
    dcb_correction();
    dcb_map();
    dcb_restore_from_buffer(image2);
    dcb_color();
    
    if (dcb_enhance) {
        dcb_refinement();
        dcb_color_full();
    }
}

} // anonymous namespace (DCB)

bool CPUAccelerator::demosaic_bayer_dcb(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, uint16_t maximum_value, int iterations, bool dcb_enhance) {

    if (!initialized_ || !raw_buffer.is_valid() || !rgb_buffer.is_valid() || raw_buffer.width != rgb_buffer.width || raw_buffer.height != rgb_buffer.height) {
        std::cerr << "❌ DCB Demosaic: Invalid buffers or initialization state." << std::endl;
        return false;
    }
    std::cout << "🔧 Starting DCB demosaic (Clean Class Implementation)..." << std::endl;
        
    const size_t width = raw_buffer.width, height = raw_buffer.height;
    const size_t num_pixels = width * height;

    auto sparse_image_ptr = std::make_unique<float[][4]>(num_pixels);
    float (*sparse_image)[4] = sparse_image_ptr.get();
    auto work_image_ptr = std::make_unique<float[][4]>(num_pixels);
    float (*work_image)[4] = work_image_ptr.get();
    
    // 手順1: 正しいRAWデータの読み出し
    for (size_t i = 0; i < num_pixels; ++i) {
        size_t row = i / width;
        size_t col = i % width;
        int color_idx = fcol_bayer(row, col, filters);

        uint16_t raw_value = raw_buffer.image[i][color_idx];

        // 全チャンネルを0で初期化
        for(int c=0; c<4; ++c) sparse_image[i][c] = 0.0f;
        
        // ネイティブカラーのみをセット
        sparse_image[i][color_idx] = static_cast<float>(raw_value);
    }
    
    // 手順2: バイリニア補間 - 隣接ピクセルのネイティブ値から補間
    std::memcpy(work_image, sparse_image, num_pixels * sizeof(float[4]));
    for (size_t r = 1; r < height - 1; ++r) {
        for (size_t c = 1; c < width - 1; ++c) {
            size_t i = r * width + c;
            int f = fcol_bayer(r, c, filters);
            
            // 各チャンネルで欠けている色を隣接ピクセルから補間
            for (int ch = 0; ch < 3; ch++) {
                if (ch == f) continue; // ネイティブカラーはスキップ
                
                float sum = 0.0f;
                int count = 0;
                
                // 隣接する8ピクセルをチェック
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        if (dr == 0 && dc == 0) continue;
                        size_t nr = r + dr;
                        size_t nc = c + dc;
                        if (nr < height && nc < width) {
                            size_t ni = nr * width + nc;
                            int neighbor_color = fcol_bayer(nr, nc, filters);
                            if (neighbor_color == ch) {
                                sum += sparse_image[ni][ch];
                                count++;
                            }
                        }
                    }
                }
                
                if (count > 0) {
                    work_image[i][ch] = sum / count;
                }
            }
        }
    }
    
    // 手順4: 完全なDCBアルゴリズムを実行
    DCB_Processor dcb_proc(work_image, static_cast<int>(width), static_cast<int>(height), filters, maximum_value);
    dcb_proc.run_dcb(iterations, dcb_enhance);
    
    // ✨✨✨ 手順5: float正規化書き込み ✨✨✨
    // work_image (4チャンネル) から rgb_buffer (3チャンネル) への変換
    for (size_t i = 0; i < num_pixels; ++i) {
        rgb_buffer.image[i][0] = work_image[i][0] / (float)maximum_value;
        rgb_buffer.image[i][1] = work_image[i][1] / (float)maximum_value;
        rgb_buffer.image[i][2] = work_image[i][2] / (float)maximum_value;
    }
    
    // ✨デモザイク後境界補間✨: 最終処理で境界を修正
    border_interpolate(rgb_buffer, filters, 3);
    
    return true;
}

//===================================================================
// AMaZE demosaicing algorithm implementation
//===================================================================

namespace { // This file is for internal use only

class AMaZE_Processor_RT {
private:
    // Constants from the RawTherapee implementation
    static constexpr int TS = 160;
    static constexpr int TSH = TS / 2;
    static constexpr float eps = 1e-5f;
    static constexpr float epssq = 1e-10f;
    static constexpr float arthresh = 0.75f;
    static constexpr float nyqthresh = 0.5f;

    static constexpr float gaussodd[4] = {0.14659727707323927f, 0.103592713382435f, 0.0732036125103057f, 0.0365543548389495f};
    static constexpr float gaussgrad[6] = {
        nyqthresh * 0.07384411893421103f, nyqthresh * 0.06207511968171489f, nyqthresh * 0.0521818194747806f,
        nyqthresh * 0.03687419286733595f, nyqthresh * 0.03099732204057846f, nyqthresh * 0.018413194161458882f
    };
    static constexpr float gausseven[2] = {0.13719494435797422f, 0.05640252782101291f};
    static constexpr float gquinc[4] = {0.169917f, 0.108947f, 0.069855f, 0.0287182f};
    
    static inline float xmul2f(float x) { return x + x; }
    static inline float xdiv2f(float x) { return x * 0.5f; }

    float initialGain;
    float clip_pt;
    float clip_pt8;

    const ImageBuffer& raw_buffer_;
    ImageBufferFloat& rgb_buffer_;
    const uint32_t filters_;
    const float (&cam_mul_)[4];
    const size_t width_, height_;
    const uint16_t maximum_value_;
    int ex = 0, ey = 0;
    unsigned int cfarray[2][2]; // Bayer pattern cache

    struct s_hv { float h; float v; };

public:
    AMaZE_Processor_RT(const ImageBuffer& raw, ImageBufferFloat& rgb, uint32_t f, const float (&cam_mul)[4], uint16_t max_val)
        : raw_buffer_(raw), rgb_buffer_(rgb), filters_(f), cam_mul_(cam_mul), width_(raw.width), height_(raw.height), maximum_value_(max_val)
    {
        // Assuming G1==G2 and are not white-balanced (pre-demosaic)
        float max_mul = std::max({cam_mul_[0], cam_mul_[2]});
        float min_mul = std::min({cam_mul_[0], cam_mul_[2]});
        initialGain = (min_mul > 1e-6f) ? (max_mul / min_mul) : 1.0f;
        
        clip_pt = 1.0f / initialGain;
        clip_pt8 = 0.8f / initialGain;
        
        // Cache the 2x2 Bayer pattern based on the provided helper function
        cfarray[0][0] = fcol_bayer(0, 0, filters_);
        cfarray[0][1] = fcol_bayer(0, 1, filters_);
        cfarray[1][0] = fcol_bayer(1, 0, filters_);
        cfarray[1][1] = fcol_bayer(1, 1, filters_);
        
        // Determine R pixel offset (ey, ex)
        if (cfarray[0][0] == 1) { // Top-left is G (GRBG or GBRG)
            if (cfarray[0][1] == 0) { ey = 0; ex = 1; } // GRBG
            else { ey = 1; ex = 0; }               // GBRG
        } else { // Top-left is R or B (RGGB or BGGR)
            if (cfarray[0][0] == 0) { ey = 0; ex = 0; } // RGGB
            else { ey = 1; ex = 1; }               // BGGR
        }
    }

    // A robust Bayer pattern lookup function, like in RawTherapee
    unsigned int fc_rt(int r, int c) const {
        return cfarray[r & 1][c & 1];
    }

    void run() {
        static const int v1 = TS, v2 = 2*TS, v3 = 3*TS, p1 = -TS+1, p2 = -2*TS+2, p3 = -3*TS+3, m1 = TS+1, m2 = 2*TS+2, m3 = 3*TS+3;
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {        
            std::vector<char> processing_buffer(15 * sizeof(float) * TS * TS + sizeof(unsigned char) * TS * TSH + 256);
            char* aligned_buffer = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(processing_buffer.data()) + 63) & ~63);

            float* rgbgreen     = reinterpret_cast<float*>(aligned_buffer);
            float* delhvsqsum   = rgbgreen + TS*TS;
            float* dirwts0      = delhvsqsum + TS*TS;
            float* dirwts1      = dirwts0 + TS*TS;
            float* vcd          = dirwts1 + TS*TS;
            float* hcd          = vcd + TS*TS;
            float* vcdalt       = hcd + TS*TS;
            float* hcdalt       = vcdalt + TS*TS;
            float* cddiffsq     = hcdalt + TS*TS;
            float* hvwt         = cddiffsq + TS*TS;
            float* dgintv       = reinterpret_cast<float*>(reinterpret_cast<char*>(hvwt) + sizeof(float) * TS * TSH);
            float* dginth       = dgintv + TS*TS;
            float* Dgrbsq1m     = dginth + TS*TS;
            float* Dgrbsq1p     = reinterpret_cast<float*>(reinterpret_cast<char*>(Dgrbsq1m) + sizeof(float) * TS * TSH);
            float* cfa          = reinterpret_cast<float*>(reinterpret_cast<char*>(Dgrbsq1p) + sizeof(float) * TS * TSH);
            unsigned char* nyquist = reinterpret_cast<unsigned char*>(cfa + TS*TS);
            unsigned char* nyquist2 = reinterpret_cast<unsigned char*>(cddiffsq);
            float* nyqutest     = reinterpret_cast<float*>(reinterpret_cast<char*>(nyquist) + sizeof(unsigned char) * TS * TSH);

#ifdef _OPENMP
            #pragma omp for collapse(2) schedule(dynamic, 8) nowait
#endif
            for (int top = -16; top < (int)height_; top += TS - 32) {
                for (int left = -16; left < (int)width_; left += TS - 32) {
                    int bottom = std::min(top + TS, (int)height_ + 16);
                    int right = std::min(left + TS, (int)width_ + 16);
                    int rr1 = bottom - top;
                    int cc1 = right - left;
                    int rrmin = (top < 0) ? 16 : 0;
                    int ccmin = (left < 0) ? 16 : 0;
                    int rrmax = (bottom > (int)height_) ? (int)height_ - top : rr1;
                    int ccmax = (right > (int)width_) ? (int)width_ - left : cc1;

                    // === Tile Initialization with 16-pixel border ===
                    const float scale = 1.0f / maximum_value_;
                    
                    // Fill upper border
                    if (rrmin > 0) {
                        for (int rr = 0; rr < 16; rr++) {
                            for (int cc = ccmin; cc < ccmax; cc++) {
                                int row = 32 - rr + top;
                                int safe_row = std::max(0, std::min((int)height_ - 1, row));
                                int safe_col = std::max(0, std::min((int)width_ - 1, cc + left));
                                int c = fc_rt(safe_row, safe_col);
                                float val = (float)raw_buffer_.image[safe_row * width_ + safe_col][c] * scale;
                                cfa[rr * TS + cc] = val;
                                rgbgreen[rr * TS + cc] = val;
                            }
                        }
                    }

                    // Fill inner part
                    for (int rr = rrmin; rr < rrmax; rr++) {
                        int row = rr + top;
                        int cc = ccmin;
                        for (; cc < ccmax; cc++) {
                            int indx1 = rr * TS + cc;
                            int c = fc_rt(row, cc + left);
                            float val = (float)raw_buffer_.image[row * width_ + (cc + left)][c] * scale;
                            cfa[indx1] = val;
                            rgbgreen[indx1] = val;
                        }
                    }
                    
                    // Fill lower border
                    if (rrmax < rr1) {
                        for (int rr = 0; rr < 16; rr++) {
                            for (int cc = ccmin; cc < ccmax; cc++) {
                                int safe_row = std::max(0, std::min((int)height_ - 1, (int)height_ - rr - 2));
                                int safe_col = std::max(0, std::min((int)width_ - 1, left + cc));
                                int c = fc_rt(safe_row, safe_col);
                                float val = (float)raw_buffer_.image[safe_row * width_ + safe_col][c] * scale;
                                cfa[(rrmax + rr) * TS + cc] = val;
                                rgbgreen[(rrmax + rr) * TS + cc] = val;
                            }
                        }
                    }
                    // Fill left/right borders and corners (scalar is fine for these)
                    if (ccmin > 0) {
                        for (int rr = rrmin; rr < rrmax; rr++) {
                            for (int cc = 0; cc < 16; cc++) {
                                int row = rr + top;
                                int safe_row = std::max(0, std::min((int)height_ - 1, row));
                                int safe_col = std::max(0, std::min((int)width_ - 1, 32 - cc + left));
                                int c = fc_rt(safe_row, safe_col);
                                float val = (float)raw_buffer_.image[safe_row * width_ + safe_col][c] * scale;
                                cfa[rr * TS + cc] = val;
                                rgbgreen[rr * TS + cc] = val;
                            }
                        }
                    }
                    if (ccmax < cc1) {
                        for (int rr = rrmin; rr < rrmax; rr++) {
                            for (int cc = 0; cc < 16; cc++) {
                                int safe_row = std::max(0, std::min((int)height_ - 1, top + rr));
                                int safe_col = std::max(0, std::min((int)width_ - 1, (int)width_ - cc - 2));
                                int c = fc_rt(safe_row, safe_col);
                                float val = (float)raw_buffer_.image[safe_row * width_ + safe_col][c] * scale;
                                cfa[rr * TS + ccmax + cc] = val;
                                rgbgreen[rr * TS + ccmax + cc] = val;
                            }
                        }
                    }
                    // ... corners ...

                    // === Start of RawTherapee AMaZE Algorithm Stages ===
                    
                    // STAGE 1: Horizontal and vertical gradients
                    for (int rr = 2; rr < rr1 - 2; rr++) {
                        for (int cc = 2, idx = rr*TS+cc; cc < cc1 - 2; cc++, idx++) {
                            float delh = fabsf(cfa[idx+1]-cfa[idx-1]);
                            float delv = fabsf(cfa[idx+v1]-cfa[idx-v1]);
                            dirwts0[idx] = eps+fabsf(cfa[idx+v2]-cfa[idx])+fabsf(cfa[idx]-cfa[idx-v2])+delv;
                            dirwts1[idx] = eps+fabsf(cfa[idx+2]-cfa[idx])+fabsf(cfa[idx]-cfa[idx-2])+delh;
                            delhvsqsum[idx] = SQR(delh)+SQR(delv);
                        }
                    }

                    // STAGE 2: Interpolate vertical and horizontal colour differences
                    for (int rr = 4; rr < rr1 - 4; rr++) {
                        for (int cc = 4, idx=rr*TS+cc; cc < cc1 - 4; cc++, idx++) {
                            int sgn = (fc_rt(rr,cc)&1) ? -1 : 1;
                            float cru = cfa[idx-v1]*(dirwts0[idx-v2]+dirwts0[idx]) / (dirwts0[idx-v2]*(eps+cfa[idx])+dirwts0[idx]*(eps+cfa[idx-v2]));
                            float crd = cfa[idx+v1]*(dirwts0[idx+v2]+dirwts0[idx]) / (dirwts0[idx+v2]*(eps+cfa[idx])+dirwts0[idx]*(eps+cfa[idx+v2]));
                            float crl = cfa[idx-1]*(dirwts1[idx-2]+dirwts1[idx]) / (dirwts1[idx-2]*(eps+cfa[idx])+dirwts1[idx]*(eps+cfa[idx-2]));
                            float crr = cfa[idx+1]*(dirwts1[idx+2]+dirwts1[idx]) / (dirwts1[idx+2]*(eps+cfa[idx])+dirwts1[idx]*(eps+cfa[idx+2]));
                            float guha=cfa[idx-v1]+xdiv2f(cfa[idx]-cfa[idx-v2]);
                            float gdha=cfa[idx+v1]+xdiv2f(cfa[idx]-cfa[idx+v2]);
                            float glha=cfa[idx-1]+xdiv2f(cfa[idx]-cfa[idx-2]);
                            float grha=cfa[idx+1]+xdiv2f(cfa[idx]-cfa[idx+2]);
                            float guar = fabsf(1.f-cru)<arthresh ? cfa[idx]*cru : guha;
                            float gdar = fabsf(1.f-crd)<arthresh ? cfa[idx]*crd : gdha;
                            float glar = fabsf(1.f-crl)<arthresh ? cfa[idx]*crl : glha;
                            float grar = fabsf(1.f-crr)<arthresh ? cfa[idx]*crr : grha;
                            float hwt=dirwts1[idx-1]/(dirwts1[idx-1]+dirwts1[idx+1]);
                            float vwt=dirwts0[idx-v1]/(dirwts0[idx+v1]+dirwts0[idx-v1]);
                            float Gintvha = vwt*gdha+(1.f-vwt)*guha;
                            float Ginthha = hwt*grha+(1.f-hwt)*glha;
                            vcdalt[idx] = sgn*(Gintvha-cfa[idx]); 
                            hcdalt[idx] = sgn*(Ginthha-cfa[idx]);
                            if(cfa[idx]>clip_pt8 || Gintvha>clip_pt8 || Ginthha>clip_pt8) {
                                vcd[idx] = vcdalt[idx]; hcd[idx] = hcdalt[idx];
                                guar=guha; gdar=gdha; glar=glha; grar=grha;
                            } else {
                                vcd[idx] = sgn*(vwt*gdar+(1.f-vwt)*guar-cfa[idx]);
                                hcd[idx] = sgn*(hwt*grar+(1.f-hwt)*glar-cfa[idx]);
                            }
                            dgintv[idx] = std::min(SQR(guha-gdha), SQR(guar-gdar));
                            dginth[idx] = std::min(SQR(glha-grha), SQR(glar-grar));
                        }
                    }
                    
                    // STAGE 3: Variance calculation and interpolation bounding
                    for (int rr = 4; rr < rr1 - 4; rr++) {
                        for (int cc = 4, idx = rr*TS+cc; cc < cc1-4; cc++, idx++) {
                            float hcdvar = 3.f*(SQR(hcd[idx-2])+SQR(hcd[idx])+SQR(hcd[idx+2]))-SQR(hcd[idx-2]+hcd[idx]+hcd[idx+2]);
                            float hcdaltvar = 3.f*(SQR(hcdalt[idx-2])+SQR(hcdalt[idx])+SQR(hcdalt[idx+2]))-SQR(hcdalt[idx-2]+hcdalt[idx]+hcdalt[idx+2]);
                            float vcdvar = 3.f*(SQR(vcd[idx-v2])+SQR(vcd[idx])+SQR(vcd[idx+v2]))-SQR(vcd[idx-v2]+vcd[idx]+vcd[idx+v2]);
                            float vcdaltvar = 3.f*(SQR(vcdalt[idx-v2])+SQR(vcdalt[idx])+SQR(vcdalt[idx+v2]))-SQR(vcdalt[idx-v2]+vcdalt[idx]+vcdalt[idx+v2]);
                            if(hcdaltvar < hcdvar) hcd[idx] = hcdalt[idx];
                            if(vcdaltvar < vcdvar) vcd[idx] = vcdalt[idx];
                            float Gintv, Ginth;
                            if(fc_rt(rr,cc)&1) {
                                Ginth=-hcd[idx]+cfa[idx]; Gintv=-vcd[idx]+cfa[idx];
                                if(hcd[idx]>0){if(3.f*hcd[idx]>(Ginth+cfa[idx]))hcd[idx]=-median(Ginth,cfa[idx-1],cfa[idx+1])+cfa[idx];else{float hwt2=1.f-3.f*hcd[idx]/(eps+Ginth+cfa[idx]);hcd[idx]=hwt2*hcd[idx]+(1.f-hwt2)*(-median(Ginth,cfa[idx-1],cfa[idx+1])+cfa[idx]);}}
                                if(vcd[idx]>0){if(3.f*vcd[idx]>(Gintv+cfa[idx]))vcd[idx]=-median(Gintv,cfa[idx-v1],cfa[idx+v1])+cfa[idx];else{float vwt2=1.f-3.f*vcd[idx]/(eps+Gintv+cfa[idx]);vcd[idx]=vwt2*vcd[idx]+(1.f-vwt2)*(-median(Gintv,cfa[idx-v1],cfa[idx+v1])+cfa[idx]);}}
                            } else {
                                Ginth=hcd[idx]+cfa[idx]; Gintv=vcd[idx]+cfa[idx];
                                if(hcd[idx]<0){if(3.f*hcd[idx]<-(Ginth+cfa[idx]))hcd[idx]=median(Ginth,cfa[idx-1],cfa[idx+1])-cfa[idx];else{float hwt2=1.f+3.f*hcd[idx]/(eps+Ginth+cfa[idx]);hcd[idx]=hwt2*hcd[idx]+(1.f-hwt2)*(median(Ginth,cfa[idx-1],cfa[idx+1])-cfa[idx]);}}
                                if(vcd[idx]<0){if(3.f*vcd[idx]<-(Gintv+cfa[idx]))vcd[idx]=median(Gintv,cfa[idx-v1],cfa[idx+v1])-cfa[idx];else{float vwt2=1.f+3.f*vcd[idx]/(eps+Gintv+cfa[idx]);vcd[idx]=vwt2*vcd[idx]+(1.f-vwt2)*(median(Gintv,cfa[idx-v1],cfa[idx+v1])-cfa[idx]);}}
                            }
                            if(Ginth > clip_pt) hcd[idx] = (fc_rt(rr,cc)&1?-1:1) * (median(Ginth,cfa[idx-1],cfa[idx+1])-cfa[idx]);
                            if(Gintv > clip_pt) vcd[idx] = (fc_rt(rr,cc)&1?-1:1) * (median(Gintv,cfa[idx-v1],cfa[idx+v1])-cfa[idx]);
                            cddiffsq[idx] = SQR(vcd[idx]-hcd[idx]);
                        }
                    }

                    // STAGE 4 & 5: Directional variance, Texture analysis & Nyquist test calculation
                    for (int rr=6; rr<rr1-6; rr++) {
                        for (int cc=6+(fc_rt(rr,2)&1), idx=rr*TS+cc; cc<cc1-6; cc+=2, idx+=2) {
                            float uave = vcd[idx]+vcd[idx-v1]+vcd[idx-v2]+vcd[idx-v3];
                            float dave = vcd[idx]+vcd[idx+v1]+vcd[idx+v2]+vcd[idx+v3];
                            float lave = hcd[idx]+hcd[idx-1]+hcd[idx-2]+hcd[idx-3];
                            float rave = hcd[idx]+hcd[idx+1]+hcd[idx+2]+hcd[idx+3];
                            float Dgrbvvaru = SQR(vcd[idx]-uave)+SQR(vcd[idx-v1]-uave)+SQR(vcd[idx-v2]-uave)+SQR(vcd[idx-v3]-uave);
                            float Dgrbvvard = SQR(vcd[idx]-dave)+SQR(vcd[idx+v1]-dave)+SQR(vcd[idx+v2]-dave)+SQR(vcd[idx+v3]-dave);
                            float Dgrbhvarl = SQR(hcd[idx]-lave)+SQR(hcd[idx-1]-lave)+SQR(hcd[idx-2]-lave)+SQR(hcd[idx-3]-lave);
                            float Dgrbhvarr = SQR(hcd[idx]-rave)+SQR(hcd[idx+1]-rave)+SQR(hcd[idx+2]-rave)+SQR(hcd[idx+3]-rave);
                            float hwt = dirwts1[idx-1]/(dirwts1[idx-1]+dirwts1[idx+1]);
                            float vwt = dirwts0[idx-v1]/(dirwts0[idx+v1]+dirwts0[idx-v1]);
                            float vcdvar = epssq+vwt*Dgrbvvard+(1.f-vwt)*Dgrbvvaru;
                            float hcdvar = epssq+hwt*Dgrbhvarr+(1.f-hwt)*Dgrbhvarl;
                            float Dgrbvvaru2 = dgintv[idx]+dgintv[idx-v1]+dgintv[idx-v2];
                            float Dgrbvvard2 = dgintv[idx]+dgintv[idx+v1]+dgintv[idx+v2];
                            float Dgrbhvarl2 = dginth[idx]+dginth[idx-1]+dginth[idx-2];
                            float Dgrbhvarr2 = dginth[idx]+dginth[idx+1]+dginth[idx+2];
                            float vcdvar1 = epssq+vwt*Dgrbvvard2+(1.f-vwt)*Dgrbvvaru2;
                            float hcdvar1 = epssq+hwt*Dgrbhvarr2+(1.f-hwt)*Dgrbhvarl2;
                            float varwt = hcdvar/(vcdvar+hcdvar);
                            float diffwt = hcdvar1/(vcdvar1+hcdvar1);
                            hvwt[idx>>1] = ((0.5f-varwt)*(0.5f-diffwt)>0.f && fabsf(0.5f-diffwt)<fabsf(0.5f-varwt)) ? varwt : diffwt;
                        }
                    }
                    for (int rr=6; rr<rr1-6; rr++) {
                        for (int cc=6+(fc_rt(rr,2)&1),idx=rr*TS+cc; cc<cc1-6; cc+=2,idx+=2){
                            nyqutest[idx>>1]=(gaussodd[0]*cddiffsq[idx]+gaussodd[1]*(cddiffsq[idx-m1]+cddiffsq[idx+p1]+cddiffsq[idx-p1]+cddiffsq[idx+m1])+gaussodd[2]*(cddiffsq[idx-v2]+cddiffsq[idx-2]+cddiffsq[idx+2]+cddiffsq[idx+v2])+gaussodd[3]*(cddiffsq[idx-m2]+cddiffsq[idx+p2]+cddiffsq[idx-p2]+cddiffsq[idx+m2]))-(gaussgrad[0]*delhvsqsum[idx]+gaussgrad[1]*(delhvsqsum[idx-v1]+delhvsqsum[idx+1]+delhvsqsum[idx-1]+delhvsqsum[idx+v1])+gaussgrad[2]*(delhvsqsum[idx-m1]+delhvsqsum[idx+p1]+delhvsqsum[idx-p1]+delhvsqsum[idx+m1])+gaussgrad[3]*(delhvsqsum[idx-v2]+delhvsqsum[idx-2]+delhvsqsum[idx+2]+delhvsqsum[idx+v2])+gaussgrad[4]*(delhvsqsum[idx-v2-1]+delhvsqsum[idx-v2+1]+delhvsqsum[idx-TS-2]+delhvsqsum[idx-TS+2]+delhvsqsum[idx+TS-2]+delhvsqsum[idx+TS+2]+delhvsqsum[idx+v2-1]+delhvsqsum[idx+v2+1])+gaussgrad[5]*(delhvsqsum[idx-m2]+delhvsqsum[idx+p2]+delhvsqsum[idx-p2]+delhvsqsum[idx+m2]));
                        }
                    }
                    bool doNyquist = false;
                    memset(nyquist, 0, sizeof(unsigned char) * TS * TSH);
                    for (int rr = 6; rr < rr1-6; rr++) for (int cc=6+(fc_rt(rr,2)&1),idx=rr*TS+cc; cc<cc1-6; cc+=2,idx+=2) if(nyqutest[idx>>1]>0.f) {nyquist[idx>>1]=1; doNyquist=true;}

                    // STAGE 6: Nyquist processing & Green interpolation
                    if(doNyquist) {
                        memset(nyquist2, 0, sizeof(unsigned char)*TS*TSH);
                        for (int rr=8; rr<rr1-8; rr++) {
                            for (int cc=8+(fc_rt(rr,2)&1),idx=rr*TS+cc; cc<cc1-8; cc+=2,idx+=2) {
                                unsigned int nqsum=(nyquist[(idx-v2)>>1]+nyquist[(idx-m1)>>1]+nyquist[(idx+p1)>>1]+nyquist[(idx-2)>>1]+nyquist[(idx+2)>>1]+nyquist[(idx-p1)>>1]+nyquist[(idx+m1)>>1]+nyquist[(idx+v2)>>1]);
                                nyquist2[idx>>1]= nqsum>4?1:(nqsum<4?0:nyquist[idx>>1]);
                            }
                        }
                        for (int rr=8; rr<rr1-8; rr++) {
                            for (int cc=8+(fc_rt(rr,2)&1),idx=rr*TS+cc; cc<cc1-8; cc+=2,idx+=2) if(nyquist2[idx>>1]) {
                                float sumcfa=0,sumh=0,sumv=0,sumsqh=0,sumsqv=0,areawt=0;
                                for(int i=-6;i<7;i+=2) for(int j=-6;j<7;j+=2) {
                                    int idx1=idx+i*TS+j;
                                    if ( (idx1 >= 0) && (idx1 < TS*TS) && nyquist2[idx1>>1]) {
                                        float cfatemplate=cfa[idx1]; sumcfa+=cfatemplate; sumh+=(cfa[idx1-1]+cfa[idx1+1]); sumv+=(cfa[idx1-v1]+cfa[idx1+v1]);
                                        sumsqh+=SQR(cfatemplate-cfa[idx1-1])+SQR(cfatemplate-cfa[idx1+1]); sumsqv+=SQR(cfatemplate-cfa[idx1-v1])+SQR(cfatemplate-cfa[idx1+v1]); areawt+=1;
                                    }
                                }
                                sumh=sumcfa-xdiv2f(sumh); sumv=sumcfa-xdiv2f(sumv); areawt=xdiv2f(areawt);
                                float hcdvar2=epssq+fabsf(areawt*sumsqh-sumh*sumh), vcdvar2=epssq+fabsf(areawt*sumsqv-sumv*sumv);
                                hvwt[idx>>1]=hcdvar2/(vcdvar2+hcdvar2);
                            }
                        }
                    }
                    float* Dgrb0 = vcdalt; 
                    s_hv* Dgrb2_ptr = reinterpret_cast<s_hv*>(Dgrbsq1m);
                    for (int rr=8; rr<rr1-8; rr++) {
                        for (int cc=8+(fc_rt(rr,2)&1),idx=rr*TS+cc; cc<cc1-8; cc+=2,idx+=2){
                            float hvwtalt=0.25f*(hvwt[(idx-m1)>>1]+hvwt[(idx+p1)>>1]+hvwt[(idx-p1)>>1]+hvwt[(idx+m1)>>1]);
                            hvwt[idx>>1] = fabsf(0.5f-hvwt[idx>>1])<fabsf(0.5f-hvwtalt) ? hvwtalt : hvwt[idx>>1];
                            Dgrb0[idx>>1] = hvwt[idx>>1]*vcd[idx] + (1.f-hvwt[idx>>1])*hcd[idx];
                            rgbgreen[idx] = cfa[idx] + Dgrb0[idx>>1];
                            Dgrb2_ptr[idx>>1].h = nyquist2[idx>>1] ? SQR(rgbgreen[idx]-xdiv2f(rgbgreen[idx-1]+rgbgreen[idx+1])) : 0.f;
                            Dgrb2_ptr[idx>>1].v = nyquist2[idx>>1] ? SQR(rgbgreen[idx]-xdiv2f(rgbgreen[idx-v1]+rgbgreen[idx+v1])) : 0.f;
                        }
                    }
                    if(doNyquist) {
                        for(int rr=8; rr<rr1-8; rr++) {
                            for(int cc=8+(fc_rt(rr,2)&1),idx=rr*TS+cc; cc<cc1-8; cc+=2,idx+=2) if(nyquist2[idx>>1]){
                                float gvarh=epssq+(gquinc[0]*Dgrb2_ptr[idx>>1].h+gquinc[1]*(Dgrb2_ptr[(idx-m1)>>1].h+Dgrb2_ptr[(idx+p1)>>1].h+Dgrb2_ptr[(idx-p1)>>1].h+Dgrb2_ptr[(idx+m1)>>1].h)+gquinc[2]*(Dgrb2_ptr[(idx-v2)>>1].h+Dgrb2_ptr[(idx-2)>>1].h+Dgrb2_ptr[(idx+2)>>1].h+Dgrb2_ptr[(idx+v2)>>1].h)+gquinc[3]*(Dgrb2_ptr[(idx-m2)>>1].h+Dgrb2_ptr[(idx+p2)>>1].h+Dgrb2_ptr[(idx-p2)>>1].h+Dgrb2_ptr[(idx+m2)>>1].h));
                                float gvarv=epssq+(gquinc[0]*Dgrb2_ptr[idx>>1].v+gquinc[1]*(Dgrb2_ptr[(idx-m1)>>1].v+Dgrb2_ptr[(idx+p1)>>1].v+Dgrb2_ptr[(idx-p1)>>1].v+Dgrb2_ptr[(idx+m1)>>1].v)+gquinc[2]*(Dgrb2_ptr[(idx-v2)>>1].v+Dgrb2_ptr[(idx-2)>>1].v+Dgrb2_ptr[(idx+2)>>1].v+Dgrb2_ptr[(idx+v2)>>1].v)+gquinc[3]*(Dgrb2_ptr[(idx-m2)>>1].v+Dgrb2_ptr[(idx+p2)>>1].v+Dgrb2_ptr[(idx-p2)>>1].v+Dgrb2_ptr[(idx+m2)>>1].v));
                                Dgrb0[idx>>1]=(hcd[idx]*gvarv+vcd[idx]*gvarh)/(gvarv+gvarh);
                                rgbgreen[idx]=cfa[idx]+Dgrb0[idx>>1];
                            }
                        }
                    }

                    // STAGE 7: Red/Blue gradient preprocessing
                    float* delp = cddiffsq;
                    float* delm = reinterpret_cast<float*>(reinterpret_cast<char*>(delp) + sizeof(float) * TS * TSH);
                    for (int rr = 6; rr < rr1 - 6; rr++) {
                        if ((fc_rt(rr, 2) & 1) == 0) {
                            for (int cc = 6, idx = rr * TS + cc; cc < cc1 - 6; cc += 2, idx += 2) {
                                delp[idx >> 1] = fabsf(cfa[idx + p1] - cfa[idx - p1]);
                                delm[idx >> 1] = fabsf(cfa[idx + m1] - cfa[idx - m1]);
                                Dgrbsq1p[idx >> 1] = SQR(cfa[idx + 1] - cfa[idx + 1 - p1]) + SQR(cfa[idx + 1] - cfa[idx + 1 + p1]);
                                Dgrbsq1m[idx >> 1] = SQR(cfa[idx + 1] - cfa[idx + 1 - m1]) + SQR(cfa[idx + 1] - cfa[idx + 1 + m1]);
                            }
                        } else {
                            for (int cc = 6, idx = rr * TS + cc; cc < cc1 - 6; cc += 2, idx += 2) {
                                Dgrbsq1p[idx >> 1] = SQR(cfa[idx] - cfa[idx - p1]) + SQR(cfa[idx] - cfa[idx + p1]);
                                Dgrbsq1m[idx >> 1] = SQR(cfa[idx] - cfa[idx - m1]) + SQR(cfa[idx] - cfa[idx + m1]);
                                delp[idx >> 1] = fabsf(cfa[idx + 1 + p1] - cfa[idx + 1 - p1]);
                                delm[idx >> 1] = fabsf(cfa[idx + 1 + m1] - cfa[idx + 1 - m1]);
                            }
                        }
                    }
                    
                    // STAGE 8: Red/Blue color ratio interpolation (Diagonal)
                    float* rbm = vcd;
                    float* rbp = hcdalt;
                    float* pmwt = dirwts1;
                    for (int rr = 8; rr < rr1 - 8; rr++) {
                        for (int cc = 8 + (fc_rt(rr, 2) & 1), idx = rr * TS + cc, idx1 = idx >> 1; cc < cc1 - 8; cc += 2, idx += 2, idx1++) {
                            float crse = xmul2f(cfa[idx + m1]) / (eps + cfa[idx] + cfa[idx + m2]);
                            float crnw = xmul2f(cfa[idx - m1]) / (eps + cfa[idx] + cfa[idx - m2]);
                            float crne = xmul2f(cfa[idx + p1]) / (eps + cfa[idx] + cfa[idx + p2]);
                            float crsw = xmul2f(cfa[idx - p1]) / (eps + cfa[idx] + cfa[idx - p2]);
                            float rbse = fabsf(1.f - crse) < arthresh ? cfa[idx] * crse : (cfa[idx + m1]) + xdiv2f(cfa[idx] - cfa[idx + m2]);
                            float rbnw = fabsf(1.f - crnw) < arthresh ? cfa[idx] * crnw : (cfa[idx - m1]) + xdiv2f(cfa[idx] - cfa[idx - m2]);
                            float rbne = fabsf(1.f - crne) < arthresh ? cfa[idx] * crne : (cfa[idx + p1]) + xdiv2f(cfa[idx] - cfa[idx + p2]);
                            float rbsw = fabsf(1.f - crsw) < arthresh ? cfa[idx] * crsw : (cfa[idx - p1]) + xdiv2f(cfa[idx] - cfa[idx - p2]);
                            float wtse = eps + delm[idx1] + delm[(idx + m1) >> 1] + delm[(idx + m2) >> 1];
                            float wtnw = eps + delm[idx1] + delm[(idx - m1) >> 1] + delm[(idx - m2) >> 1];
                            float wtne = eps + delp[idx1] + delp[(idx + p1) >> 1] + delp[(idx + p2) >> 1];
                            float wtsw = eps + delp[idx1] + delp[(idx - p1) >> 1] + delp[(idx - p2) >> 1];
                            rbm[idx1] = (wtse * rbnw + wtnw * rbse) / (wtse + wtnw);
                            rbp[idx1] = (wtne * rbsw + wtsw * rbne) / (wtne + wtsw);
                            if (rbp[idx1] < cfa[idx]) { if (xmul2f(rbp[idx1]) < cfa[idx]) rbp[idx1] = median(rbp[idx1] , cfa[idx - p1], cfa[idx + p1]); else { float pwt = xmul2f(cfa[idx] - rbp[idx1]) / (eps + rbp[idx1] + cfa[idx]); rbp[idx1] = pwt * rbp[idx1] + (1.f - pwt) * median(rbp[idx1], cfa[idx - p1], cfa[idx + p1]); } }
                            if (rbm[idx1] < cfa[idx]) { if (xmul2f(rbm[idx1]) < cfa[idx]) rbm[idx1] = median(rbm[idx1] , cfa[idx - m1], cfa[idx + m1]); else { float mwt = xmul2f(cfa[idx] - rbm[idx1]) / (eps + rbm[idx1] + cfa[idx]); rbm[idx1] = mwt * rbm[idx1] + (1.f - mwt) * median(rbm[idx1], cfa[idx - m1], cfa[idx + m1]); } }
                            if (rbp[idx1] > clip_pt) rbp[idx1] = median(rbp[idx1], cfa[idx - p1], cfa[idx + p1]);
                            if (rbm[idx1] > clip_pt) rbm[idx1] = median(rbm[idx1], cfa[idx - m1], cfa[idx + m1]);
                            float rbvarm = epssq + (gausseven[0] * (Dgrbsq1m[(idx - v1) >> 1] + Dgrbsq1m[(idx - 1) >> 1] + Dgrbsq1m[(idx + 1) >> 1] + Dgrbsq1m[(idx + v1) >> 1]) + gausseven[1] * (Dgrbsq1m[(idx - v2 - 1) >> 1] + Dgrbsq1m[(idx - v2 + 1) >> 1] + Dgrbsq1m[(idx - 2 - v1) >> 1] + Dgrbsq1m[(idx + 2 - v1) >> 1] + Dgrbsq1m[(idx - 2 + v1) >> 1] + Dgrbsq1m[(idx + 2 + v1) >> 1] + Dgrbsq1m[(idx + v2 - 1) >> 1] + Dgrbsq1m[(idx + v2 + 1) >> 1]));
                            float rbvarp = epssq + (gausseven[0] * (Dgrbsq1p[(idx - v1) >> 1] + Dgrbsq1p[(idx - 1) >> 1] + Dgrbsq1p[(idx + 1) >> 1] + Dgrbsq1p[(idx + v1) >> 1]) + gausseven[1] * (Dgrbsq1p[(idx - v2 - 1) >> 1] + Dgrbsq1p[(idx - v2 + 1) >> 1] + Dgrbsq1p[(idx - 2 - v1) >> 1] + Dgrbsq1p[(idx + 2 - v1) >> 1] + Dgrbsq1p[(idx - 2 + v1) >> 1] + Dgrbsq1p[(idx + 2 + v1) >> 1] + Dgrbsq1p[(idx + v2 - 1) >> 1] + Dgrbsq1p[(idx + v2 + 1) >> 1]));
                            pmwt[idx1] = rbvarm / (rbvarp + rbvarm);
                        }
                    }

                    // STAGE 9: Final Green interpolation and Chrominance interpolation
                    float* rbint = delhvsqsum;
                    for (int rr = 10; rr < rr1 - 10; rr++) {
                        for (int cc = 10 + (fc_rt(rr, 2) & 1), idx = rr * TS + cc, idx1 = idx >> 1; cc < cc1 - 10; cc += 2, idx += 2, idx1++) {
                            float pmwtalt = 0.25f * (pmwt[(idx - m1) >> 1] + pmwt[(idx + p1) >> 1] + pmwt[(idx - p1) >> 1] + pmwt[(idx + m1) >> 1]);
                            if (fabsf(0.5f - pmwt[idx1]) < fabsf(0.5f - pmwtalt)) pmwt[idx1] = pmwtalt;
                            rbint[idx1] = xdiv2f(cfa[idx] + rbm[idx1] * (1.f - pmwt[idx1]) + rbp[idx1] * pmwt[idx1]);
                        }
                    }
                    for (int rr = 12; rr < rr1 - 12; rr++) {
                        for (int cc = 12 + (fc_rt(rr, 2) & 1), idx = rr * TS + cc, idx1 = idx >> 1; cc < cc1 - 12; cc += 2, idx += 2, idx1++) {
                            if (fabsf(0.5f - pmwt[idx1]) >= fabsf(0.5f - hvwt[idx1])) continue;
                            float cru = xmul2f(cfa[idx-v1])/(eps+rbint[idx1]+rbint[idx1-v1]);
                            float crd = xmul2f(cfa[idx+v1])/(eps+rbint[idx1]+rbint[idx1+v1]);
                            float crl = xmul2f(cfa[idx-1])/(eps+rbint[idx1]+rbint[idx1-1]);
                            float crr = xmul2f(cfa[idx+1])/(eps+rbint[idx1]+rbint[idx1+1]);
                            float gu = fabsf(1.f - cru) < arthresh ? rbint[idx1] * cru : cfa[idx-v1] + xdiv2f(rbint[idx1] - rbint[idx1-v1]);
                            float gd = fabsf(1.f - crd) < arthresh ? rbint[idx1] * crd : cfa[idx+v1] + xdiv2f(rbint[idx1] - rbint[idx1+v1]);
                            float gl = fabsf(1.f - crl) < arthresh ? rbint[idx1] * crl : cfa[idx-1] + xdiv2f(rbint[idx1] - rbint[idx1-1]);
                            float gr = fabsf(1.f - crr) < arthresh ? rbint[idx1] * crr : cfa[idx+1] + xdiv2f(rbint[idx1] - rbint[idx1+1]);
                            float Gintv = (dirwts0[idx - v1] * gd + dirwts0[idx + v1] * gu) / (dirwts0[idx + v1] + dirwts0[idx - v1]);
                            float Ginth = (dirwts1[idx - 1] * gr + dirwts1[idx + 1] * gl) / (dirwts1[idx - 1] + dirwts1[idx + 1]);
                            if (Gintv < rbint[idx1]) { if (xmul2f(Gintv) < rbint[idx1]) Gintv = median(Gintv, cfa[idx - v1], cfa[idx + v1]); else { float vwt2 = xmul2f(rbint[idx1] - Gintv) / (eps + Gintv + rbint[idx1]); Gintv = vwt2 * Gintv + (1.f - vwt2) * median(Gintv, cfa[idx - v1], cfa[idx + v1]); } }
                            if (Ginth < rbint[idx1]) { if (xmul2f(Ginth) < rbint[idx1]) Ginth = median(Ginth, cfa[idx - 1], cfa[idx + 1]); else { float hwt2 = xmul2f(rbint[idx1] - Ginth) / (eps + Ginth + rbint[idx1]); Ginth = hwt2 * Ginth + (1.f - hwt2) * median(Ginth, cfa[idx - 1], cfa[idx + 1]); } }
                            if (Ginth > clip_pt) Ginth = median(Ginth, cfa[idx - 1], cfa[idx + 1]);
                            if (Gintv > clip_pt) Gintv = median(Gintv, cfa[idx - v1], cfa[idx + v1]);
                            rgbgreen[idx] = Ginth * (1.f - hvwt[idx1]) + Gintv * hvwt[idx1];
                            Dgrb0[idx1] = rgbgreen[idx] - cfa[idx];
                        }
                    }
                    float* Dgrb1 = hcd;
                    
                    // Dgrb1初期化
                    
                    for (int rr = 13 - ey; rr < rr1 - 12; rr += 2) {
                        for (int idx1 = (rr * TS + 13 - ex) >> 1; idx1 < (rr * TS + cc1 - 12) >> 1; idx1++) {
                            // 処理範囲内のコピー
                            Dgrb1[idx1] = Dgrb0[idx1];
                            Dgrb0[idx1] = 0;
                        }
                    }
                    for (int rr = 14; rr < rr1 - 14; rr++) {
                        for (int cc = 14 + (fc_rt(rr, 2) & 1), idx = rr * TS + cc; cc < cc1 - 14; cc += 2, idx += 2) {
                            // In RawTherapee, R=0, B=2. c becomes 1 for R-sites, 0 for B-sites
                            int c = 1 - fc_rt(rr, cc) / 2;
                            // Dgrb0 is for G-R, Dgrb1 is for G-B. But the interpolation logic uses the *other* color's buffer.
                            float* Dgrb_c = c ? Dgrb1 : Dgrb0;
                            float wtnw = 1.f / (eps + fabsf(Dgrb_c[(idx - m1) >> 1] - Dgrb_c[(idx + m1) >> 1]) + fabsf(Dgrb_c[(idx - m1) >> 1] - Dgrb_c[(idx - m3) >> 1]) + fabsf(Dgrb_c[(idx + m1) >> 1] - Dgrb_c[(idx - m3) >> 1]));
                            float wtne = 1.f / (eps + fabsf(Dgrb_c[(idx + p1) >> 1] - Dgrb_c[(idx - p1) >> 1]) + fabsf(Dgrb_c[(idx + p1) >> 1] - Dgrb_c[(idx + p3) >> 1]) + fabsf(Dgrb_c[(idx - p1) >> 1] - Dgrb_c[(idx + p3) >> 1]));
                            float wtsw = 1.f / (eps + fabsf(Dgrb_c[(idx - p1) >> 1] - Dgrb_c[(idx + p1) >> 1]) + fabsf(Dgrb_c[(idx - p1) >> 1] - Dgrb_c[(idx + m3) >> 1]) + fabsf(Dgrb_c[(idx + p1) >> 1] - Dgrb_c[(idx - p3) >> 1]));
                            float wtse = 1.f / (eps + fabsf(Dgrb_c[(idx + m1) >> 1] - Dgrb_c[(idx - m1) >> 1]) + fabsf(Dgrb_c[(idx + m1) >> 1] - Dgrb_c[(idx - p3) >> 1]) + fabsf(Dgrb_c[(idx - m1) >> 1] - Dgrb_c[(idx + m3) >> 1]));
                            // The buffer to *write to* is the original color's buffer.
                            Dgrb_c = c ? Dgrb1 : Dgrb0;
                            Dgrb_c[idx >> 1] = (wtnw * (1.325f * Dgrb_c[(idx - m1) >> 1] - 0.175f * Dgrb_c[(idx - m3) >> 1] - 0.075f * (Dgrb_c[(idx - m1 - 2) >> 1] + Dgrb_c[(idx - m1 - v2) >> 1])) +
                                            wtne * (1.325f * Dgrb_c[(idx + p1) >> 1] - 0.175f * Dgrb_c[(idx + p3) >> 1] - 0.075f * (Dgrb_c[(idx + p1 + 2) >> 1] + Dgrb_c[(idx + p1 + v2) >> 1])) +
                                            wtsw * (1.325f * Dgrb_c[(idx - p1) >> 1] - 0.175f * Dgrb_c[(idx - p3) >> 1] - 0.075f * (Dgrb_c[(idx - p1 - 2) >> 1] + Dgrb_c[(idx - p1 - v2) >> 1])) +
                                            wtse * (1.325f * Dgrb_c[(idx + m1) >> 1] - 0.175f * Dgrb_c[(idx + m3) >> 1] - 0.075f * (Dgrb_c[(idx + m1 + 2) >> 1] + Dgrb_c[(idx + m1 + v2) >> 1]))) / (wtnw + wtne + wtsw + wtse);
                        }
                    }
                    
                    // === STAGE 10: Final Output Composition (BUG FIXED) ===
                    for (int rr = 16; rr < rr1 - 16; rr++) {
                        int row = rr + top; 
                        if (row < 0 || row >= (int)height_) continue;
                        for (int cc = 16; cc < cc1 - 16; cc++) {
                            int col = cc + left; 
                            if (col < 0 || col >= (int)width_) continue;
                            
                            int out_idx = row * width_ + col;
                            int tile_idx = rr * TS + cc;
                            
                            float r, g, b;

                            // Use a direct and robust check for the G-site.
                            if (fc_rt(row, col) == 1) { // G site
                                g = cfa[tile_idx];
                                float wsum_inv = 1.0f / (hvwt[(tile_idx - v1) >> 1] + 2.f - hvwt[(tile_idx + 1) >> 1] - hvwt[(tile_idx - 1) >> 1] + hvwt[(tile_idx + v1) >> 1]);                            float r_diff = (hvwt[(tile_idx-v1)>>1]*Dgrb0[(tile_idx-v1)>>1] + (1.f-hvwt[(tile_idx+1)>>1])*Dgrb0[(tile_idx+1)>>1] + (1.f-hvwt[(tile_idx-1)>>1])*Dgrb0[(tile_idx-1)>>1] + hvwt[(tile_idx+v1)>>1]*Dgrb0[(tile_idx+v1)>>1]) * wsum_inv;
                                float b_diff = (hvwt[(tile_idx-v1)>>1]*Dgrb1[(tile_idx-v1)>>1] + (1.f-hvwt[(tile_idx+1)>>1])*Dgrb1[(tile_idx+1)>>1] + (1.f-hvwt[(tile_idx-1)>>1])*Dgrb1[(tile_idx-1)>>1] + hvwt[(tile_idx+v1)>>1]*Dgrb1[(tile_idx+v1)>>1]) * wsum_inv;
                                r = g - r_diff;
                                b = g - b_diff;
                            } else { // R or B site
                                g = rgbgreen[tile_idx];
                                // ネイティブの色(cfa)を使わず、Dgrb0とDgrb1から両方の色を計算する
                                r = g - Dgrb0[tile_idx >> 1];
                                b = g - Dgrb1[tile_idx >> 1];
                                // R/B-site処理完了
                            }
                            rgb_buffer_.image[out_idx][0] = std::max(0.f, r);
                            rgb_buffer_.image[out_idx][1] = std::max(0.f, g);
                            rgb_buffer_.image[out_idx][2] = std::max(0.f, b);
                        }
                    }
                }
            }
        }
    }
};

} // anonymous namespace


bool CPUAccelerator::demosaic_bayer_amaze(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, uint32_t filters, const float (&cam_mul)[4], uint16_t maximum_value) {
    if (!initialized_ || !raw_buffer.is_valid() || !rgb_buffer.is_valid() ||
        raw_buffer.width != rgb_buffer.width || raw_buffer.height != rgb_buffer.height) {
        std::cerr << "❌ AMaZE Demosaic (RT Port): Invalid buffers or initialization state." << std::endl;
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    // AMaZE demosaic processing
    
    try {
        AMaZE_Processor_RT amaze_proc(raw_buffer, rgb_buffer, filters, cam_mul, maximum_value);
        amaze_proc.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ An exception occurred during AMaZE processing: " << e.what() << std::endl;
        return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();    
    std::chrono::duration<double> diff = end_time - start_time;
    last_processing_time_ = diff.count();
    
    border_interpolate(rgb_buffer, filters, 4);
    
    return true;
}

//===================================================================
// XTrans Demosaicing - RawTherapee Complete Port
//===================================================================

namespace {

//#define LIM(x, mn, mx) std::max((mn), std::min((mx), (x)))
#define CLIP(x) (x)
#define RGB_BUFFER(r, c, f) rgb_buffer_.image[(r) * width_ + (c)][f]

class XTrans_Processor {
private:
    const libraw_enhanced::ImageBuffer& raw_buffer_;
    libraw_enhanced::ImageBufferFloat& rgb_buffer_;
    const char (&xtrans_)[6][6];
    const float (&color_matrix_)[3][4];
    uint32_t width_, height_;
    const float maximum_value_;

    struct s_minmaxgreen {
        float min;
        float max;
    };

    inline int fcol_xtrans_(int row, int col) const {
        return fcol_xtrans(row, col, xtrans_);
    }

    inline int isgreen_(int row, int col) {
        return (xtrans_[(row) % 3][(col) % 3] & 1);
    }

    inline float raw_buffer(int row, int col) const {
        int f = fcol_xtrans_(row, col);
        return static_cast<float>(raw_buffer_.image[row * width_ + col][f]) / maximum_value_;
    }

    inline float raw_buffer_hex(int row, int col, short hex) const {
        int pos = row * width_ + col + hex;
        int f = fcol_xtrans_(pos / width_, pos % width_);
        return static_cast<float>(raw_buffer_.image[pos][f]) / maximum_value_;
    }
    
    void cielab2(const float (*rgb)[3], float* l, float* a, float* b, 
                const int width, const int height, const int labWidth, 
                const float xyz_cam[3][3]) 
    {
        static constexpr double eps = 216.0f / 24389.0f;
        static constexpr double kappa = 24389.0f / 27.0f;
        static constexpr int table_size = 1<<16;
        
        // cbrtテーブルの初期化（スレッドセーフな初期化）
        static const std::vector<float> cbrt = [] {
            std::vector<float> table(table_size);
            for (int i = 0; i < table_size; ++i) {
                double r = i / static_cast<double>(table_size - 1);
                table[i] = static_cast<float>(r > eps ? std::cbrt(r) : (kappa * r + 16.0f) / 116.0f);
            }
            return table;
        }();

        if (!rgb) return; // テーブル初期化のみ実行

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < labWidth; ++j) {
                // RGBからXYZへの変換
                float xyz[3] = {0.0f, 0.0f, 0.0f};
                for (int c = 0; c < 3; ++c) {
                    float val = rgb[i * width + j][c];
                    xyz[0] += xyz_cam[0][c] * val;
                    xyz[1] += xyz_cam[1][c] * val;
                    xyz[2] += xyz_cam[2][c] * val;
                }

                // XYZ値を0.0-1.0にクリップしてテーブルアクセス
                auto clamp = [](float v) { 
                    return std::max(0.0f, std::min(1.0f, v)); 
                };
                int idx0 = static_cast<int>(clamp(xyz[0]) * (table_size - 1));
                int idx1 = static_cast<int>(clamp(xyz[1]) * (table_size - 1));
                int idx2 = static_cast<int>(clamp(xyz[2]) * (table_size - 1));

                // L*a*b*値の計算
                const int idx = i * labWidth + j;
                l[idx] = 116.0f * cbrt[idx1] - 16.0f;
                a[idx] = 500.0f * (cbrt[idx0] - cbrt[idx1]);
                b[idx] = 200.0f * (cbrt[idx1] - cbrt[idx2]);
            }
        }
    }

    void cielab3(const float (*rgb)[3], float* l, float* a, float* b, 
                const int width, const int height, const int labWidth, 
                const float xyz_cam[3][3]) 
    {
        static constexpr float eps = 216.0f / 24389.0f;
        static constexpr float kappa = 24389.0f / 27.0f;
        
        // XYZからLabへの変換関数
        auto xyz_to_lab = [](float x, float y, float z) -> std::tuple<float, float, float> {
            // 値を安全な範囲にクリップ
            auto clamp = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };
            x = clamp(x);
            y = clamp(y);
            z = clamp(z);
            
            // Lab変換
            auto f = [](float t) -> float {
                return t > eps ? std::cbrt(t) : (kappa * t + 16.0f) / 116.0f;
            };
            
            float fx = f(x);
            float fy = f(y);
            float fz = f(z);
            
            float L = 116.0f * fy - 16.0f;
            float a_val = 500.0f * (fx - fy);
            float b_val = 200.0f * (fy - fz);
            
            return {L, a_val, b_val};
        };

        if (!rgb) return; // テーブルがないので何もしない

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < labWidth; ++j) {
                // RGBからXYZへの変換
                float xyz[3] = {0.0f, 0.0f, 0.0f};
                for (int c = 0; c < 3; ++c) {
                    float val = rgb[i * width + j][c];
                    xyz[0] += xyz_cam[0][c] * val;
                    xyz[1] += xyz_cam[1][c] * val;
                    xyz[2] += xyz_cam[2][c] * val;
                }

                // XYZからLabへの直接計算
                auto [L_val, a_val, b_val] = xyz_to_lab(xyz[0], xyz[1], xyz[2]);
                
                const int idx = i * labWidth + j;
                l[idx] = L_val;
                a[idx] = a_val;
                b[idx] = b_val;
            }
        }
    }
    void xtrans_border_interpolate(int border)
    {
        const float weight[3][3] = {
            {0.25f, 0.5f, 0.25f},
            {0.5f,  0.f,  0.5f},
            {0.25f, 0.5f, 0.25f}
        };

        for (int row = 0; row < (int)height_; row++) {
            for (int col = 0; col < (int)width_; col++) {
                if (col == border && row >= border && row < (int)height_ - border) {
                    col = width_ - border;
                }

                float sum[6] = {0.f};

                for (int y = std::max(0, row - 1), v = row == 0 ? 0 : -1; y <= std::min(row + 1, (int)height_ - 1); y++, v++) {
                    for (int x = std::max(0, col - 1), h = col == 0 ? 0 : -1; x <= std::min(col + 1, (int)width_ - 1); x++, h++) {
                        int f = fcol_xtrans_(y, x);
                        sum[f] += raw_buffer(y, x) * weight[v + 1][h + 1];
                        sum[f + 3] += weight[v + 1][h + 1];
                    }
                }

                switch(fcol_xtrans_(row, col)) {
                case 0:
                    RGB_BUFFER(row, col, 0) = raw_buffer(row, col);
                    RGB_BUFFER(row, col, 1) = (sum[1] / sum[4]);
                    RGB_BUFFER(row, col, 2) = (sum[2] / sum[5]);
                    break;

                case 1:
                    if(sum[3] == 0.f) { // at the 4 corner pixels it can happen, that we have only green pixels in 2x2 area
                        RGB_BUFFER(row, col, 0) = RGB_BUFFER(row, col, 1) = RGB_BUFFER(row, col, 2) = raw_buffer(row, col);
                    } else {
                        RGB_BUFFER(row, col, 0) = (sum[0] / sum[3]);
                        RGB_BUFFER(row, col, 1) = raw_buffer(row, col);
                        RGB_BUFFER(row, col, 2) = (sum[2] / sum[5]);
                    }

                    break;

                case 2:
                    RGB_BUFFER(row, col, 0) = (sum[0] / sum[3]);
                    RGB_BUFFER(row, col, 1) = (sum[1] / sum[4]);
                    RGB_BUFFER(row, col, 2) = raw_buffer(row, col);
                }
            }
        }
    }

public:
    XTrans_Processor(const libraw_enhanced::ImageBuffer& raw, libraw_enhanced::ImageBufferFloat& rgb, 
                    const char (&xtrans)[6][6], const float (&color_matrix)[3][4], uint16_t max_val)
        : raw_buffer_(raw), rgb_buffer_(rgb), xtrans_(xtrans), color_matrix_(color_matrix),
            width_(raw.width), height_(raw.height), maximum_value_((float)max_val){
    }

    void run_3pass() {
        constexpr int ts = 114;
        
        constexpr short orth[12] = { 1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1 };
        constexpr short patt[2][16] = {
            { 0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0 },
            { 0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1 }
        };
        constexpr short dir[4] = { 1, ts, ts + 1, ts - 1 };
        
        float xyz_cam[3][3];

/*
        constexpr float xyz_rgb[3][3] = {
            { 0.412453f, 0.357580f, 0.180423f },
            { 0.212671f, 0.715160f, 0.072169f },
            { 0.019334f, 0.119193f, 0.950227f }
        };

        constexpr float d65_white[3] = { 0.950456f, 1.0f, 1.088754f };

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                xyz_cam[i][j] = 0.0f;
                for (int k = 0; k < 3; k++) {
                    xyz_cam[i][j] += xyz_rgb[i][k] * color_matrix_[k][j] / d65_white[i];
                }
            }
        }
*/

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                xyz_cam[i][j] = color_matrix_[i][j];
            }
        }

        /* Map a green hexagon around each non-green pixel and vice versa:  */
        ushort sgrow = 0, sgcol = 0;
        short allhex[2][3][3][8];
        {
            int gint, d, h, v, ng, row, col;

            for (row = 0; row < 3; row++) {
                for (col = 0; col < 3; col++) {
                    gint = isgreen_(row, col);

                    for (ng = d = 0; d < 10; d += 2) {
                        if (isgreen_(row + orth[d] + 6, col + orth[d + 2] + 6)) {
                            ng = 0;
                        } else {
                            ng++;
                        }

                        if (ng == 4) {
                            // if there are four non-green pixels adjacent in cardinal
                            // directions, this is the solitary green pixel
                            sgrow = row;
                            sgcol = col;
                        }

                        if (ng == gint + 1) {
                            for (int c = 0; c < 8; c++) {
                                v = orth[d] * patt[gint][c * 2] + orth[d + 1] * patt[gint][c * 2 + 1];
                                h = orth[d + 2] * patt[gint][c * 2] + orth[d + 3] * patt[gint][c * 2 + 1];
                                allhex[0][row][col][c ^ (gint * 2 & d)] = h + v * width_;
                                allhex[1][row][col][c ^ (gint * 2 & d)] = h + v * ts;
                            }
                        }
                    }
                }
            }
        }

        const int passes = 3;
        const int ndir = 4 << (passes > 1);
        cielab2 (nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr);

        int RightShift[3];

        for(int row = 0; row < 3; row++) {
            // count number of green pixels in three cols
            int greencount = 0;

            for(int col = 0; col < 3; col++) {
                greencount += isgreen_(row, col);
            }

            RightShift[row] = (greencount == 2);
        }

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<float> rgb_buffer_tile(ndir * ts * ts * 3);
            std::vector<float> lab_buffer_tile(3 * (ts - 8) * (ts - 8));
            std::vector<float> drv_buffer_tile(ndir * (ts - 10) * (ts - 10));
            std::vector<uint8_t> homo_buffer_tile(ndir * ts * ts);
            std::vector<uint8_t> homosum_buffer_tile(ndir * ts * ts);
            std::vector<uint8_t> homosummax_buffer_tile(ts * ts);
            std::vector<s_minmaxgreen> greenminmax_buffer_tile(ts * ts / 2);

            auto rgb = reinterpret_cast<float(*)[ts][ts][3]>(rgb_buffer_tile.data());
            auto lab = reinterpret_cast<float(*)[ts - 8][ts - 8]>(lab_buffer_tile.data());
            auto drv = reinterpret_cast<float(*)[ts - 10][ts - 10]>(drv_buffer_tile.data());
            auto homo = reinterpret_cast<uint8_t(*)[ts][ts]>(homo_buffer_tile.data());
            auto homosum = reinterpret_cast<uint8_t(*)[ts][ts]>(homosum_buffer_tile.data());
            auto homosummax = reinterpret_cast<uint8_t(*)[ts]>(homosummax_buffer_tile.data());
            auto greenminmaxtile = reinterpret_cast<s_minmaxgreen(*)[ts / 2]>(greenminmax_buffer_tile.data());

#ifdef _OPENMP
            #pragma omp for collapse(2) schedule(dynamic, 4) nowait
#endif
            //std::cout << "[DEBUG] X-Trans tile loop begin" << std::endl;
            for (int top = 3; top < (int)height_ - 19; top += ts - 16) {
                for (int left = 3; left < (int)width_ - 19; left += ts - 16) {
                    int mrow = std::min (top + ts, (int)height_ - 3);
                    int mcol = std::min (left + ts, (int)width_ - 3);

                    /* Set greenmin and greenmax to the minimum and maximum allowed values: */
                    for (int row = top; row < mrow; row++) {
                        // find first non-green pixel
                        int leftstart = left;

                        for(; leftstart < mcol; leftstart++)
                            if(!isgreen_(row, leftstart)) {
                                break;
                            }

                        int coloffset = (RightShift[row % 3] == 1 ? 3 : 1 + (fcol_xtrans_(row, leftstart + 1) & 1));

                        float minval = std::numeric_limits<float>::max();
                        float maxval = 0.f;

                        if(coloffset == 3) {
                            short *hex = allhex[0][row % 3][leftstart % 3];

                            for (int col = leftstart; col < mcol; col += coloffset) {
                                minval = std::numeric_limits<float>::max();
                                maxval = 0.f;

                                for(int c = 0; c < 6; c++) {
                                    float val = raw_buffer_hex(row, col, hex[c]);

                                    minval = minval < val ? minval : val;
                                    maxval = maxval > val ? maxval : val;
                                }

                                greenminmaxtile[row - top][(col - left) >> 1].min = minval;
                                greenminmaxtile[row - top][(col - left) >> 1].max = maxval;
                            }
                        } else {
                            int col = leftstart;

                            if(coloffset == 2) {
                                minval = std::numeric_limits<float>::max();
                                maxval = 0.f;
                                short *hex = allhex[0][row % 3][col % 3];

                                for(int c = 0; c < 6; c++) {
                                    float val = raw_buffer_hex(row, col, hex[c]);

                                    minval = minval < val ? minval : val;
                                    maxval = maxval > val ? maxval : val;
                                }

                                greenminmaxtile[row - top][(col - left) >> 1].min = minval;
                                greenminmaxtile[row - top][(col - left) >> 1].max = maxval;
                                col += 2;
                            }

                            short *hex = allhex[0][row % 3][col % 3];

                            for (; col < mcol - 1; col += 3) {
                                minval = std::numeric_limits<float>::max();
                                maxval = 0.f;

                                for(int c = 0; c < 6; c++) {
                                    float val = raw_buffer_hex(row, col, hex[c]);

                                    minval = minval < val ? minval : val;
                                    maxval = maxval > val ? maxval : val;
                                }

                                greenminmaxtile[row - top][(col - left) >> 1].min = minval;
                                greenminmaxtile[row - top][(col - left) >> 1].max = maxval;
                                greenminmaxtile[row - top][(col + 1 - left) >> 1].min = minval;
                                greenminmaxtile[row - top][(col + 1 - left) >> 1].max = maxval;
                            }

                            if(col < mcol) {
                                minval = std::numeric_limits<float>::max();
                                maxval = 0.f;

                                for(int c = 0; c < 6; c++) {
                                    float val = raw_buffer_hex(row, col, hex[c]);

                                    minval = minval < val ? minval : val;
                                    maxval = maxval > val ? maxval : val;
                                }

                                greenminmaxtile[row - top][(col - left) >> 1].min = minval;
                                greenminmaxtile[row - top][(col - left) >> 1].max = maxval;
                            }
                        }
                    }

                    memset(rgb, 0, ts * ts * 3 * sizeof(float));

                    for (int row = top; row < mrow; row++)
                        for (int col = left; col < mcol; col++) {
                            rgb[0][row - top][col - left][fcol_xtrans_(row, col)] = raw_buffer(row, col);
                        }

                    for(int c = 0; c < 3; c++) {
                        memcpy (rgb[c + 1], rgb[0], sizeof * rgb);
                    }

                    /* Interpolate green horizontally, vertically, and along both diagonals: */
                    // std::cout << "[DEBUG] Interpolate green horizontally, vertically, and along both diagonals:" << std::endl;
                    for (int row = top; row < mrow; row++) {
                        // find first non-green pixel
                        int leftstart = left;

                        for(; leftstart < mcol; leftstart++)
                            if(!isgreen_(row, leftstart)) {
                                break;
                            }

                        int coloffset = (RightShift[row % 3] == 1 ? 3 : 1 + (fcol_xtrans_(row, leftstart + 1) & 1));

                        if(coloffset == 3) {
                            short *hex = allhex[0][row % 3][leftstart % 3];

                            for (int col = leftstart; col < mcol; col += coloffset) {
                                float color[4];
                                color[0] = 0.6796875f * (raw_buffer_hex(row, col, hex[1]) + raw_buffer_hex(row, col, hex[0])) -
                                        0.1796875f * (raw_buffer_hex(row, col, 2 * hex[1]) + raw_buffer_hex(row, col, 2 * hex[0]));
                                color[1] = 0.87109375f * raw_buffer_hex(row, col, hex[3]) + raw_buffer_hex(row, col, hex[2]) * 0.12890625f +
                                        0.359375f * (raw_buffer_hex(row, col, 0) - raw_buffer_hex(row, col, -hex[2]));

                                for(int c = 0; c < 2; c++)
                                    color[2 + c] = 0.640625f * raw_buffer_hex(row, col, hex[4 + c]) + 0.359375f * raw_buffer_hex(row, col, -2 * hex[4 + c]) + 0.12890625f *
                                                (2.f * raw_buffer_hex(row, col, 0) - raw_buffer_hex(row, col, 3 * hex[4 + c]) - raw_buffer_hex(row, col, -3 * hex[4 + c]));

                                for(int c = 0; c < 4; c++) {
                                    rgb[c][row - top][col - left][1] = LIM(color[c], greenminmaxtile[row - top][(col - left) >> 1].min, greenminmaxtile[row - top][(col - left) >> 1].max);
                                }
                            }
                        } else {
                            short *hexmod[2];
                            hexmod[0] = allhex[0][row % 3][leftstart % 3];
                            hexmod[1] = allhex[0][row % 3][(leftstart + coloffset) % 3];

                            for (int col = leftstart, hexindex = 0; col < mcol; col += coloffset, coloffset ^= 3, hexindex ^= 1) {
                                short *hex = hexmod[hexindex];
                                float color[4];
                                color[0] = 0.6796875f * (raw_buffer_hex(row, col, hex[1]) + raw_buffer_hex(row, col, hex[0])) -
                                        0.1796875f * (raw_buffer_hex(row, col, 2 * hex[1]) + raw_buffer_hex(row, col, 2 * hex[0]));
                                color[1] = 0.87109375f *  raw_buffer_hex(row, col, hex[3]) + raw_buffer_hex(row, col, hex[2]) * 0.12890625f +
                                        0.359375f * (raw_buffer_hex(row, col, 0) - raw_buffer_hex(row, col, -hex[2]));

                                for(int c = 0; c < 2; c++)
                                    color[2 + c] = 0.640625f * raw_buffer_hex(row, col, hex[4 + c]) + 0.359375f * raw_buffer_hex(row, col, -2 * hex[4 + c]) + 0.12890625f *
                                                (2.f * raw_buffer_hex(row, col, 0) - raw_buffer_hex(row, col, 3 * hex[4 + c]) - raw_buffer_hex(row, col, -3 * hex[4 + c]));

                                for(int c = 0; c < 4; c++) {
                                    rgb[c ^ 1][row - top][col - left][1] = LIM(color[c], greenminmaxtile[row - top][(col - left) >> 1].min, greenminmaxtile[row - top][(col - left) >> 1].max);
                                }
                            }
                        }
                    }

                    for (int pass = 0; pass < passes; pass++) {
                        if (pass == 1) {
                            memcpy (rgb += 4, rgb_buffer_tile.data(), 4 * sizeof * rgb);
                        }

                        /* Recalculate green from interpolated values of closer pixels: */
                        //std::cout << "[DEBUG] Recalculate green from interpolated values of closer pixels: " << std::endl;
                        if (pass) {
                            for (int row = top + 2; row < mrow - 2; row++) {
                                int leftstart = left + 2;

                                for(; leftstart < mcol - 2; leftstart++)
                                    if(!isgreen_(row, leftstart)) {
                                        break;
                                    }

                                int coloffset = (RightShift[row % 3] == 1 ? 3 : 1 + (fcol_xtrans_(row, leftstart + 1) & 1));

                                if(coloffset == 3) {
                                    int f = fcol_xtrans_(row, leftstart);
                                    short *hex = allhex[1][row % 3][leftstart % 3];

                                    for (int col = leftstart; col < mcol - 2; col += coloffset, f ^= 2) {
                                        for (int d = 3; d < 6; d++) {
                                            float (*rix)[3] = &rgb[(d - 2)][row - top][col - left];
                                            float val = 0.33333333f * (rix[-2 * hex[d]][1] + 2 * (rix[hex[d]][1] - rix[hex[d]][f])
                                                                    - rix[-2 * hex[d]][f]) + rix[0][f];
                                            rix[0][1] = LIM(val, greenminmaxtile[row - top][(col - left) >> 1].min, greenminmaxtile[row - top][(col - left) >> 1].max);
                                        }
                                    }
                                } else {
                                    int f = fcol_xtrans_(row, leftstart);
                                    short *hexmod[2];
                                    hexmod[0] = allhex[1][row % 3][leftstart % 3];
                                    hexmod[1] = allhex[1][row % 3][(leftstart + coloffset) % 3];

                                    for (int col = leftstart, hexindex = 0; col < mcol - 2; col += coloffset, coloffset ^= 3, f = f ^ (coloffset & 2), hexindex ^= 1 ) {
                                        short *hex = hexmod[hexindex];

                                        for (int d = 3; d < 6; d++) {
                                            float (*rix)[3] = &rgb[(d - 2) ^ 1][row - top][col - left];
                                            float val = 0.33333333f * (rix[-2 * hex[d]][1] + 2 * (rix[hex[d]][1] - rix[hex[d]][f])
                                                                    - rix[-2 * hex[d]][f]) + rix[0][f];
                                            rix[0][1] = LIM(val, greenminmaxtile[row - top][(col - left) >> 1].min, greenminmaxtile[row - top][(col - left) >> 1].max);
                                        }
                                    }
                                }
                            }
                        }

                        /* Interpolate red and blue values for solitary green pixels:   */
                        //std::cout << "[DEBUG] Interpolate red and blue values for solitary green pixels:" << std::endl;
                        int sgstartcol = (left - sgcol + 4) / 3 * 3 + sgcol;
                        float color[3][6];

                        for (int row = (top - sgrow + 4) / 3 * 3 + sgrow; row < mrow - 2; row += 3) {
                            for (int col = sgstartcol, h = fcol_xtrans_(row, col + 1); col < mcol - 2; col += 3, h ^= 2) {
                                float (*rix)[3] = &rgb[0][row - top][col - left];
                                float diff[6] = {0.f};

                                for (int i = 1, d = 0; d < 6; d++, i ^= ts ^ 1, h ^= 2) {
                                    for (int c = 0; c < 2; c++, h ^= 2) {
                                        float g = rix[0][1] + rix[0][1] - rix[i << c][1] - rix[-i << c][1];
                                        color[h][d] = g + rix[i << c][h] + rix[-i << c][h];

                                        if (d > 1)
                                            diff[d] += SQR (rix[i << c][1] - rix[-i << c][1]
                                                            - rix[i << c][h] + rix[-i << c][h]) + SQR(g);
                                    }

                                    if (d > 2 && (d & 1))    // 3, 5
                                        if (diff[d - 1] < diff[d])
                                            for(int c = 0; c < 2; c++) {
                                                color[c * 2][d] = color[c * 2][d - 1];
                                            }

                                    if ((d & 1) || d < 2) { // d: 0, 1, 3, 5
                                        for(int c = 0; c < 2; c++) {
                                            rix[0][c * 2] = CLIP(0.5f * color[c * 2][d]);
                                        }

                                        rix += ts * ts;
                                    }
                                }
                            }
                        }

                        /* Interpolate red for blue pixels and vice versa:      */
                        //std::cout << "[DEBUG] Interpolate red for blue pixels and vice versa: " << std::endl;
                        for (int row = top + 3; row < mrow - 3; row++) {
                            int leftstart = left + 3;

                            for(; leftstart < mcol - 1; leftstart++)
                                if(!isgreen_(row, leftstart)) {
                                    break;
                                }

                            int coloffset = (RightShift[row % 3] == 1 ? 3 : 1);
                            int c = ((row - sgrow) % 3) ? ts : 1;
                            int h = 3 * (c ^ ts ^ 1);

                            if(coloffset == 3) {
                                int f = 2 - fcol_xtrans_(row, leftstart);

                                for (int col = leftstart; col < mcol - 3; col += coloffset, f ^= 2) {
                                    float (*rix)[3] = &rgb[0][row - top][col - left];

                                    for (int d = 0; d < 4; d++, rix += ts * ts) {
                                        int i = d > 1 || ((d ^ c) & 1) ||
                                                ((fabsf(rix[0][1] - rix[c][1]) + fabsf(rix[0][1] - rix[-c][1])) < 2.f * (fabsf(rix[0][1] - rix[h][1]) + fabsf(rix[0][1] - rix[-h][1]))) ? c : h;

                                        rix[0][f] = CLIP(rix[0][1] + 0.5f * (rix[i][f] + rix[-i][f] - rix[i][1] - rix[-i][1]));
                                    }
                                }
                            } else {
                                coloffset = fcol_xtrans_(row, leftstart + 1) == 1 ? 2 : 1;
                                int f = 2 - fcol_xtrans_(row, leftstart);

                                for (int col = leftstart; col < mcol - 3; col += coloffset, coloffset ^= 3, f = f ^ (coloffset & 2) ) {
                                    float (*rix)[3] = &rgb[0][row - top][col - left];

                                    for (int d = 0; d < 4; d++, rix += ts * ts) {
                                        int i = d > 1 || ((d ^ c) & 1) ||
                                                ((fabsf(rix[0][1] - rix[c][1]) + fabsf(rix[0][1] - rix[-c][1])) < 2.f * (fabsf(rix[0][1] - rix[h][1]) + fabsf(rix[0][1] - rix[-h][1]))) ? c : h;

                                        rix[0][f] = CLIP(rix[0][1] + 0.5f * (rix[i][f] + rix[-i][f] - rix[i][1] - rix[-i][1]));
                                    }
                                }
                            }
                        }

                        /* Fill in red and blue for 2x2 blocks of green:        */
                        //std::cout << "[DEBUG] Fill in red and blue for 2x2 blocks of green:" << std::endl;
                        // Find first row of 2x2 green
                        int topstart = top + 2;

                        for(; topstart < mrow - 2; topstart++)
                            if((topstart - sgrow) % 3) {
                                break;
                            }

                        int leftstart = left + 2;

                        for(; leftstart < mcol - 2; leftstart++)
                            if((leftstart - sgcol) % 3) {
                                break;
                            }

                        int coloffsetstart = 2 - (fcol_xtrans_(topstart, leftstart + 1) & 1);

                        for (int row = topstart; row < mrow - 2; row++) {
                            if ((row - sgrow) % 3) {
                                short *hexmod[2];
                                hexmod[0] = allhex[1][row % 3][leftstart % 3];
                                hexmod[1] = allhex[1][row % 3][(leftstart + coloffsetstart) % 3];

                                for (int col = leftstart, coloffset = coloffsetstart, hexindex = 0; col < mcol - 2; col += coloffset, coloffset ^= 3, hexindex ^= 1) {
                                    float (*rix)[3] = &rgb[0][row - top][col - left];
                                    short *hex = hexmod[hexindex];

                                    for (int d = 0; d < ndir; d += 2, rix += ts * ts) {
                                        if (hex[d] + hex[d + 1]) {
                                            float g = 3 * rix[0][1] - 2 * rix[hex[d]][1] - rix[hex[d + 1]][1];

                                            for (int c = 0; c < 4; c += 2) {
                                                rix[0][c] = CLIP((g + 2 * rix[hex[d]][c] + rix[hex[d + 1]][c]) * 0.33333333f);
                                            }
                                        } else {
                                            float g = 2 * rix[0][1] - rix[hex[d]][1] - rix[hex[d + 1]][1];

                                            for (int c = 0; c < 4; c += 2) {
                                                rix[0][c] = CLIP((g + rix[hex[d]][c] + rix[hex[d + 1]][c]) * 0.5f);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
    // end of multipass part

                    rgb = reinterpret_cast<float(*)[ts][ts][3]>(rgb_buffer_tile.data());
                    mrow -= top;
                    mcol -= left;

                    if(false) {
                        /* Convert to CIELab and differentiate in all directions:   */
                        //std::cout << "[DEBUG] Convert to CIELab and differentiate in all directions:" << std::endl;
                        // Original dcraw algorithm uses CIELab as perceptual space
                        // (presumably coming from original AHD) and converts taking
                        // camera matrix into account.  We use this in RT.
                        for (size_t d = 0; d < ndir; d++) {
                            cielab2(&rgb[d][4][4], &lab[0][0][0], &lab[1][0][0], &lab[2][0][0], ts, mrow - 8, ts - 8, xyz_cam);
                            int f = dir[d & 3];
                            f = f == 1 ? 1 : f - 8;

                            for (int row = 5; row < mrow - 5; row++) {
                                for (int col = 5; col < mcol - 5; col++) {
                                    float *l = &lab[0][row - 4][col - 4];
                                    float *a = &lab[1][row - 4][col - 4];
                                    float *b = &lab[2][row - 4][col - 4];

                                    float g = 2 * l[0] - l[f] - l[-f];
                                    drv[d][row - 5][col - 5] =  SQR(g)
                                                                + SQR((2 * a[0] - a[f] - a[-f] + g * 2.1551724f))
                                                                + SQR((2 * b[0] - b[f] - b[-f] - g * 0.86206896f));
                                }
                            }
                        }
                    } else {
                        // For 1-pass demosaic we use YPbPr which requires much
                        // less code and is nearly indistinguishable. It assumes the
                        // camera RGB is roughly linear.
                        //std::cout << "[DEBUG] For 1-pass demosaic we use YPbPr which requires much" << std::endl;
                        for (int d = 0; d < ndir; d++) {
                            float (*yuv)[ts - 8][ts - 8] = lab; // we use the lab buffer, which has the same dimensions
    #ifdef __ARM_NEON
                            float32x4_t zd2627v = vdupq_n_f32(0.2627f);
                            float32x4_t zd6780v = vdupq_n_f32(0.6780f);
                            float32x4_t zd0593v = vdupq_n_f32(0.0593f);
                            float32x4_t zd56433v = vdupq_n_f32(0.56433f);
                            float32x4_t zd67815v = vdupq_n_f32(0.67815f);
    #endif

                            for (int row = 4; row < mrow - 4; row++) {
                                int col = 4;
    #ifdef __ARM_NEON
                                for (; col < mcol - 7; col += 4) {
                                    // use ITU-R BT.2020 YPbPr, which is great, but could use
                                    // a better/simpler choice? note that imageop.h provides
                                    // dt_iop_RGB_to_YCbCr which uses Rec. 601 conversion,
                                    // which appears less good with specular highlights

                                    // Load 4 interleaved RGB pixels
                                    float32x4x3_t rgb_pixels = vld3q_f32((const float*)&rgb[d][row][col]);
                                    float32x4_t redv = rgb_pixels.val[0];
                                    float32x4_t greenv = rgb_pixels.val[1];
                                    float32x4_t bluev = rgb_pixels.val[2];
                                    
                                    // Calculate Y component: 0.2627*R + 0.6780*G + 0.0593*B
                                    float32x4_t yv = vmulq_f32(redv, zd2627v);
                                    yv = vmlaq_f32(yv, greenv, zd6780v);
                                    yv = vmlaq_f32(yv, bluev, zd0593v);
                                    
                                    // Calculate U/Pb component: (B - Y) * 0.56433
                                    float32x4_t uv = vmulq_f32(vsubq_f32(bluev, yv), zd56433v);
                                    
                                    // Calculate V/Pr component: (R - Y) * 0.67815
                                    float32x4_t vv = vmulq_f32(vsubq_f32(redv, yv), zd67815v);
                                    
                                    // Store results
                                    vst1q_f32(&yuv[0][row - 4][col - 4], yv);
                                    vst1q_f32(&yuv[1][row - 4][col - 4], uv);
                                    vst1q_f32(&yuv[2][row - 4][col - 4], vv);
                                }
    #endif
                                for (; col < mcol - 4; col++) {
                                    // use ITU-R BT.2020 YPbPr, which is great, but could use
                                    // a better/simpler choice? note that imageop.h provides
                                    // dt_iop_RGB_to_YCbCr which uses Rec. 601 conversion,
                                    // which appears less good with specular highlights
                                    float y = 0.2627f * rgb[d][row][col][0] + 0.6780f * rgb[d][row][col][1] + 0.0593f * rgb[d][row][col][2];
                                    yuv[0][row - 4][col - 4] = y;
                                    yuv[1][row - 4][col - 4] = (rgb[d][row][col][2] - y) * 0.56433f;
                                    yuv[2][row - 4][col - 4] = (rgb[d][row][col][0] - y) * 0.67815f;
                                }
                            }

                            int f = dir[d & 3];
                            f = f == 1 ? 1 : f - 8;

                            for (int row = 5; row < mrow - 5; row++) {
                                for (int col = 5; col < mcol - 5; col++) {
                                    float *y = &yuv[0][row - 4][col - 4];
                                    float *u = &yuv[1][row - 4][col - 4];
                                    float *v = &yuv[2][row - 4][col - 4];
                                    drv[d][row - 5][col - 5] = SQR(2 * y[0] - y[f] - y[-f])
                                                            + SQR(2 * u[0] - u[f] - u[-f])
                                                            + SQR(2 * v[0] - v[f] - v[-f]);
                                }
                            }
                        }
                    }

                    /* Build homogeneity maps from the derivatives:         */
                    //std::cout << "[DEBUG] Build homogeneity maps from the derivatives:" << std::endl;
    #ifdef __ARM_NEON
                    float32x4_t eightv = vdupq_n_f32(8.f);
                    float32x4_t zerov = vdupq_n_f32(0.f);
                    float32x4_t onev = vdupq_n_f32(1.f);
    #endif

                    for (int row = 6; row < mrow - 6; row++) {
                        int col = 6;
    #ifdef __ARM_NEON
                        for (; col < mcol - 9; col += 4) {
                            float32x4_t tr1v = vminq_f32(vld1q_f32(&drv[0][row - 5][col - 5]), vld1q_f32(&drv[1][row - 5][col - 5]));
                            float32x4_t tr2v = vminq_f32(vld1q_f32(&drv[2][row - 5][col - 5]), vld1q_f32(&drv[3][row - 5][col - 5]));

                            if(ndir > 4) {
                                float32x4_t tr3v = vminq_f32(vld1q_f32(&drv[4][row - 5][col - 5]), vld1q_f32(&drv[5][row - 5][col - 5]));
                                float32x4_t tr4v = vminq_f32(vld1q_f32(&drv[6][row - 5][col - 5]), vld1q_f32(&drv[7][row - 5][col - 5]));
                                tr1v = vminq_f32(tr1v, tr3v);
                                tr1v = vminq_f32(tr1v, tr4v);
                            }

                            tr1v = vminq_f32(tr1v, tr2v);
                            tr1v = vmulq_f32(tr1v, eightv);

                            for (int d = 0; d < ndir; d++) {
                                //uint8_t tempstore[16];
                                float32x4_t tempv = zerov;

                                for (int v = -1; v <= 1; v++) {
                                    for (int h = -1; h <= 1; h++) {
                                        float32x4_t drv_val = vld1q_f32(&drv[d][row + v - 5][col + h - 5]);
                                        uint32x4_t mask = vcleq_f32(drv_val, tr1v);
                                        tempv = vaddq_f32(tempv, vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(onev))));
                                    }
                                }

                                // Convert float32x4_t to int32x4_t and extract values
                                int32x4_t temp_int = vcvtq_s32_f32(tempv);
                                int32_t temp_arr[4];
                                vst1q_s32(temp_arr, temp_int);
                                
                                homo[d][row][col] = (uint8_t)temp_arr[0];
                                homo[d][row][col + 1] = (uint8_t)temp_arr[1];
                                homo[d][row][col + 2] = (uint8_t)temp_arr[2];
                                homo[d][row][col + 3] = (uint8_t)temp_arr[3];
                            }
                        }
    #endif

                        for (; col < mcol - 6; col++) {
                            float tr = drv[0][row - 5][col - 5] < drv[1][row - 5][col - 5] ? drv[0][row - 5][col - 5] : drv[1][row - 5][col - 5];

                            for (int d = 2; d < ndir; d++) {
                                tr = (drv[d][row - 5][col - 5] < tr ? drv[d][row - 5][col - 5] : tr);
                            }

                            tr *= 8;

                            for (int d = 0; d < ndir; d++) {
                                uint8_t temp = 0;

                                for (int v = -1; v <= 1; v++) {
                                    for (int h = -1; h <= 1; h++) {
                                        temp += (drv[d][row + v - 5][col + h - 5] <= tr ? 1 : 0);
                                    }
                                }

                                homo[d][row][col] = temp;
                            }
                        }
                    }

                    if (height_ - top < ts + 4) {
                        mrow = height_ - top + 2;
                    }

                    if (width_ - left < ts + 4) {
                        mcol = width_ - left + 2;
                    }

                    /* Build 5x5 sum of homogeneity maps */
                    //std::cout << "[DEBUG] Build 5x5 sum of homogeneity maps" << std::endl;
                    const int startcol = std::min(left, 8);

                    for(int d = 0; d < ndir; d++) {
                        for (int row = std::min(top, 8); row < mrow - 8; row++) {
                            int col = startcol;
    #ifdef __ARM_NEON
                            int endcol = row < mrow - 9 ? mcol - 8 : mcol - 23;

                            // crunching 16 values at once is faster than summing up column sums
                            for (; col < endcol; col += 16) {
                                uint8x16_t v5sumv = vdupq_n_u8(0);

                                for(int v = -2; v <= 2; v++) {
                                    for(int h = -2; h <= 2; h++) {
                                        uint8x16_t homov = vld1q_u8(&homo[d][row + v][col + h]);
                                        v5sumv = vqaddq_u8(homov, v5sumv);
                                    }
                                }

                                vst1q_u8(&homosum[d][row][col], v5sumv);
                            }
    #endif

                            if(col < mcol - 8) {
                                int v5sum[5] = {0};

                                for(int v = -2; v <= 2; v++)
                                    for(int h = -2; h <= 2; h++) {
                                        v5sum[2 + h] += homo[d][row + v][col + h];
                                    }

                                int blocksum = v5sum[0] + v5sum[1] + v5sum[2] + v5sum[3] + v5sum[4];
                                homosum[d][row][col] = blocksum;
                                col++;

                                // now we can subtract a column of five from blocksum and get new colsum of 5
                                for (int voffset = 0; col < mcol - 8; col++, voffset++) {
                                    int colsum = homo[d][row - 2][col + 2] + homo[d][row - 1][col + 2] + homo[d][row][col + 2] + homo[d][row + 1][col + 2] + homo[d][row + 2][col + 2];
                                    voffset = voffset == 5 ? 0 : voffset;  // faster than voffset %= 5;
                                    blocksum -= v5sum[voffset];
                                    blocksum += colsum;
                                    v5sum[voffset] = colsum;
                                    homosum[d][row][col] = blocksum;
                                }
                            }
                        }
                    }


                    // calculate maximum of homogeneity maps per pixel. Vectorized calculation is a tiny bit faster than on the fly calculation in next step
                    //std::cout << "[DEBUG] calculate maximum of homogeneity maps per pixel. Vectorized calculation is a tiny bit faster than on the fly calculation in next step" << std::endl;
    #ifdef __ARM_NEON
                    uint8x16_t maskv = vdupq_n_u8(31);
    #endif

                    for (int row = std::min(top, 8); row < mrow - 8; row++) {
                        int col = startcol;
    #ifdef __ARM_NEON
                        int endcol = row < mrow - 9 ? mcol - 8 : mcol - 23;

                        for (; col < endcol; col += 16) {
                            uint8x16_t maxval1 = vmaxq_u8(vld1q_u8(&homosum[0][row][col]), vld1q_u8(&homosum[1][row][col]));
                            uint8x16_t maxval2 = vmaxq_u8(vld1q_u8(&homosum[2][row][col]), vld1q_u8(&homosum[3][row][col]));

                            if(ndir > 4) {
                                uint8x16_t maxval3 = vmaxq_u8(vld1q_u8(&homosum[4][row][col]), vld1q_u8(&homosum[5][row][col]));
                                uint8x16_t maxval4 = vmaxq_u8(vld1q_u8(&homosum[6][row][col]), vld1q_u8(&homosum[7][row][col]));
                                maxval1 = vmaxq_u8(maxval1, maxval3);
                                maxval1 = vmaxq_u8(maxval1, maxval4);
                            }

                            maxval1 = vmaxq_u8(maxval1, maxval2);
                            
                            // NEONには8ビット単位のシフト命令がないため、16ビットに拡張してシフト
                            uint16x8_t maxval1_high = vshll_n_u8(vget_high_u8(maxval1), 0); // 16ビットに拡張
                            uint16x8_t maxval1_low = vshll_n_u8(vget_low_u8(maxval1), 0);   // 16ビットに拡張
                            
                            // 32ビットに拡張して3ビット右シフト
                            uint32x4_t subv_high_high = vshrq_n_u32(vmovl_u16(vget_high_u16(maxval1_high)), 3);
                            uint32x4_t subv_high_low = vshrq_n_u32(vmovl_u16(vget_low_u16(maxval1_high)), 3);
                            uint32x4_t subv_low_high = vshrq_n_u32(vmovl_u16(vget_high_u16(maxval1_low)), 3);
                            uint32x4_t subv_low_low = vshrq_n_u32(vmovl_u16(vget_low_u16(maxval1_low)), 3);
                            
                            // 32ビットから16ビットに戻す
                            uint16x4_t subv_high_high_16 = vmovn_u32(subv_high_high);
                            uint16x4_t subv_high_low_16 = vmovn_u32(subv_high_low);
                            uint16x4_t subv_low_high_16 = vmovn_u32(subv_low_high);
                            uint16x4_t subv_low_low_16 = vmovn_u32(subv_low_low);
                            
                            uint16x8_t subv_high = vcombine_u16(subv_high_low_16, subv_high_high_16);
                            uint16x8_t subv_low = vcombine_u16(subv_low_low_16, subv_low_high_16);
                            
                            // 16ビットから8ビットに戻す
                            uint8x16_t subv = vcombine_u8(vmovn_u16(subv_low), vmovn_u16(subv_high));
                            
                            // 8ビットのマスクを適用（31 = 0x1F）
                            subv = vandq_u8(subv, maskv);
                            
                            // 飽和減算
                            maxval1 = vqsubq_u8(maxval1, subv);
                            
                            vst1q_u8(&homosummax[row][col], maxval1);
                        }
    #endif

                        for (; col < mcol - 8; col ++) {
                            uint8_t maxval = homosum[0][row][col];

                            for(int d = 1; d < ndir; d++) {
                                maxval = maxval < homosum[d][row][col] ? homosum[d][row][col] : maxval;
                            }

                            maxval -= maxval >> 3;
                            homosummax[row][col] = maxval;
                        }
                    }

                    /* Average the most homogeneous pixels for the final result: */
                    //std::cout << "[DEBUG] Average the most homogeneous pixels for the final result:" << std::endl;
                    uint8_t hm[8] = {};

                    for (int row = std::min(top, 8); row < mrow - 8; row++) {
                        for (int col = std::min(left, 8); col < mcol - 8; col++) {

                            for (int d = 0; d < 4; d++) {
                                hm[d] = homosum[d][row][col];
                            }

                            for (int d = 4; d < ndir; d++) {
                                hm[d] = homosum[d][row][col];

                                if (hm[d - 4] < hm[d]) {
                                    hm[d - 4] = 0;
                                } else if (hm[d - 4] > hm[d]) {
                                    hm[d] = 0;
                                }
                            }

                            float avg[4] = {0.f, 0.f, 0.f, 0.f};

                            uint8_t maxval = homosummax[row][col];

                            for (int d = 0; d < ndir; d++)
                                if (hm[d] >= maxval) {
                                    for (int c = 0; c < 3; c++) {
                                        avg[c] += rgb[d][row][col][c];
                                    }
                                    avg[3]++;
                                }

                            RGB_BUFFER(row + top, col + left, 0) = std::max(0.f, avg[0] / avg[3]);
                            RGB_BUFFER(row + top, col + left, 1) = std::max(0.f, avg[1] / avg[3]);
                            RGB_BUFFER(row + top, col + left, 2) = std::max(0.f, avg[2] / avg[3]);
                        }
                    }
                }
            }

            xtrans_border_interpolate(8);
        }
    }

   // 1-pass fast demosaic - RawTherapee's exact fast_xtrans_interpolate
    void run_1pass() {
        // Border interpolation first (RawTherapee does this first)
        xtrans_border_interpolate(1);
        
        // RawTherapee's exact weight matrix
        const float weight[3][3] = {
            {0.25f, 0.5f, 0.25f},
            {0.5f,  0.0f, 0.5f},
            {0.25f, 0.5f, 0.25f}
        };
        
        for (int row = 1; row < (int)height_ - 1; ++row) {
            for (int col = 1; col < (int)width_ - 1; ++col) {
                float sum[3] = {0.0f, 0.0f, 0.0f};
                
                // Calculate weighted sum for each color channel
                for (int v = -1; v <= 1; v++) {
                    for (int h = -1; h <= 1; h++) {
                        int src_row = row + v;
                        int src_col = col + h;
                        int src_idx = src_row * width_ + src_col;
                        int src_color = fcol_xtrans_(src_row, src_col);
                        
                        float raw_val = (float)raw_buffer_.image[src_idx][src_color] / maximum_value_;
                        sum[src_color] += raw_val * weight[v + 1][h + 1];
                    }
                }
                
                int out_idx = row * width_ + col;
                int pixel_color = fcol_xtrans_(row, col);
                float rgb[3];
                
                // RawTherapee's exact color interpolation logic
                switch(pixel_color) {
                case 0: // red pixel
                    {
                        rgb[0] = (float)raw_buffer_.image[out_idx][0] / maximum_value_; // Current red
                        rgb[1] = sum[1] * 0.5f;  // Green interpolation
                        rgb[2] = sum[2];         // Blue interpolation
                    }
                    break;
                    
                case 1: // green pixel
                    {
                        rgb[1] = (float)raw_buffer_.image[out_idx][1] / maximum_value_; // Current green
                        
                        // Check if this is a solitary green pixel
                        int left_color = fcol_xtrans_(row, col - 1);
                        int right_color = fcol_xtrans_(row, col + 1);
                        
                        if (left_color == right_color) {
                            // Solitary green pixel: exactly two direct red and blue neighbors
                            rgb[0] = sum[0];
                            rgb[2] = sum[2];
                        } else {
                            // Non-solitary green: one direct and one diagonal neighbor  
                            rgb[0] = sum[0] * 1.3333333f; // 4/3 coefficient
                            rgb[2] = sum[2] * 1.3333333f; // 4/3 coefficient
                        }
                    }
                    break;
                    
                case 2: // blue pixel
                    {
                        rgb[0] = sum[0];         // Red interpolation
                        rgb[1] = sum[1] * 0.5f;  // Green interpolation
                        rgb[2] = (float)raw_buffer_.image[out_idx][2] / maximum_value_; // Current blue
                    }
                    break;
                }
                
                // Set output values
                rgb_buffer_.image[out_idx][0] = rgb[0]; // Float正規化 (0.0-1.0)
                rgb_buffer_.image[out_idx][1] = rgb[1]; // Float正規化 (0.0-1.0)
                rgb_buffer_.image[out_idx][2] = rgb[2]; // Float正規化 (0.0-1.0)
                // ImageBufferFloat has only 3 channels (RGB), no need for G2
            }
        }
        
        std::cout << "✅ 1-pass XTrans demosaic completed" << std::endl;
    }
};

} // namespace

bool CPUAccelerator::demosaic_xtrans_1pass(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, const char (&xtrans)[6][6], const float (&color_matrix)[3][4], uint16_t maximum_value) {
    if (!raw_buffer.is_valid() || !rgb_buffer.is_valid()) {
        return false;
    }
    XTrans_Processor processor(raw_buffer, rgb_buffer, xtrans, color_matrix, maximum_value);
    processor.run_1pass();
    return true;
}

bool CPUAccelerator::demosaic_xtrans_3pass(const ImageBuffer& raw_buffer, ImageBufferFloat& rgb_buffer, const char (&xtrans)[6][6], const float (&color_matrix)[3][4], uint16_t maximum_value) {
    if (!raw_buffer.is_valid() || !rgb_buffer.is_valid()) {
        return false;
    }
    XTrans_Processor processor(raw_buffer, rgb_buffer, xtrans, color_matrix, maximum_value);
    processor.run_3pass();
    return true;
}

//===================================================================
// カラースペース変換
//===================================================================

bool CPUAccelerator::convert_color_space(const ImageBufferFloat& rgb_input, ImageBufferFloat& rgb_output, const float transform[3][4]) {

    if (!initialized_) return false;

    // 引数チェック
    if (!rgb_input.is_valid() || !rgb_output.is_valid() || rgb_input.width != rgb_output.width || rgb_input.height != rgb_output.height || rgb_input.channels != 3 || rgb_output.channels != 3) {
        std::cerr << "❌ Invalid or mismatched buffers for color space conversion" << std::endl;
        return false;
    }

    const size_t pixel_count = rgb_input.width * rgb_input.height;
    for (size_t i = 0; i < pixel_count; i++) {
        const float* in = rgb_input.image[i];
        float* out = rgb_output.image[i];
        out[0] = fmaxf(0.0f, fminf(1.0f, transform[0][0] * in[0] + transform[0][1] * in[1] + transform[0][2] * in[2] + transform[0][3]));
        out[1] = fmaxf(0.0f, fminf(1.0f, transform[1][0] * in[0] + transform[1][1] * in[1] + transform[1][2] * in[2] + transform[1][3]));
        out[2] = fmaxf(0.0f, fminf(1.0f, transform[2][0] * in[0] + transform[2][1] * in[1] + transform[2][2] * in[2] + transform[2][3]));
    }
    return true;
}

//===================================================================
// ガンマ補正
//===================================================================

bool CPUAccelerator::gamma_correct(const ImageBufferFloat& rgb_input, ImageBufferFloat& rgb_output, float gamma_power, float gamma_slope, int output_color_space) {

    if (!initialized_) return false;

    // 引数チェック
    if (!rgb_input.is_valid() || !rgb_output.is_valid() || rgb_input.width != rgb_output.width || rgb_input.height != rgb_output.height || rgb_input.channels != 3 || rgb_output.channels != 3) {
        std::cerr << "❌ Invalid or mismatched buffers for gamma correction" << std::endl;
        return false;
    }

    // バッファが違ったらコピー
    if (rgb_input.image != rgb_output.image) {
        std::memcpy(rgb_output.image, rgb_input.image, rgb_input.width * rgb_input.height * 3 * sizeof(float));
    }

    std::cout << "🎯 Gamma Correction: color_space=" << output_color_space << ", γ=" << gamma_power << ", slope=" << gamma_slope << std::endl;
    for (size_t i = 0; i < rgb_input.width * rgb_input.height; i++) {
        float* pixel = rgb_output.image[i];
        for (int c = 0; c < 3; c++) {
            float linear_value = pixel[c], gamma_corrected;
            switch (output_color_space) {
            case ColorSpace::sRGB:
            case ColorSpace::P3D65:
                gamma_corrected = apply_srgb_gamma_encode(linear_value);
                break;
            case ColorSpace::AdobeRGB:
            case ColorSpace::WideGamutRGB:
                gamma_corrected = apply_pure_power_gamma_encode(linear_value, 2.222f);
                break;
            case ColorSpace::ProPhotoRGB:
               gamma_corrected = apply_pure_power_gamma_encode(linear_value, 1.8f);
               break;
            //case ColorSpace::ACES:
            //    gamma_corrected = apply_aces_gamma_encode(linear_value);
            //    break;
            case ColorSpace::Rec2020:
                gamma_corrected = apply_rec2020_gamma_encode(linear_value);
                break;
            default:
                gamma_corrected = apply_pure_power_gamma_encode_with_slope(linear_value, gamma_power, gamma_slope);
                break;
            }
            pixel[c] = gamma_corrected;
        }
    }
    return true;
}

double CPUAccelerator::get_last_processing_time() const { return last_processing_time_; }
size_t CPUAccelerator::get_memory_usage() const { return 0; }
std::string CPUAccelerator::get_device_info() const { return device_name_; }

//===================================================================
// ガンマ補正関数
//===================================================================

float CPUAccelerator::apply_srgb_gamma_encode(float v) const {
    return (v <= 0.0031308f) ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

float CPUAccelerator::apply_aces_gamma_encode(float v) const {
    if (v <= 0.0f) return 0.0f;
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return std::max(0.0f, std::min(1.0f, (v * (a * v + b)) / (v * (c * v + d) + e)));
}

float CPUAccelerator::apply_rec2020_gamma_encode(float v) const {
    const float a = 1.09929682680944f, b = 0.018053968510807f;
    return (v < b) ? 4.5f * v : a * std::pow(v, 0.45f) - (a - 1.0f);
}

float CPUAccelerator::apply_pure_power_gamma_encode(float v, float p) const {
    return std::pow(v, 1.0f / p);
}

float CPUAccelerator::apply_pure_power_gamma_encode_with_slope(float v, float p, float s) const {
    if (s <= 0.f) return std::pow(v, 1.0f / p);
    return (v < 1.0f / s) ? v * s : std::pow(v, 1.0f / p);
}

//===================================================================
// 境界補間
//===================================================================

void CPUAccelerator::border_interpolate(const ImageBufferFloat& rgb_buffer, uint32_t filters, int border_int) {
    const size_t width = rgb_buffer.width, height = rgb_buffer.height;
    const int border = border_int;
    float (*image)[3] = rgb_buffer.image;
    std::cout << "🔧 LibRaw-exact border interpolation: border=" << border << " pixels" << std::endl;
    
    // ✨LibRaw完全移植✨: 元のLibRaw border_interpolate を正確に再現
    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col++) {
            // LibRawの最適化: 境界領域のみ処理、内部は早期スキップ
            if (col == static_cast<size_t>(border) && row >= static_cast<size_t>(border) && 
                row < (height - static_cast<size_t>(border))) {
                col = width - static_cast<size_t>(border);
            }
            
            // sum配列: [R, G, B, G2, R_count, G_count, B_count, G2_count]
            float sum[8] = {0};
            
            // 3x3近傍を走査（LibRaw方式）
            for (size_t y = (row > 0) ? row - 1 : 0; y < row + 2 && y < height; y++) {
                for (size_t x = (col > 0) ? col - 1 : 0; x < col + 2 && x < width; x++) {
                    // その位置のBayerパターン色を取得
                    int f = fcol_bayer(y, x, filters);
                    
                    // その色のピクセル値を累積
                    sum[f] += image[y * width + x][f];
                    sum[f + 4]++;  // カウント
                }
            }
            
            // 現在位置のネイティブ色
            int f = fcol_bayer(row, col, filters);
            
            // 非ネイティブ色のみを補間（LibRaw方式）
            for (int c = 0; c < 3; c++) {
                if (c != f && sum[c + 4] > 0) {
                    image[row * width + col][c] = sum[c] / sum[c + 4];
                }
            }
        }
    }
    std::cout << "✅ LibRaw-exact border interpolation completed" << std::endl;
}

//===================================================================
// トーンマッピング
//===================================================================

bool CPUAccelerator::tone_mapping(const ImageBufferFloat& rgb_input,
                               ImageBufferFloat& rgb_output,
                               float after_scale) {

        auto acesToneMap = [](float x) {
            static constexpr float a = 2.51f;
            static constexpr float b = 0.03f;
            static constexpr float c = 2.43f;
            static constexpr float d = 0.59f;
            static constexpr float e = 0.14f;
            
            return (x * (a * x + b)) / (x * (c * x + d) + e);
        };

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t idx = 0; idx < rgb_output.width * rgb_output.height; ++idx) {
            float* in = rgb_input.image[idx];
            float* out = rgb_output.image[idx];

            out[0] = acesToneMap(in[0]) * after_scale;
            out[1] = acesToneMap(in[1]) * after_scale;
            out[2] = acesToneMap(in[2]) * after_scale;
        }
        return true;
   }

} // namespace libraw_enhanced
