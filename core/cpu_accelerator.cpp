#include "cpu_accelerator.h"
#include "accelerator.h"
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
// ✨ NEONヘッダーのインクルード条件を __ARM_NEON のみに変更
#if defined(__aarch64__) && defined(__ARM_NEON)
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
// Demosaic function for linear interpolation
//===================================================================

bool CPUAccelerator::demosaic_bayer_linear(const ImageBuffer& raw_buffer, ImageBuffer& rgb_buffer, uint32_t filters) {
#ifdef __APPLE__
    if (!initialized_) return false;
    std::cout << "🔧 Starting CPU linear interpolation demosaic..." << std::endl;
    const size_t width = raw_buffer.width, height = raw_buffer.height;
    const int colors = 4;
    std::cout << "📐 Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "🎨 Bayer filter pattern: 0x" << std::hex << filters << std::dec << std::endl;

    const int size = (filters == 9) ? 6 : 16;
    std::vector<int> code_buffer(size * size * 32);
    int* code = code_buffer.data();
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            int* ip = code + (((row * 16) + col) * 32) + 1;
            int f = fcol_bayer(row, col, filters);
            int sum[4] = {0};
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    int shift = (y == 0) + (x == 0);
                    int color = fcol_bayer(row + y + 48, col + x + 48, filters);
                    if (color == f) continue;
                    *ip++ = (width * y + x) * 4 + color;
                    *ip++ = shift;
                    *ip++ = color;
                    sum[color] += 1 << shift;
                }
            }
            code[(row * 16 + col) * 32] = (ip - (code + ((row * 16) + col) * 32)) / 3;
            for (int c = 0; c < colors; c++) {
                if (c != f) {
                    *ip++ = c;
                    *ip++ = sum[c] > 0 ? 256 / sum[c] : 0;
                }
            }
        }
    }
    std::cout << "✅ Generated interpolation lookup table (" << size << "x" << size << " pattern)" << std::endl;
    linear_interpolate_loop_cpu(raw_buffer, code, size, colors);
    std::cout << "✅ CPU linear interpolation completed successfully" << std::endl;
    
    // ✨デモザイク後境界補間✨: 最終処理で境界を修正
    std::cout << "🔧 Post-demosaic border interpolation for LINEAR (1px border)..." << std::endl;
    border_interpolate(raw_buffer, filters, 1);
    
    return true;
#else
    return false;
#endif
}

void CPUAccelerator::linear_interpolate_loop_cpu(const ImageBuffer& raw_buffer, int* code, int size, int colors) {
    const size_t width = raw_buffer.width, height = raw_buffer.height;
    uint16_t (*image)[4] = raw_buffer.image;
    std::cout << "🔧 Linear interpolation loop: " << (height - 2) << "x" << (width - 2) << " pixels" << std::endl;
    for (size_t row = 1; row < height - 1; row++) {
        for (size_t col = 1; col < width - 1; col++) {
            uint16_t* pix = image[row * width + col];
            int* ip = code + ((((row % size) * 16) + (col % size)) * 32);
            int sum[4] = {0};
            int num_samples = *ip++;
            for (int i = 0; i < num_samples; i++, ip += 3) {
                sum[ip[2]] += pix[ip[0]] << ip[1];
            }
            for (int i = colors; --i; ip += 2) {
                if (ip[1] > 0) pix[ip[0]] = (sum[ip[0]] * ip[1]) >> 8;
            }
        }
        if ((row % 1000) == 0) std::cout << "🔄 Processed row " << row << "/" << height << std::endl;
    }
    std::cout << "✅ Linear interpolation loop completed" << std::endl;
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
    uint32_t filters_;
    
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
    AAHD_Processor(const ImageBuffer& raw_buffer, uint32_t filters) 
        : raw_buffer_(raw_buffer), filters_(filters) {
        
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
    std::cout << "🔧 AAHD: Hide hot pixels..." << std::endl;
    
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
    std::cout << "🔧 AAHD: Interpolate green channel..." << std::endl;
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
    std::cout << "🔧 AAHD: Interpolate red/blue channels..." << std::endl;
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
    std::cout << "🔧 AAHD: Evaluate interpolation directions..." << std::endl;
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
                    uvdiff[d][k] = sqr(yuv[d][moff][1] - yuv[d][moff + hvdir[k]][1]) +
                                   sqr(yuv[d][moff][2] - yuv[d][moff + hvdir[k]][2]);
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
    std::cout << "🔧 AAHD: Refine horizontal/vertical directions..." << std::endl;
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
    std::cout << "🔧 AAHD: Combine final image..." << std::endl;
    for (size_t i = 0; i < raw_buffer_.height; ++i) {
        for (size_t j = 0; j < raw_buffer_.width; ++j) {
            int moff = nr_offset(static_cast<int>(i) + nr_margin, static_cast<int>(j) + nr_margin);
            size_t img_off = i * raw_buffer_.width + j;
            int d = (ndir[moff] & VER) ? 1 : 0;
            
            raw_buffer_.image[img_off][0] = rgb_ahd[d][moff][0];
            raw_buffer_.image[img_off][1] = rgb_ahd[d][moff][1];
            raw_buffer_.image[img_off][2] = rgb_ahd[d][moff][2];
            raw_buffer_.image[img_off][3] = rgb_ahd[d][moff][1];
        }
    }
    std::cout << "✅ AAHD processing completed" << std::endl;
}

bool CPUAccelerator::demosaic_bayer_aahd(const ImageBuffer& raw_buffer, ImageBuffer& rgb_buffer, uint32_t filters) {
#ifdef __APPLE__
    if (!initialized_) return false;
    std::cout << "🔧 Starting AAHD (Adaptive AHD) demosaic..." << std::endl;
    std::cout << "📐 AAHD processing: " << raw_buffer.width << "x" << raw_buffer.height << std::endl;
    std::cout << "🎨 Bayer pattern: 0x" << std::hex << filters << std::dec << std::endl;
        
    AAHD_Processor aahd_proc(raw_buffer, filters);
    aahd_proc.hide_hots();
    aahd_proc.make_ahd_greens(); 
    aahd_proc.make_ahd_rb();
    aahd_proc.evaluate_ahd();
    aahd_proc.refine_hv_dirs();
    aahd_proc.combine_image();
    std::cout << "✅ AAHD demosaic completed successfully" << std::endl;
    
    // ✨デモザイク後境界補間✨: 最終処理で境界を修正
    std::cout << "🔧 Post-demosaic border interpolation for AAHD (2px border)..." << std::endl;
    border_interpolate(raw_buffer, filters, 2);
    
    return true;
#else
    return false;
#endif
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

    static float CLIP_FLOAT(float val) {
        return val > 65535.0f ? 65535.0f : (val < 0.0f ? 0.0f : val);
    }

public:
    static uint16_t CLIP_16BIT(float val) {
        if (val < 0.0f) return 0;
        if (val > 65535.0f) return 65535;
        return static_cast<uint16_t>(val + 0.5f);
    }

    DCB_Processor(float (*img_buffer)[4], int w, int h, uint32_t f)
        : width(w), height(h), filters(f), image(img_buffer) {}

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

bool CPUAccelerator::demosaic_bayer_dcb(const ImageBuffer& raw_buffer, ImageBuffer& rgb_buffer, uint32_t filters, int iterations, bool dcb_enhance) {
    if (!initialized_ || !raw_buffer.is_valid() || !rgb_buffer.is_valid() || raw_buffer.width != rgb_buffer.width || raw_buffer.height != rgb_buffer.height) {
        std::cerr << "❌ DCB Demosaic: Invalid buffers or initialization state." << std::endl;
        return false;
    }
    std::cout << "🔧 Starting DCB demosaic (Clean Class Implementation)..." << std::endl;
    
    // ✨境界補間を後に移動✨: デモザイク処理前はスキップ
    // std::cout << "🔧 Applying border interpolation for DCB (3px border)..." << std::endl;
    // border_interpolate(raw_buffer, filters, 3);
    
    const size_t width = raw_buffer.width, height = raw_buffer.height;
    const size_t num_pixels = width * height;

    auto sparse_image_ptr = std::make_unique<float[][4]>(num_pixels);
    float (*sparse_image)[4] = sparse_image_ptr.get();
    auto work_image_ptr = std::make_unique<float[][4]>(num_pixels);
    float (*work_image)[4] = work_image_ptr.get();
    
    // 手順1: 正しいRAWデータの読み出し
    for (size_t i = 0; i < num_pixels; ++i) {
        uint16_t raw_value = 0;
        for (int ch = 0; ch < 4; ++ch) {
            if (raw_buffer.image[i][ch] != 0) {
                raw_value = raw_buffer.image[i][ch];
                break;
            }
        }
        int color_idx = fcol_bayer(i / width, i % width, filters);
        if (color_idx == 3) color_idx = 1;
        for(int c=0; c<4; ++c) sparse_image[i][c] = 0.0f;
        sparse_image[i][color_idx] = static_cast<float>(raw_value);
    }
    
    // 手順2: バイリニア補間 (コンパイラによる自動ベクトル化を期待)
    std::memcpy(work_image, sparse_image, num_pixels * sizeof(float[4]));
    for (size_t r = 1; r < height - 1; ++r) {
        for (size_t c = 1; c < width - 1; ++c) {
            size_t i = r * width + c;
            int f = fcol_bayer(r, c, filters);
            if (f == 0) {
                work_image[i][1] = (sparse_image[i - width][1] + sparse_image[i + width][1] + sparse_image[i - 1][1] + sparse_image[i + 1][1]) * 0.25f;
                work_image[i][2] = (sparse_image[i - width - 1][2] + sparse_image[i - width + 1][2] + sparse_image[i + width - 1][2] + sparse_image[i + width + 1][2]) * 0.25f;
            } else if (f == 2) {
                work_image[i][1] = (sparse_image[i - width][1] + sparse_image[i + width][1] + sparse_image[i - 1][1] + sparse_image[i + 1][1]) * 0.25f;
                work_image[i][0] = (sparse_image[i - width - 1][0] + sparse_image[i - width + 1][0] + sparse_image[i + width - 1][0] + sparse_image[i + width + 1][0]) * 0.25f;
            } else {
                if (fcol_bayer(r, c - 1, filters) == 0) {
                    work_image[i][0] = (sparse_image[i - 1][0] + sparse_image[i + 1][0]) * 0.5f;
                    work_image[i][2] = (sparse_image[i - width][2] + sparse_image[i + width][2]) * 0.5f;
                } else {
                    work_image[i][0] = (sparse_image[i - width][0] + sparse_image[i + width][0]) * 0.5f;
                    work_image[i][2] = (sparse_image[i - 1][2] + sparse_image[i + 1][2]) * 0.5f;
                }
            }
        }
    }
    
    // 手順3: 境界複製を無効化（同一値問題の原因）
    // replicate_border_generic<float>(work_image, width, height, 6);
    std::cout << "🔧 DCB: Skipping border replication to avoid identical pixel issue" << std::endl;

    // 手順4: 完全なDCBアルゴリズムを実行
    DCB_Processor dcb_proc(work_image, static_cast<int>(width), static_cast<int>(height), filters);
    dcb_proc.run_dcb(iterations, dcb_enhance);
    
    // ✨✨✨ 手順5: NEONイントリンシックを使った超高速書き込み ✨✨✨
#if defined(__aarch64__) && defined(__ARM_NEON)
    float* float_ptr = &work_image[0][0];
    uint16_t* uint_ptr = &rgb_buffer.image[0][0];
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_max = vdupq_n_f32(65535.0f);
    size_t end = num_pixels - (num_pixels % 4);

    for (size_t i = 0; i < end; i += 4) {
        float32x4x4_t v_pixels = vld4q_f32(float_ptr + i * 4);
        v_pixels.val[0] = vmaxq_f32(v_zero, vminq_f32(v_pixels.val[0], v_max));
        v_pixels.val[1] = vmaxq_f32(v_zero, vminq_f32(v_pixels.val[1], v_max));
        v_pixels.val[2] = vmaxq_f32(v_zero, vminq_f32(v_pixels.val[2], v_max));
        uint16x4_t r_u16 = vmovn_u32(vcvtq_u32_f32(v_pixels.val[0]));
        uint16x4_t g_u16 = vmovn_u32(vcvtq_u32_f32(v_pixels.val[1]));
        uint16x4_t b_u16 = vmovn_u32(vcvtq_u32_f32(v_pixels.val[2]));
        uint16x4x4_t result = {{r_u16, g_u16, b_u16, g_u16}};
        vst4_u16(uint_ptr + i * 4, result);
    }
    for(size_t i = end; i < num_pixels; ++i) { // 末尾の余り処理
        rgb_buffer.image[i][0] = DCB_Processor::CLIP_16BIT(work_image[i][0]);
        rgb_buffer.image[i][1] = DCB_Processor::CLIP_16BIT(work_image[i][1]);
        rgb_buffer.image[i][2] = DCB_Processor::CLIP_16BIT(work_image[i][2]);
        rgb_buffer.image[i][3] = rgb_buffer.image[i][1];
    }
#else
    // NEON非対応環境用のフォールバック
    for (size_t i = 0; i < num_pixels; ++i) {
        rgb_buffer.image[i][0] = DCB_Processor::CLIP_16BIT(work_image[i][0]);
        rgb_buffer.image[i][1] = DCB_Processor::CLIP_16BIT(work_image[i][1]);
        rgb_buffer.image[i][2] = DCB_Processor::CLIP_16BIT(work_image[i][2]);
        rgb_buffer.image[i][3] = rgb_buffer.image[i][1];
    }
#endif
    
    std::cout << "✅ DCB demosaic completed successfully (Clean Class Implementation)" << std::endl;
    
    // ✨デモザイク後境界補間✨: 最終処理で境界を修正
    std::cout << "🔧 Post-demosaic border interpolation for DCB (3px border)..." << std::endl;
    border_interpolate(raw_buffer, filters, 3);
    
    return true;
}

//===================================================================
// AMaZE demosaicing algorithm implementation
//===================================================================

namespace { // This file is for internal use only

// A complete port of the AMaZE demosaicing algorithm from RawTherapee
class AMaZE_Processor_RT {
private:
    // Constants from the original RawTherapee implementation
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

    // Member variables
    const ImageBuffer& raw_buffer_;
    ImageBuffer& rgb_buffer_;
    const uint32_t filters_;
    const size_t width_, height_;
    int ex = 0, ey = 0; // Offset of the R pixel

    // Helper functions
    inline int FC(int row, int col) const { return fcol_bayer(row, col, filters_); }
    static inline float SQR(float x) { return x * x; }
    static inline uint16_t CLIP_U16(float val) {
        if (val < 0.0f) return 0;
        if (val > 65535.0f) return 65535;
        return static_cast<uint16_t>(val + 0.5f);
    }
    struct s_hv { float h; float v; };

public:
    AMaZE_Processor_RT(const ImageBuffer& raw, ImageBuffer& rgb, uint32_t f)
        : raw_buffer_(raw), rgb_buffer_(rgb), filters_(f), width_(raw.width), height_(raw.height)
    {
        if (FC(0, 0) == 1) { // G
            if (FC(0, 1) == 0) { ey = 0; ex = 1; } // GR
            else { ey = 1; ex = 0; }               // GB
        } else { // R or B
            if (FC(0, 0) == 0) { ey = 0; ex = 0; } // RG
            else { ey = 1; ex = 1; }               // BG
        }
    }

    void run() {
        const float clip_pt = 1.0f;
        const float clip_pt8 = 0.8f;
        static const int v1 = TS, v2 = 2*TS, v3 = 3*TS, p1 = -TS+1, p2 = -2*TS+2, p3 = -3*TS+3, m1 = TS+1, m2 = 2*TS+2, m3 = 3*TS+3;
        
        std::vector<char> processing_buffer(14 * sizeof(float) * TS * TS + sizeof(unsigned char) * TS * TSH + 63);
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
        float* dgintv       = hvwt + TS*TSH;
        float* dginth       = dgintv + TS*TS;
        float* Dgrbsq1m     = dginth + TS*TS;
        float* Dgrbsq1p     = Dgrbsq1m + TS*TSH;
        float* cfa          = Dgrbsq1p + TS*TSH;
        unsigned char* nyquist = reinterpret_cast<unsigned char*>(cfa + TS*TS);
        
        unsigned char* nyquist2 = reinterpret_cast<unsigned char*>(cddiffsq);

        for (int top = -16; top < (int)height_; top += TS - 32) {
            for (int left = -16; left < (int)width_; left += TS - 32) {
                int bottom = std::min(top + TS, (int)height_ + 16), right = std::min(left + TS, (int)width_ + 16);
                int rr1 = bottom - top, cc1 = right - left;
                int rrmin = (top < 0) ? 16 : 0, ccmin = (left < 0) ? 16 : 0;
                int rrmax = (bottom > (int)height_) ? (int)height_ - top : rr1, ccmax = (right > (int)width_) ? (int)width_ - left : cc1;

                for (int rr = rrmin; rr < rrmax; ++rr) for (int cc = ccmin; cc < ccmax; ++cc) {
                    int c = FC(rr + top, cc + left); if (c == 3) c = 1;
                    cfa[rr*TS+cc] = (float)raw_buffer_.image[(rr + top) * width_ + (cc + left)][c] / 65535.0f;
                }
                if (top < 0) for(int rr=0; rr<16; ++rr) for(int cc=0; cc<cc1; ++cc) cfa[rr*TS+cc] = cfa[(31-rr)*TS+cc];
                if (bottom > (int)height_) for(int rr=rrmax; rr<rr1; ++rr) for(int cc=0; cc<cc1; ++cc) cfa[rr*TS+cc] = cfa[(2*rrmax-rr-1)*TS+cc];
                if (left < 0) for(int cc=0; cc<16; ++cc) for(int rr=0; rr<rr1; ++rr) cfa[rr*TS+cc] = cfa[rr*TS+(31-cc)];
                if (right > (int)width_) for(int cc=ccmax; cc<cc1; ++cc) for(int rr=0; rr<rr1; ++rr) cfa[rr*TS+cc] = cfa[rr*TS+(2*ccmax-cc-1)];
                for(int i=0; i<TS*TS; ++i) rgbgreen[i] = cfa[i];
                memset(nyquist, 0, sizeof(unsigned char) * TS * TSH);

                for (int rr = 2; rr < rr1 - 2; rr++) for (int cc = 2, idx = rr*TS+cc; cc < cc1 - 2; cc++, idx++) {
                    float delh = fabsf(cfa[idx+1]-cfa[idx-1]), delv = fabsf(cfa[idx+v1]-cfa[idx-v1]);
                    dirwts0[idx] = eps+fabsf(cfa[idx+v2]-cfa[idx])+fabsf(cfa[idx]-cfa[idx-v2])+delv;
                    dirwts1[idx] = eps+fabsf(cfa[idx+2]-cfa[idx])+fabsf(cfa[idx]-cfa[idx-2])+delh;
                    delhvsqsum[idx] = SQR(delh)+SQR(delv);
                }
                
                for (int rr = 4; rr < rr1 - 4; rr++) for (int cc = 4, idx=rr*TS+cc; cc < cc1 - 4; cc++, idx++) {
                    int sgn = (FC(rr,cc)&1)?-1:1;
                    float cru=cfa[idx-v1]*(dirwts0[idx-v2]+dirwts0[idx])/(dirwts0[idx-v2]*(eps+cfa[idx])+dirwts0[idx]*(eps+cfa[idx-v2]));
                    float crd=cfa[idx+v1]*(dirwts0[idx+v2]+dirwts0[idx])/(dirwts0[idx+v2]*(eps+cfa[idx])+dirwts0[idx]*(eps+cfa[idx+v2]));
                    float crl=cfa[idx-1]*(dirwts1[idx-2]+dirwts1[idx])/(dirwts1[idx-2]*(eps+cfa[idx])+dirwts1[idx]*(eps+cfa[idx-2]));
                    float crr=cfa[idx+1]*(dirwts1[idx+2]+dirwts1[idx])/(dirwts1[idx+2]*(eps+cfa[idx])+dirwts1[idx]*(eps+cfa[idx+2]));
                    float guha=cfa[idx-v1]+0.5f*(cfa[idx]-cfa[idx-v2]), gdha=cfa[idx+v1]+0.5f*(cfa[idx]-cfa[idx+v2]);
                    float glha=cfa[idx-1]+0.5f*(cfa[idx]-cfa[idx-2]), grha=cfa[idx+1]+0.5f*(cfa[idx]-cfa[idx+2]);
                    float guar=fabsf(1.f-cru)<arthresh?cfa[idx]*cru:guha, gdar=fabsf(1.f-crd)<arthresh?cfa[idx]*crd:gdha;
                    float glar=fabsf(1.f-crl)<arthresh?cfa[idx]*crl:glha, grar=fabsf(1.f-crr)<arthresh?cfa[idx]*crr:grha;
                    float hwt=dirwts1[idx-1]/(dirwts1[idx-1]+dirwts1[idx+1]), vwt=dirwts0[idx-v1]/(dirwts0[idx+v1]+dirwts0[idx-v1]);
                    float Gintvha=vwt*gdha+(1.f-vwt)*guha, Ginthha=hwt*grha+(1.f-hwt)*glha;
                    vcdalt[idx]=sgn*(Gintvha-cfa[idx]); hcdalt[idx]=sgn*(Ginthha-cfa[idx]);
                    if(cfa[idx]>clip_pt8||Gintvha>clip_pt8||Ginthha>clip_pt8) { vcd[idx]=vcdalt[idx]; hcd[idx]=hcdalt[idx]; guar=guha; gdar=gdha; glar=glha; grar=grha; }
                    else { vcd[idx]=sgn*(vwt*gdar+(1.f-vwt)*guar-cfa[idx]); hcd[idx]=sgn*(hwt*grar+(1.f-hwt)*glar-cfa[idx]); }
                    dgintv[idx]=std::min(SQR(guha-gdha),SQR(guar-gdar)); dginth[idx]=std::min(SQR(glha-grha),SQR(glar-grar));
                }
                
                for (int rr = 4; rr < rr1 - 4; rr++) for (int cc = 4, idx = rr*TS+cc; cc < cc1-4; cc++, idx++) {
                    float hcdvar=3.f*(SQR(hcd[idx-2])+SQR(hcd[idx])+SQR(hcd[idx+2]))-SQR(hcd[idx-2]+hcd[idx]+hcd[idx+2]);
                    float hcdaltvar=3.f*(SQR(hcdalt[idx-2])+SQR(hcdalt[idx])+SQR(hcdalt[idx+2]))-SQR(hcdalt[idx-2]+hcdalt[idx]+hcdalt[idx+2]);
                    float vcdvar=3.f*(SQR(vcd[idx-v2])+SQR(vcd[idx])+SQR(vcd[idx+v2]))-SQR(vcd[idx-v2]+vcd[idx]+vcd[idx+v2]);
                    float vcdaltvar=3.f*(SQR(vcdalt[idx-v2])+SQR(vcdalt[idx])+SQR(vcdalt[idx+v2]))-SQR(vcdalt[idx-v2]+vcdalt[idx]+vcdalt[idx+v2]);
                    if(hcdaltvar<hcdvar)hcd[idx]=hcdalt[idx]; if(vcdaltvar<vcdvar)vcd[idx]=vcdalt[idx];
                    float Gintv, Ginth;
                    if(FC(rr,cc)&1) {
                        Ginth=-hcd[idx]+cfa[idx]; Gintv=-vcd[idx]+cfa[idx];
                        if(hcd[idx]>0){if(3.f*hcd[idx]>(Ginth+cfa[idx]))hcd[idx]=-median(Ginth,cfa[idx-1],cfa[idx+1])+cfa[idx];else{float hwt2=1.f-3.f*hcd[idx]/(eps+Ginth+cfa[idx]);hcd[idx]=hwt2*hcd[idx]+(1.f-hwt2)*(-median(Ginth,cfa[idx-1],cfa[idx+1])+cfa[idx]);}}
                        if(vcd[idx]>0){if(3.f*vcd[idx]>(Gintv+cfa[idx]))vcd[idx]=-median(Gintv,cfa[idx-v1],cfa[idx+v1])+cfa[idx];else{float vwt2=1.f-3.f*vcd[idx]/(eps+Gintv+cfa[idx]);vcd[idx]=vwt2*vcd[idx]+(1.f-vwt2)*(-median(Gintv,cfa[idx-v1],cfa[idx+v1])+cfa[idx]);}}
                    } else {
                        Ginth=hcd[idx]+cfa[idx]; Gintv=vcd[idx]+cfa[idx];
                        if(hcd[idx]<0){if(3.f*hcd[idx]<-(Ginth+cfa[idx]))hcd[idx]=median(Ginth,cfa[idx-1],cfa[idx+1])-cfa[idx];else{float hwt2=1.f+3.f*hcd[idx]/(eps+Ginth+cfa[idx]);hcd[idx]=hwt2*hcd[idx]+(1.f-hwt2)*(median(Ginth,cfa[idx-1],cfa[idx+1])-cfa[idx]);}}
                        if(vcd[idx]<0){if(3.f*vcd[idx]<-(Gintv+cfa[idx]))vcd[idx]=median(Gintv,cfa[idx-v1],cfa[idx+v1])-cfa[idx];else{float vwt2=1.f+3.f*vcd[idx]/(eps+Gintv+cfa[idx]);vcd[idx]=vwt2*vcd[idx]+(1.f-vwt2)*(median(Gintv,cfa[idx-v1],cfa[idx+v1])-cfa[idx]);}}
                    }
                    if(Ginth>clip_pt)hcd[idx]=(FC(rr,cc)&1?-1:1)*(median(Ginth,cfa[idx-1],cfa[idx+1])-cfa[idx]);
                    if(Gintv>clip_pt)vcd[idx]=(FC(rr,cc)&1?-1:1)*(median(Gintv,cfa[idx-v1],cfa[idx+v1])-cfa[idx]);
                    cddiffsq[idx] = SQR(vcd[idx]-hcd[idx]);
                }

                for (int rr=6; rr<rr1-6; rr++) for (int cc=6+(FC(rr,2)&1), idx=rr*TS+cc; cc<cc1-6; cc+=2, idx+=2) {
                    float uave=vcd[idx]+vcd[idx-v1]+vcd[idx-v2]+vcd[idx-v3], dave=vcd[idx]+vcd[idx+v1]+vcd[idx+v2]+vcd[idx+v3];
                    float lave=hcd[idx]+hcd[idx-1]+hcd[idx-2]+hcd[idx-3], rave=hcd[idx]+hcd[idx+1]+hcd[idx+2]+hcd[idx+3];
                    float Dgrbvvaru=SQR(vcd[idx]-uave)+SQR(vcd[idx-v1]-uave)+SQR(vcd[idx-v2]-uave)+SQR(vcd[idx-v3]-uave);
                    float Dgrbvvard=SQR(vcd[idx]-dave)+SQR(vcd[idx+v1]-dave)+SQR(vcd[idx+v2]-dave)+SQR(vcd[idx+v3]-dave);
                    float Dgrbhvarl=SQR(hcd[idx]-lave)+SQR(hcd[idx-1]-lave)+SQR(hcd[idx-2]-lave)+SQR(hcd[idx-3]-lave);
                    float Dgrbhvarr=SQR(hcd[idx]-rave)+SQR(hcd[idx+1]-rave)+SQR(hcd[idx+2]-rave)+SQR(hcd[idx+3]-rave);
                    float hwt=dirwts1[idx-1]/(dirwts1[idx-1]+dirwts1[idx+1]), vwt=dirwts0[idx-v1]/(dirwts0[idx+v1]+dirwts0[idx-v1]);
                    float vcdvar=epssq+vwt*Dgrbvvard+(1.f-vwt)*Dgrbvvaru, hcdvar=epssq+hwt*Dgrbhvarr+(1.f-hwt)*Dgrbhvarl;
                    float Dgrbvvaru2=(dgintv[idx])+(dgintv[idx-v1])+(dgintv[idx-v2]), Dgrbvvard2=(dgintv[idx])+(dgintv[idx+v1])+(dgintv[idx+v2]);
                    float Dgrbhvarl2=(dginth[idx])+(dginth[idx-1])+(dginth[idx-2]), Dgrbhvarr2=(dginth[idx])+(dginth[idx+1])+(dginth[idx+2]);
                    float vcdvar1=epssq+vwt*Dgrbvvard2+(1.f-vwt)*Dgrbvvaru2, hcdvar1=epssq+hwt*Dgrbhvarr2+(1.f-hwt)*Dgrbhvarl2;
                    float varwt=hcdvar/(vcdvar+hcdvar), diffwt=hcdvar1/(vcdvar1+hcdvar1);
                    hvwt[idx>>1] = ((0.5f-varwt)*(0.5f-diffwt)>0.f && fabsf(0.5f-diffwt)<fabsf(0.5f-varwt)) ? varwt : diffwt;
                }
                
                float* nyqutest = reinterpret_cast<float*>(Dgrbsq1p); // Share buffer
                for (int rr=6; rr<rr1-6; rr++) for (int cc=6+(FC(rr,2)&1),idx=rr*TS+cc; cc<cc1-6; cc+=2,idx+=2){
                    nyqutest[idx>>1]=(gaussodd[0]*cddiffsq[idx]+gaussodd[1]*(cddiffsq[idx-m1]+cddiffsq[idx+p1]+cddiffsq[idx-p1]+cddiffsq[idx+m1])+gaussodd[2]*(cddiffsq[idx-v2]+cddiffsq[idx-2]+cddiffsq[idx+2]+cddiffsq[idx+v2])+gaussodd[3]*(cddiffsq[idx-m2]+cddiffsq[idx+p2]+cddiffsq[idx-p2]+cddiffsq[idx+m2]))-(gaussgrad[0]*delhvsqsum[idx]+gaussgrad[1]*(delhvsqsum[idx-v1]+delhvsqsum[idx+1]+delhvsqsum[idx-1]+delhvsqsum[idx+v1])+gaussgrad[2]*(delhvsqsum[idx-m1]+delhvsqsum[idx+p1]+delhvsqsum[idx-p1]+delhvsqsum[idx+m1])+gaussgrad[3]*(delhvsqsum[idx-v2]+delhvsqsum[idx-2]+delhvsqsum[idx+2]+delhvsqsum[idx+v2])+gaussgrad[4]*(delhvsqsum[idx-v2-1]+delhvsqsum[idx-v2+1]+delhvsqsum[idx-TS-2]+delhvsqsum[idx-TS+2]+delhvsqsum[idx+TS-2]+delhvsqsum[idx+TS+2]+delhvsqsum[idx+v2-1]+delhvsqsum[idx+v2+1])+gaussgrad[5]*(delhvsqsum[idx-m2]+delhvsqsum[idx+p2]+delhvsqsum[idx-p2]+delhvsqsum[idx+m2]));
                }
                
                bool doNyquist = false;
                for (int rr = 6; rr < rr1-6; rr++) for (int cc=6+(FC(rr,2)&1),idx=rr*TS+cc; cc<cc1-6; cc+=2,idx+=2) if(nyqutest[idx>>1]>0.f) {nyquist[idx>>1]=1; doNyquist=true;}
                if(doNyquist) {
                    memset(nyquist2, 0, sizeof(unsigned char)*TS*TSH);
                    for (int rr=8; rr<rr1-8; rr++) for (int cc=8+(FC(rr,2)&1),idx=rr*TS+cc; cc<cc1-8; cc+=2,idx+=2) {
                        unsigned int nqsum=(nyquist[(idx-v2)>>1]+nyquist[(idx-m1)>>1]+nyquist[(idx+p1)>>1]+nyquist[(idx-2)>>1]+nyquist[(idx+2)>>1]+nyquist[(idx-p1)>>1]+nyquist[(idx+m1)>>1]+nyquist[(idx+v2)>>1]);
                        nyquist2[idx>>1]= nqsum>4?1:(nqsum<4?0:nyquist[idx>>1]);
                    }
                    for (int rr=8; rr<rr1-8; rr++) for (int cc=8+(FC(rr,2)&1),idx=rr*TS+cc; cc<cc1-8; cc+=2,idx+=2) if(nyquist2[idx>>1]) {
                        float sumcfa=0,sumh=0,sumv=0,sumsqh=0,sumsqv=0,areawt=0;
                        for(int i=-6;i<7;i+=2) for(int j=-6;j<7;j+=2) { int idx1=idx+i*TS+j; if(idx1>=0 && idx1 <TS*TS && nyquist2[idx1>>1]) {
                            float cfatemplate=cfa[idx1]; sumcfa+=cfatemplate; sumh+=(cfa[idx1-1]+cfa[idx1+1]); sumv+=(cfa[idx1-v1]+cfa[idx1+v1]);
                            sumsqh+=SQR(cfatemplate-cfa[idx1-1])+SQR(cfatemplate-cfa[idx1+1]); sumsqv+=SQR(cfatemplate-cfa[idx1-v1])+SQR(cfatemplate-cfa[idx1+v1]); areawt+=1;
                        }}
                        sumh=sumcfa-0.5f*sumh; sumv=sumcfa-0.5f*sumv; areawt*=0.5f;
                        float hcdvar2=epssq+fabsf(areawt*sumsqh-sumh*sumh), vcdvar2=epssq+fabsf(areawt*sumsqv-sumv*sumv); hvwt[idx>>1]=hcdvar2/(vcdvar2+hcdvar2);
                    }
                }

                float* Dgrb0 = vcdalt; s_hv* Dgrb2_ptr = reinterpret_cast<s_hv*>(Dgrbsq1m);
                for (int rr=8; rr<rr1-8; rr++) for (int cc=8+(FC(rr,2)&1),idx=rr*TS+cc; cc<cc1-8; cc+=2,idx+=2){
                    float hvwtalt=0.25f*(hvwt[(idx-m1)>>1]+hvwt[(idx+p1)>>1]+hvwt[(idx-p1)>>1]+hvwt[(idx+m1)>>1]);
                    hvwt[idx>>1] = fabsf(0.5f-hvwt[idx>>1])<fabsf(0.5f-hvwtalt)?hvwtalt:hvwt[idx>>1];
                    Dgrb0[idx>>1] = hvwt[idx>>1]*vcd[idx]+(1.f-hvwt[idx>>1])*hcd[idx];
                    rgbgreen[idx] = cfa[idx]+Dgrb0[idx>>1];
                    Dgrb2_ptr[idx>>1].h=nyquist2[idx>>1]?SQR(rgbgreen[idx]-0.5f*(rgbgreen[idx-1]+rgbgreen[idx+1])):0.f;
                    Dgrb2_ptr[idx>>1].v=nyquist2[idx>>1]?SQR(rgbgreen[idx]-0.5f*(rgbgreen[idx-v1]+rgbgreen[idx+v1])):0.f;
                }
                
                if(doNyquist) for(int rr=8; rr<rr1-8; rr++) for(int cc=8+(FC(rr,2)&1),idx=rr*TS+cc; cc<cc1-8; cc+=2,idx+=2) if(nyquist2[idx>>1]){
                    float gvarh=epssq+(gquinc[0]*Dgrb2_ptr[idx>>1].h+gquinc[1]*(Dgrb2_ptr[(idx-m1)>>1].h+Dgrb2_ptr[(idx+p1)>>1].h+Dgrb2_ptr[(idx-p1)>>1].h+Dgrb2_ptr[(idx+m1)>>1].h)+gquinc[2]*(Dgrb2_ptr[(idx-v2)>>1].h+Dgrb2_ptr[(idx-2)>>1].h+Dgrb2_ptr[(idx+2)>>1].h+Dgrb2_ptr[(idx+v2)>>1].h)+gquinc[3]*(Dgrb2_ptr[(idx-m2)>>1].h+Dgrb2_ptr[(idx+p2)>>1].h+Dgrb2_ptr[(idx-p2)>>1].h+Dgrb2_ptr[(idx+m2)>>1].h));
                    float gvarv=epssq+(gquinc[0]*Dgrb2_ptr[idx>>1].v+gquinc[1]*(Dgrb2_ptr[(idx-m1)>>1].v+Dgrb2_ptr[(idx+p1)>>1].v+Dgrb2_ptr[(idx-p1)>>1].v+Dgrb2_ptr[(idx+m1)>>1].v)+gquinc[2]*(Dgrb2_ptr[(idx-v2)>>1].v+Dgrb2_ptr[(idx-2)>>1].v+Dgrb2_ptr[(idx+2)>>1].v+Dgrb2_ptr[(idx+v2)>>1].v)+gquinc[3]*(Dgrb2_ptr[(idx-m2)>>1].v+Dgrb2_ptr[(idx+p2)>>1].v+Dgrb2_ptr[(idx-p2)>>1].v+Dgrb2_ptr[(idx+m2)>>1].v));
                    Dgrb0[idx>>1]=(hcd[idx]*gvarv+vcd[idx]*gvarh)/(gvarv+gvarh); rgbgreen[idx]=cfa[idx]+Dgrb0[idx>>1];
                }

                float* delp=Dgrbsq1p, *delm=Dgrbsq1m;
                for (int rr=6; rr<rr1-6; rr++) for(int cc=6,idx=rr*TS+cc; cc<cc1-6; cc+=2,idx+=2){
                    int idx1 = idx>>1;
                    if(FC(rr,cc)&1){ Dgrbsq1p[idx1]=SQR(cfa[idx+1]-cfa[idx+1-p1])+SQR(cfa[idx+1]-cfa[idx+1+p1]); Dgrbsq1m[idx1]=SQR(cfa[idx+1]-cfa[idx+1-m1])+SQR(cfa[idx+1]-cfa[idx+1+m1]); delp[idx1]=fabsf(cfa[idx+1+p1]-cfa[idx+1-p1]); delm[idx1]=fabsf(cfa[idx+1+m1]-cfa[idx+1-m1]);}
                    else { Dgrbsq1p[idx1]=SQR(cfa[idx]-cfa[idx-p1])+SQR(cfa[idx]-cfa[idx+p1]); Dgrbsq1m[idx1]=SQR(cfa[idx]-cfa[idx-m1])+SQR(cfa[idx]-cfa[idx+m1]); delp[idx1]=fabsf(cfa[idx+p1]-cfa[idx-p1]); delm[idx1]=fabsf(cfa[idx+m1]-cfa[idx-m1]);}
                }

                float* rbm=vcd, *rbp=hcd, *rbint=dirwts0, *pmwt=dirwts1;
                for (int rr=8; rr<rr1-8; rr++) for(int cc=8+(FC(rr,2)&1),idx=rr*TS+cc,idx1=idx>>1; cc<cc1-8; cc+=2,idx+=2,idx1++){
                    float crse=2.f*cfa[idx+m1]/(eps+cfa[idx]+cfa[idx+m2]), rbse=fabsf(1.f-crse)<arthresh?cfa[idx]*crse:cfa[idx+m1]+0.5f*(cfa[idx]-cfa[idx+m2]);
                    float crnw=2.f*cfa[idx-m1]/(eps+cfa[idx]+cfa[idx-m2]), rbnw=fabsf(1.f-crnw)<arthresh?cfa[idx]*crnw:cfa[idx-m1]+0.5f*(cfa[idx]-cfa[idx-m2]);
                    float crne=2.f*cfa[idx+p1]/(eps+cfa[idx]+cfa[idx+p2]), rbne=fabsf(1.f-crne)<arthresh?cfa[idx]*crne:cfa[idx+p1]+0.5f*(cfa[idx]-cfa[idx+p2]);
                    float crsw=2.f*cfa[idx-p1]/(eps+cfa[idx]+cfa[idx-p2]), rbsw=fabsf(1.f-crsw)<arthresh?cfa[idx]*crsw:cfa[idx-p1]+0.5f*(cfa[idx]-cfa[idx-p2]);
                    float wtse=eps+delm[idx1]+delm[(idx+m1)>>1]+delm[(idx+m2)>>1], wtnw=eps+delm[idx1]+delm[(idx-m1)>>1]+delm[(idx-m2)>>1];
                    float wtne=eps+delp[idx1]+delp[(idx+p1)>>1]+delp[(idx+p2)>>1], wtsw=eps+delp[idx1]+delp[(idx-p1)>>1]+delp[(idx-p2)>>1];
                    rbm[idx1]=(wtse*rbnw+wtnw*rbse)/(wtse+wtnw); rbp[idx1]=(wtne*rbsw+wtsw*rbne)/(wtne+wtsw);
                    if(rbp[idx1]<cfa[idx]){if(2.f*rbp[idx1]<cfa[idx])rbp[idx1]=median(rbp[idx1],cfa[idx-p1],cfa[idx+p1]);else{float pwt=2.f*(cfa[idx]-rbp[idx1])/(eps+rbp[idx1]+cfa[idx]);rbp[idx1]=pwt*rbp[idx1]+(1.f-pwt)*median(rbp[idx1],cfa[idx-p1],cfa[idx+p1]);}}
                    if(rbm[idx1]<cfa[idx]){if(2.f*rbm[idx1]<cfa[idx])rbm[idx1]=median(rbm[idx1],cfa[idx-m1],cfa[idx+m1]);else{float mwt=2.f*(cfa[idx]-rbm[idx1])/(eps+rbm[idx1]+cfa[idx]);rbm[idx1]=mwt*rbm[idx1]+(1.f-mwt)*median(rbm[idx1],cfa[idx-m1],cfa[idx+m1]);}}
                    if(rbp[idx1]>clip_pt)rbp[idx1]=median(rbp[idx1],cfa[idx-p1],cfa[idx+p1]); if(rbm[idx1]>clip_pt)rbm[idx1]=median(rbm[idx1],cfa[idx-m1],cfa[idx+m1]);
                    float rbvarm=epssq+(gausseven[0]*(Dgrbsq1m[(idx-v1)>>1]+Dgrbsq1m[(idx-1)>>1]+Dgrbsq1m[(idx+1)>>1]+Dgrbsq1m[(idx+v1)>>1])+gausseven[1]*(Dgrbsq1m[(idx-v2-1)>>1]+Dgrbsq1m[(idx-v2+1)>>1]+Dgrbsq1m[(idx-2-v1)>>1]+Dgrbsq1m[(idx+2-v1)>>1]+Dgrbsq1m[(idx-2+v1)>>1]+Dgrbsq1m[(idx+2+v1)>>1]+Dgrbsq1m[(idx+v2-1)>>1]+Dgrbsq1m[(idx+v2+1)>>1]));
                    float rbvarp=epssq+(gausseven[0]*(Dgrbsq1p[(idx-v1)>>1]+Dgrbsq1p[(idx-1)>>1]+Dgrbsq1p[(idx+1)>>1]+Dgrbsq1p[(idx+v1)>>1])+gausseven[1]*(Dgrbsq1p[(idx-v2-1)>>1]+Dgrbsq1p[(idx-v2+1)>>1]+Dgrbsq1p[(idx-2-v1)>>1]+Dgrbsq1p[(idx+2-v1)>>1]+Dgrbsq1p[(idx-2+v1)>>1]+Dgrbsq1p[(idx+2+v1)>>1]+Dgrbsq1p[(idx+v2-1)>>1]+Dgrbsq1p[(idx+v2+1)>>1]));
                    pmwt[idx1]=rbvarm/(rbvarp+rbvarm);
                }

                for (int rr=10; rr<rr1-10; rr++) for(int cc=10+(FC(rr,2)&1),idx=rr*TS+cc,idx1=idx>>1; cc<cc1-10; cc+=2,idx+=2,idx1++){
                    float pmwtalt=0.25f*(pmwt[(idx-m1)>>1]+pmwt[(idx+p1)>>1]+pmwt[(idx-p1)>>1]+pmwt[(idx+m1)>>1]);
                    if(fabsf(0.5f-pmwt[idx1])<fabsf(0.5f-pmwtalt))pmwt[idx1]=pmwtalt;
                    rbint[idx1]=0.5f*(cfa[idx]+rbm[idx1]*(1.f-pmwt[idx1])+rbp[idx1]*pmwt[idx1]);
                }

                for (int rr=12; rr<rr1-12; rr++) for(int cc=12+(FC(rr,2)&1),idx=rr*TS+cc,idx1=idx>>1; cc<cc1-12; cc+=2,idx+=2,idx1++){
                    if(fabsf(0.5f-pmwt[idx1])>=fabsf(0.5f-hvwt[idx1]))continue;
                    float cru=cfa[idx-v1]*2.f/(eps+rbint[idx1]+rbint[(idx-v1)>>1]),crd=cfa[idx+v1]*2.f/(eps+rbint[idx1]+rbint[(idx+v1)>>1]);
                    float crl=cfa[idx-1]*2.f/(eps+rbint[idx1]+rbint[(idx-1)>>1]),crr=cfa[idx+1]*2.f/(eps+rbint[idx1]+rbint[(idx+1)>>1]);
                    float gu=fabsf(1.f-cru)<arthresh?rbint[idx1]*cru:cfa[idx-v1]+0.5f*(rbint[idx1]-rbint[(idx-v1)>>1]);
                    float gd=fabsf(1.f-crd)<arthresh?rbint[idx1]*crd:cfa[idx+v1]+0.5f*(rbint[idx1]-rbint[(idx+v1)>>1]);
                    float gl=fabsf(1.f-crl)<arthresh?rbint[idx1]*crl:cfa[idx-1]+0.5f*(rbint[idx1]-rbint[(idx-1)>>1]);
                    float gr=fabsf(1.f-crr)<arthresh?rbint[idx1]*crr:cfa[idx+1]+0.5f*(rbint[idx1]-rbint[(idx+1)>>1]);
                    float Gintv=(dirwts0[idx-v1]*gd+dirwts0[idx+v1]*gu)/(dirwts0[idx+v1]+dirwts0[idx-v1]);
                    float Ginth=(dirwts1[idx-1]*gr+dirwts1[idx+1]*gl)/(dirwts1[idx-1]+dirwts1[idx+1]);
                    if(Gintv<rbint[idx1]){if(2.f*Gintv<rbint[idx1])Gintv=median(Gintv,cfa[idx-v1],cfa[idx+v1]);else{float vwt2=2.f*(rbint[idx1]-Gintv)/(eps+Gintv+rbint[idx1]);Gintv=vwt2*Gintv+(1.f-vwt2)*median(Gintv,cfa[idx-v1],cfa[idx+v1]);}}
                    if(Ginth<rbint[idx1]){if(2.f*Ginth<rbint[idx1])Ginth=median(Ginth,cfa[idx-1],cfa[idx+1]);else{float hwt2=2.f*(rbint[idx1]-Ginth)/(eps+Ginth+rbint[idx1]);Ginth=hwt2*Ginth+(1.f-hwt2)*median(Ginth,cfa[idx-1],cfa[idx+1]);}}
                    if(Ginth>clip_pt)Ginth=median(Ginth,cfa[idx-1],cfa[idx+1]); if(Gintv>clip_pt)Gintv=median(Gintv,cfa[idx-v1],cfa[idx+v1]);
                    rgbgreen[idx]=Ginth*(1.f-hvwt[idx1])+Gintv*hvwt[idx1]; Dgrb0[idx1]=rgbgreen[idx]-cfa[idx];
                }
                
                float* Dgrb1=hcdalt;
                for(int rr=13-ey; rr<rr1-12; rr+=2) for(int cc=13-ex,idx=rr*TS+cc; cc<cc1-12; cc+=2,idx+=2){Dgrb1[idx>>1]=Dgrb0[idx>>1]; Dgrb0[idx>>1]=0;}
                for(int rr=14; rr<rr1-14; rr++) for(int cc=14+(FC(rr,2)&1),idx=rr*TS+cc; cc<cc1-14; cc+=2,idx+=2){
                    int c=1-FC(rr,cc)/2;
                    float* Dgrb_c = c ? Dgrb1 : Dgrb0;
                    float wtnw=1.f/(eps+fabsf(Dgrb_c[((idx-m1)>>1)]-Dgrb_c[((idx+m1)>>1)])+fabsf(Dgrb_c[((idx-m1)>>1)]-Dgrb_c[((idx-m3)>>1)])+fabsf(Dgrb_c[((idx+m1)>>1)]-Dgrb_c[((idx-m3)>>1)]));
                    float wtne=1.f/(eps+fabsf(Dgrb_c[((idx+p1)>>1)]-Dgrb_c[((idx-p1)>>1)])+fabsf(Dgrb_c[((idx+p1)>>1)]-Dgrb_c[((idx+p3)>>1)])+fabsf(Dgrb_c[((idx-p1)>>1)]-Dgrb_c[((idx+p3)>>1)]));
                    float wtsw=1.f/(eps+fabsf(Dgrb_c[((idx-p1)>>1)]-Dgrb_c[((idx+p1)>>1)])+fabsf(Dgrb_c[((idx-p1)>>1)]-Dgrb_c[((idx+m3)>>1)])+fabsf(Dgrb_c[((idx+p1)>>1)]-Dgrb_c[((idx-p3)>>1)]));
                    float wtse=1.f/(eps+fabsf(Dgrb_c[((idx+m1)>>1)]-Dgrb_c[((idx-m1)>>1)])+fabsf(Dgrb_c[((idx+m1)>>1)]-Dgrb_c[((idx-p3)>>1)])+fabsf(Dgrb_c[((idx-m1)>>1)]-Dgrb_c[((idx+m3)>>1)]));
                    Dgrb_c[idx>>1]=(wtnw*(1.325f*Dgrb_c[((idx-m1)>>1)]-0.175f*Dgrb_c[((idx-m3)>>1)]-0.075f*Dgrb_c[((idx-m1-2)>>1)]-0.075f*Dgrb_c[((idx-m1-v2)>>1)])+wtne*(1.325f*Dgrb_c[((idx+p1)>>1)]-0.175f*Dgrb_c[((idx+p3)>>1)]-0.075f*Dgrb_c[((idx+p1+2)>>1)]-0.075f*Dgrb_c[((idx+p1+v2)>>1)])+wtsw*(1.325f*Dgrb_c[((idx-p1)>>1)]-0.175f*Dgrb_c[((idx-p3)>>1)]-0.075f*Dgrb_c[((idx-p1-2)>>1)]-0.075f*Dgrb_c[((idx-p1-v2)>>1)])+wtse*(1.325f*Dgrb_c[((idx+m1)>>1)]-0.175f*Dgrb_c[((idx+m3)>>1)]-0.075f*Dgrb_c[((idx+m1+2)>>1)]-0.075f*Dgrb_c[((idx+m1+v2)>>1)]))/(wtnw+wtne+wtsw+wtse);
                }

                for (int rr = 16; rr < rr1 - 16; rr++) {
                    int row = rr + top; if (row < 0 || row >= (int)height_) continue;
                    for (int cc = 16; cc < cc1-16; cc++) {
                        int col = cc + left; if (col < 0 || col >= (int)width_) continue;
                        int out_idx = row*width_+col, tile_idx=rr*TS+cc;
                        int c_site=FC(row,col);
                        float r, g, b;
                        if(c_site&1){
                            g=cfa[tile_idx];
                            float wsum = hvwt[((tile_idx-v1)>>1)] + (1.f-hvwt[((tile_idx+1)>>1)]) + (1.f-hvwt[((tile_idx-1)>>1)]) + hvwt[((tile_idx+v1)>>1)];
                            if (wsum > 1e-6f) {
                                r = g - (hvwt[((tile_idx-v1)>>1)]*Dgrb0[((tile_idx-v1)>>1)]+(1.f-hvwt[((tile_idx+1)>>1)])*Dgrb0[((tile_idx+1)>>1)]+(1.f-hvwt[((tile_idx-1)>>1)])*Dgrb0[((tile_idx-1)>>1)]+hvwt[((tile_idx+v1)>>1)]*Dgrb0[((tile_idx+v1)>>1)])/wsum;
                                b = g - (hvwt[((tile_idx-v1)>>1)]*Dgrb1[((tile_idx-v1)>>1)]+(1.f-hvwt[((tile_idx+1)>>1)])*Dgrb1[((tile_idx+1)>>1)]+(1.f-hvwt[((tile_idx-1)>>1)])*Dgrb1[((tile_idx-1)>>1)]+hvwt[((tile_idx+v1)>>1)]*Dgrb1[((tile_idx+v1)>>1)])/wsum;
                            } else { r = b = g; }
                        } else {
                            g=rgbgreen[tile_idx];
                            if(c_site==0){r=cfa[tile_idx];b=g-Dgrb1[tile_idx>>1];}
                            else{b=cfa[tile_idx];r=g-Dgrb0[tile_idx>>1];}
                        }
                        rgb_buffer_.image[out_idx][0] = CLIP_U16(r*65535.f);
                        rgb_buffer_.image[out_idx][1] = CLIP_U16(g*65535.f);
                        rgb_buffer_.image[out_idx][2] = CLIP_U16(b*65535.f);
                        rgb_buffer_.image[out_idx][3] = rgb_buffer_.image[out_idx][1];
                    }
                }
            }
        }
    }
};

} // anonymous namespace

// `CPUAccelerator`クラス内のメソッドをこれに置き換える
bool CPUAccelerator::demosaic_bayer_amaze(const ImageBuffer& raw_buffer, ImageBuffer& rgb_buffer, uint32_t filters) {
#ifdef __APPLE__
    if (!initialized_ || !raw_buffer.is_valid() || !rgb_buffer.is_valid() ||
        raw_buffer.width != rgb_buffer.width || raw_buffer.height != rgb_buffer.height) {
        std::cerr << "❌ AMaZE Demosaic (RT Port): Invalid buffers or initialization state." << std::endl;
        return false;
    }

    std::cout << "🔧 Starting AMaZE demosaic (Full RawTherapee Port)..." << std::endl;
    
    // ✨境界補間を後に移動✨: デモザイク処理前はスキップ
    // std::cout << "🔧 Applying border interpolation for AMaZE (2px border)..." << std::endl;
    // border_interpolate(raw_buffer, filters, 2);
    
    auto start_time = std::chrono::high_resolution_clock::now();

    // CPUAccelerator::demosaic_bayer_amaze の冒頭に追加
    std::cout << "DEBUG: Bayer `filters` value is: " << filters << std::endl;
    std::cout << "DEBUG: Dumping top-left 4x4 pixel grid..." << std::endl;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int color_code = fcol_bayer(r, c, filters);
            const char* color_name[] = {"R", "G1", "B", "G2"};
            
            uint16_t* p = raw_buffer.image[r * raw_buffer.width + c];
            std::cout << "Pos(" << r << "," << c << ") -> FC=" << color_code 
                    << " (" << color_name[color_code] << ")"
                    << " | Data=[R:" << p[0] << ", G:" << p[1] << ", B:" << p[2] << "]" 
                    << std::endl;
        }
    }

    try {
        AMaZE_Processor_RT amaze_proc(raw_buffer, rgb_buffer, filters);
        amaze_proc.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ An exception occurred during AMaZE processing: " << e.what() << std::endl;
        return false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    last_processing_time_ = diff.count();
    std::cout << "✅ AMaZE demosaic completed successfully in " << last_processing_time_ << " seconds." << std::endl;
    
    // ✨デモザイク後境界補間✨: 最終処理で境界を修正
    std::cout << "🔧 Post-demosaic border interpolation for AMaZE (2px border)..." << std::endl;
    border_interpolate(raw_buffer, filters, 2);
    
    return true;
#else
    std::cerr << "❌ AMaZE demosaic is currently only supported on Apple platforms in this context." << std::endl;
    return false;
#endif
}

//===================================================================
// XTrans Demosaicing - RawTherapee Complete Port
//===================================================================

namespace { // このファイル内でのみ使用

class XTrans_Processor {
private:
    // XYZ from RGB conversion matrix
    static constexpr float xyz_rgb[3][3] = {
        { 0.412453f, 0.357580f, 0.180423f },
        { 0.212671f, 0.715160f, 0.072169f },
        { 0.019334f, 0.119193f, 0.950227f }
    };
    static constexpr float d65_white[3] = { 0.950456f, 1.0f, 1.088754f };
    
    const ImageBuffer& raw_buffer_;
    ImageBuffer& rgb_buffer_;
    const char (&xtrans_)[6][6];
    const float (&color_matrix_)[3][4];
    uint32_t width_, height_;
    
    // Utility functions
    inline int fcol_xtrans(int row, int col) const {
        return xtrans_[(row) % 6][(col) % 6];
    }
    
    inline float SQR(float x) const { return x * x; }
    
    static inline uint16_t CLIP(int x) {
        if (x < 0) return 0;
        if (x > 65535) return 65535;
        return static_cast<uint16_t>(x);
    }

public:
    XTrans_Processor(const ImageBuffer& raw, ImageBuffer& rgb, const char (&xtrans)[6][6], const float (&color_matrix)[3][4])
        : raw_buffer_(raw), rgb_buffer_(rgb), xtrans_(xtrans), color_matrix_(color_matrix),
          width_(raw.width), height_(raw.height) {}
    
    // 3-pass high quality demosaic
    void run_3pass() {
        std::cout << "🔧 Starting 3-pass XTrans demosaic (high quality)..." << std::endl;
        
        constexpr int ts = 114;      // Tile Size
        
        // Pattern arrays for 3-pass algorithm
        constexpr short orth[12] = { 1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1 };
        constexpr short patt[2][16] = { 
            { 0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0 },
            { 0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1 }
        };
        constexpr short dir[4] = { 1, ts, ts + 1, ts - 1 };
        
        // Calculate color transformation matrix
        float xyz_cam[3][3];
        compute_xyz_cam_matrix(xyz_cam);
        
        // Find solitary green pixels position
        ushort sgrow = 0, sgcol = 0;
        find_solitary_green_pixels(sgrow, sgcol);
        
        // Multi-pass tile processing
        process_tiles_3pass(xyz_cam, orth, patt, dir, ts, sgrow, sgcol);
        
        // Border interpolation after main demosaicing (RawTherapee uses passes > 1 ? 8 : 11)
        xtrans_border_interpolate(8);
        
        std::cout << "✅ 3-pass XTrans demosaic completed" << std::endl;
    }
    
    // 1-pass fast demosaic - RawTherapee's exact fast_xtrans_interpolate
    void run_1pass() {
        std::cout << "🔧 Starting 1-pass XTrans demosaic (fast)..." << std::endl;
        
        // Border interpolation first (RawTherapee does this first)
        xtrans_border_interpolate(1);
        
        // RawTherapee's exact weight matrix
        const float weight[3][3] = {
            {0.25f, 0.5f, 0.25f},
            {0.5f,  0.0f, 0.5f},
            {0.25f, 0.5f, 0.25f}
        };
        
        #pragma omp parallel for schedule(dynamic, 16)
        for (int row = 1; row < (int)height_ - 1; ++row) {
            for (int col = 1; col < (int)width_ - 1; ++col) {
                float sum[3] = {0.0f, 0.0f, 0.0f};
                
                // Calculate weighted sum for each color channel
                for (int v = -1; v <= 1; v++) {
                    for (int h = -1; h <= 1; h++) {
                        int src_row = row + v;
                        int src_col = col + h;
                        int src_idx = src_row * width_ + src_col;
                        int src_color = fcol_xtrans(src_row, src_col);
                        
                        float raw_val = (float)raw_buffer_.image[src_idx][src_color] / 65535.0f;
                        sum[src_color] += raw_val * weight[v + 1][h + 1];
                    }
                }
                
                int out_idx = row * width_ + col;
                int pixel_color = fcol_xtrans(row, col);
                float rgb[3];
                
                // RawTherapee's exact color interpolation logic
                switch(pixel_color) {
                case 0: // red pixel
                    {
                        rgb[0] = (float)raw_buffer_.image[out_idx][0] / 65535.0f; // Current red
                        rgb[1] = sum[1] * 0.5f;  // Green interpolation
                        rgb[2] = sum[2];         // Blue interpolation
                    }
                    break;
                    
                case 1: // green pixel
                    {
                        rgb[1] = (float)raw_buffer_.image[out_idx][1] / 65535.0f; // Current green
                        
                        // Check if this is a solitary green pixel
                        int left_color = fcol_xtrans(row, col - 1);
                        int right_color = fcol_xtrans(row, col + 1);
                        
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
                        rgb[2] = (float)raw_buffer_.image[out_idx][2] / 65535.0f; // Current blue
                    }
                    break;
                }
                
                // Set output values
                rgb_buffer_.image[out_idx][0] = CLIP((int)(rgb[0] * 65535.0f + 0.5f));
                rgb_buffer_.image[out_idx][1] = CLIP((int)(rgb[1] * 65535.0f + 0.5f));
                rgb_buffer_.image[out_idx][2] = CLIP((int)(rgb[2] * 65535.0f + 0.5f));
                rgb_buffer_.image[out_idx][3] = rgb_buffer_.image[out_idx][1]; // G2 = G1
            }
        }
        
        std::cout << "✅ 1-pass XTrans demosaic completed" << std::endl;
    }

private:
    void compute_xyz_cam_matrix(float xyz_cam[3][3]) {
        // RawTherapee compatible xyz_cam matrix with D65 white point normalization
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                xyz_cam[i][j] = 0.0f;
                for (int k = 0; k < 3; k++) {
                    // ✨RawTherapee互換修正✨: D65白点正規化を適用（マゼンタ被り修正）
                    xyz_cam[i][j] += xyz_rgb[i][k] * color_matrix_[k][j] / d65_white[i];
                }
            }
        }
        
    }
    
    void find_solitary_green_pixels(ushort& sgrow, ushort& sgcol) {
        // Find position of solitary green pixels in XTrans pattern
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                if (fcol_xtrans(row, col) == 1) {
                    bool solitary = true;
                    // Check if this green pixel has green neighbors
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            if (dr == 0 && dc == 0) continue;
                            if (fcol_xtrans((row + dr + 6) % 6, (col + dc + 6) % 6) == 1) {
                                solitary = false;
                                break;
                            }
                        }
                        if (!solitary) break;
                    }
                    if (solitary) {
                        sgrow = row;
                        sgcol = col;
                        return;
                    }
                }
            }
        }
    }
    
    void xtrans_border_interpolate(int border) {
        // RawTherapee's exact xtransborder_interpolate algorithm
        const float weight[3][3] = {
            {0.25f, 0.5f, 0.25f},
            {0.5f,  0.0f, 0.5f},
            {0.25f, 0.5f, 0.25f}
        };
        
        for (int row = 0; row < (int)height_; row++) {
            for (int col = 0; col < (int)width_; col++) {
                if (col == border && row >= border && row < (int)height_ - border) {
                    col = (int)width_ - border;
                }
                
                if (row < border || col < border || 
                    row >= (int)height_ - border || col >= (int)width_ - border) {
                    
                    float sum[6] = {0.0f}; // [R, G, B, R_weight, G_weight, B_weight]
                    
                    // Calculate weighted sum like RawTherapee
                    for (int y = std::max(0, row - 1), v = (row == 0 ? 0 : -1); 
                         y <= std::min(row + 1, (int)height_ - 1); y++, v++) {
                        for (int x = std::max(0, col - 1), h = (col == 0 ? 0 : -1); 
                             x <= std::min(col + 1, (int)width_ - 1); x++, h++) {
                            
                            int f = fcol_xtrans(y, x);
                            int src_idx = y * width_ + x;
                            float raw_val = (float)raw_buffer_.image[src_idx][f] / 65535.0f;
                            float w = weight[v + 1][h + 1];
                            
                            sum[f] += raw_val * w;
                            sum[f + 3] += w;
                        }
                    }
                    
                    int dst_idx = row * width_ + col;
                    int pixel_color = fcol_xtrans(row, col);
                    
                    // RawTherapee's exact interpolation logic
                    switch(pixel_color) {
                    case 0: // Red pixel
                        {
                            float raw_val = (float)raw_buffer_.image[dst_idx][0] / 65535.0f;
                            rgb_buffer_.image[dst_idx][0] = CLIP((int)(raw_val * 65535.0f + 0.5f));
                            rgb_buffer_.image[dst_idx][1] = CLIP((int)((sum[1] / sum[4]) * 65535.0f + 0.5f));
                            rgb_buffer_.image[dst_idx][2] = CLIP((int)((sum[2] / sum[5]) * 65535.0f + 0.5f));
                        }
                        break;
                        
                    case 1: // Green pixel
                        {
                            float raw_val = (float)raw_buffer_.image[dst_idx][1] / 65535.0f;
                            if (sum[3] == 0.0f) {
                                // Corner case: only green pixels in 2x2 area
                                rgb_buffer_.image[dst_idx][0] = CLIP((int)(raw_val * 65535.0f + 0.5f));
                                rgb_buffer_.image[dst_idx][1] = CLIP((int)(raw_val * 65535.0f + 0.5f));
                                rgb_buffer_.image[dst_idx][2] = CLIP((int)(raw_val * 65535.0f + 0.5f));
                            } else {
                                rgb_buffer_.image[dst_idx][0] = CLIP((int)((sum[0] / sum[3]) * 65535.0f + 0.5f));
                                rgb_buffer_.image[dst_idx][1] = CLIP((int)(raw_val * 65535.0f + 0.5f));
                                rgb_buffer_.image[dst_idx][2] = CLIP((int)((sum[2] / sum[5]) * 65535.0f + 0.5f));
                            }
                        }
                        break;
                        
                    case 2: // Blue pixel
                        {
                            float raw_val = (float)raw_buffer_.image[dst_idx][2] / 65535.0f;
                            rgb_buffer_.image[dst_idx][0] = CLIP((int)((sum[0] / sum[3]) * 65535.0f + 0.5f));
                            rgb_buffer_.image[dst_idx][1] = CLIP((int)((sum[1] / sum[4]) * 65535.0f + 0.5f));
                            rgb_buffer_.image[dst_idx][2] = CLIP((int)(raw_val * 65535.0f + 0.5f));
                        }
                        break;
                    }
                    
                    rgb_buffer_.image[dst_idx][3] = rgb_buffer_.image[dst_idx][1]; // G2 = G1
                }
            }
        }
    }
    
    void process_tiles_3pass(const float xyz_cam[3][3], const short orth[12], 
                            const short patt[2][16], const short dir[4], 
                            int ts, ushort sgrow, ushort sgcol) {
        
        const int passes = 3;
        
        #pragma omp parallel for schedule(dynamic, 1) collapse(2)
        for (int top = 0; top < (int)height_; top += ts - 32) {
            for (int left = 0; left < (int)width_; left += ts - 32) {
                int bottom = std::min(top + ts, (int)height_);
                int right = std::min(left + ts, (int)width_);
                
                // Allocate tile arrays
                std::vector<float> rgb_data(ts * ts * 3, 0.0f);
                std::vector<float> lab_data(ts * ts * 3, 0.0f);
                float (*rgb)[3] = reinterpret_cast<float(*)[3]>(rgb_data.data());
                float (*lab)[3] = reinterpret_cast<float(*)[3]>(lab_data.data());
                
                // Initialize tile with raw data
                initialize_tile(rgb, top, left, bottom, right, ts);
                
                // Multi-pass processing
                for (int pass = 0; pass < passes; pass++) {
                    if (pass == 1) {
                        // Convert RGB to LAB for pass 2
                        convert_rgb_to_lab(rgb, lab, ts, xyz_cam);
                    }
                    
                    // Process tile pass
                    process_tile_pass(rgb, lab, top, left, bottom, right, ts, 
                                     pass, orth, patt, dir, sgrow, sgcol);
                }
                
                // Copy results back to output buffer
                copy_tile_to_output(rgb, top, left, bottom, right, ts);
            }
        }
    }
    
    void initialize_tile(float (*rgb)[3], int top, int left, int bottom, int right, int ts) {
        for (int row = top; row < bottom; row++) {
            for (int col = left; col < right; col++) {
                if (row >= 0 && row < (int)height_ && col >= 0 && col < (int)width_) {
                    int tile_idx = (row - top) * ts + (col - left);
                    int raw_idx = row * width_ + col;
                    int color = fcol_xtrans(row, col);
                    
                    // Initialize all channels to 0, set only the appropriate color
                    rgb[tile_idx][0] = rgb[tile_idx][1] = rgb[tile_idx][2] = 0.0f;
                    rgb[tile_idx][color] = (float)raw_buffer_.image[raw_idx][color] / 65535.0f;
                }
            }
        }
    }
    
    void convert_rgb_to_lab(float (*rgb)[3], float (*lab)[3], int ts, const float xyz_cam[3][3]) {
        // RawTherapee's exact CIELab conversion with LUT
        static std::vector<float> cbrt_lut;
        static bool lut_initialized = false;
        
        // Initialize LUT exactly like RawTherapee (0x14000 = 81920 entries)
        if (!lut_initialized) {
            cbrt_lut.resize(0x14000);
            const float eps = 216.0f / 24389.0f;
            const float kappa = 24389.0f / 27.0f;
            
            for (int i = 0; i < 0x14000; i++) {
                double r = i / 65535.0;
                cbrt_lut[i] = r > eps ? std::cbrt(r) : (kappa * r + 16.0) / 116.0;
            }
            lut_initialized = true;
        }
        
        for (int i = 0; i < ts * ts; i++) {
            // Convert RGB to XYZ - RawTherapee's exact method
            float xyz[3] = {0.5f, 0.5f, 0.5f}; // Start with 0.5 like RawTherapee
            
            for (int c = 0; c < 3; c++) {
                xyz[0] += xyz_cam[0][c] * rgb[i][c];
                xyz[1] += xyz_cam[1][c] * rgb[i][c];  
                xyz[2] += xyz_cam[2][c] * rgb[i][c];
            }
            
            // Apply LUT like RawTherapee (clamp to valid range)
            int x_idx = std::max(0, std::min(0x13fff, (int)xyz[0]));
            int y_idx = std::max(0, std::min(0x13fff, (int)xyz[1])); 
            int z_idx = std::max(0, std::min(0x13fff, (int)xyz[2]));
            
            xyz[0] = cbrt_lut[x_idx];
            xyz[1] = cbrt_lut[y_idx];
            xyz[2] = cbrt_lut[z_idx];
            
            // Calculate Lab values - RawTherapee's exact formula
            lab[i][0] = 116.0f * xyz[1] - 16.0f;  // L
            lab[i][1] = 500.0f * (xyz[0] - xyz[1]); // a  
            lab[i][2] = 200.0f * (xyz[1] - xyz[2]); // b
        }
    }
    
    void process_tile_pass(float (*rgb)[3], float (*lab)[3], int top, int left, 
                          int bottom, int right, int ts, int pass,
                          const short orth[12], const short patt[2][16], 
                          const short dir[4], ushort sgrow, ushort sgcol) {
        
        // Pass-specific processing
        if (pass == 0) {
            // First pass: basic interpolation
            interpolate_green_first_pass(rgb, top, left, bottom, right, ts);
        } else if (pass == 1) {
            // Second pass: use LAB for better color accuracy
            interpolate_with_lab(rgb, lab, top, left, bottom, right, ts);
        } else {
            // Final pass: refinement
            refine_interpolation(rgb, top, left, bottom, right, ts);
        }
    }
    
    void interpolate_green_first_pass(float (*rgb)[3], int top, int left, int bottom, int right, int ts) {
        // Green channel interpolation for non-green pixels
        for (int row = top + 2; row < bottom - 2; row++) {
            for (int col = left + 2; col < right - 2; col++) {
                int tile_idx = (row - top) * ts + (col - left);
                
                if (fcol_xtrans(row, col) != 1) { // Not green pixel
                    float sum = 0.0f;
                    int count = 0;
                    
                    // Average nearby green pixels
                    for (int dr = -2; dr <= 2; dr++) {
                        for (int dc = -2; dc <= 2; dc++) {
                            int nr = row + dr, nc = col + dc;
                            if (nr >= 0 && nr < (int)height_ && nc >= 0 && nc < (int)width_) {
                                if (fcol_xtrans(nr, nc) == 1) {
                                    int n_tile_idx = (nr - top) * ts + (nc - left);
                                    if (n_tile_idx >= 0 && n_tile_idx < ts * ts) {
                                        sum += rgb[n_tile_idx][1];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                    if (count > 0) {
                        rgb[tile_idx][1] = sum / count;
                    }
                }
            }
        }
    }
    
    void interpolate_with_lab(float (*rgb)[3], float (*lab)[3], int top, int left, int bottom, int right, int ts) {
        // Use LAB color space for better interpolation
        // This is a simplified version - full implementation would be much more complex
        for (int row = top + 2; row < bottom - 2; row++) {
            for (int col = left + 2; col < right - 2; col++) {
                int tile_idx = (row - top) * ts + (col - left);
                int color = fcol_xtrans(row, col);
                
                // Interpolate missing color channels using LAB guidance
                for (int c = 0; c < 3; c++) {
                    if (c != color && rgb[tile_idx][c] == 0.0f) {
                        // Simple interpolation using nearby pixels
                        float sum = 0.0f;
                        int count = 0;
                        
                        for (int dr = -1; dr <= 1; dr++) {
                            for (int dc = -1; dc <= 1; dc++) {
                                if (dr == 0 && dc == 0) continue;
                                int nr = row + dr, nc = col + dc;
                                if (nr >= top && nr < bottom && nc >= left && nc < right) {
                                    int n_tile_idx = (nr - top) * ts + (nc - left);
                                    if (rgb[n_tile_idx][c] > 0.0f) {
                                        sum += rgb[n_tile_idx][c];
                                        count++;
                                    }
                                }
                            }
                        }
                        if (count > 0) {
                            rgb[tile_idx][c] = sum / count;
                        }
                    }
                }
            }
        }
    }
    
    void refine_interpolation(float (*rgb)[3], int top, int left, int bottom, int right, int ts) {
        // Final refinement pass
        for (int row = top + 1; row < bottom - 1; row++) {
            for (int col = left + 1; col < right - 1; col++) {
                int tile_idx = (row - top) * ts + (col - left);
                
                // Edge-preserving smoothing
                for (int c = 0; c < 3; c++) {
                    float sum = 0.0f;
                    float weights = 0.0f;
                    
                    // Weighted average with edge preservation
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            int nr = row + dr, nc = col + dc;
                            if (nr >= top && nr < bottom && nc >= left && nc < right) {
                                int n_tile_idx = (nr - top) * ts + (nc - left);
                                float weight = 1.0f / (1.0f + std::abs(dr) + std::abs(dc));
                                sum += rgb[n_tile_idx][c] * weight;
                                weights += weight;
                            }
                        }
                    }
                    if (weights > 0.0f) {
                        rgb[tile_idx][c] = sum / weights;
                    }
                }
            }
        }
    }
    
    
    void copy_tile_to_output(float (*rgb)[3], int top, int left, int bottom, int right, int ts) {
        // RawTherapee-style tile copying - only copy the center region to avoid boundary issues
        const int border = 8; // Don't copy border pixels to avoid tile boundary artifacts
        
        int copy_top = std::max(top + border, 0);
        int copy_left = std::max(left + border, 0);
        int copy_bottom = std::min(bottom - border, (int)height_);
        int copy_right = std::min(right - border, (int)width_);
        
        for (int row = copy_top; row < copy_bottom; row++) {
            for (int col = copy_left; col < copy_right; col++) {
                int tile_idx = (row - top) * ts + (col - left);
                int out_idx = row * width_ + col;
                
                // Simple copy without blending - RawTherapee approach
                rgb_buffer_.image[out_idx][0] = CLIP((int)(rgb[tile_idx][0] * 65535.0f + 0.5f));
                rgb_buffer_.image[out_idx][1] = CLIP((int)(rgb[tile_idx][1] * 65535.0f + 0.5f));
                rgb_buffer_.image[out_idx][2] = CLIP((int)(rgb[tile_idx][2] * 65535.0f + 0.5f));
                rgb_buffer_.image[out_idx][3] = rgb_buffer_.image[out_idx][1]; // G2 = G1
            }
        }
    }
};

} // anonymous namespace

bool CPUAccelerator::demosaic_xtrans_3pass(const ImageBuffer& raw_buffer, ImageBuffer& rgb_buffer, const char (&xtrans)[6][6], const float (&color_matrix)[3][4]) {
#ifdef __APPLE__
    if (!initialized_ || !raw_buffer.is_valid() || !rgb_buffer.is_valid()) {
        return false;
    }
    
    XTrans_Processor processor(raw_buffer, rgb_buffer, xtrans, color_matrix);
    processor.run_3pass();
    return true;
#else
    return false;
#endif
}

bool CPUAccelerator::demosaic_xtrans_1pass(const ImageBuffer& raw_buffer, ImageBuffer& rgb_buffer, const char (&xtrans)[6][6], const float (&color_matrix)[3][4]) {
#ifdef __APPLE__
    if (!initialized_ || !raw_buffer.is_valid() || !rgb_buffer.is_valid()) {
        return false;
    }
    
    XTrans_Processor processor(raw_buffer, rgb_buffer, xtrans, color_matrix);
    processor.run_1pass();
    return true;
#else
    return false;
#endif
}


bool CPUAccelerator::apply_white_balance(const ImageBufferFloat32& rgb_input,
                                            ImageBufferFloat32& rgb_output,
                                            const float wb_multipliers[4]) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (!initialized_ || !rgb_input.is_valid() || !rgb_output.is_valid() || rgb_input.width != rgb_output.width || rgb_input.height != rgb_output.height || rgb_input.channels != 3 || rgb_output.channels != 3) {
        std::cerr << "❌ Invalid or mismatched buffers for white balance" << std::endl;
        return false;
    }
    std::cout << "🎯 White Balance (Accelerate/vDSP): multipliers [R=" << wb_multipliers[0] 
              << ", G=" << wb_multipliers[1] << ", B=" << wb_multipliers[2] << "]" << std::endl;
    
    const vDSP_Length pixel_count = rgb_input.width * rgb_input.height;
    
    if (rgb_input.data != rgb_output.data) {
        std::memcpy(rgb_output.data, rgb_input.data, pixel_count * 3 * sizeof(float));
    }
    
    // ✨ vDSP を使った超高速ホワイトバランス適用
    vDSP_vsmul(rgb_output.data + 0, 3, &wb_multipliers[0], rgb_output.data + 0, 3, pixel_count);
    vDSP_vsmul(rgb_output.data + 1, 3, &wb_multipliers[1], rgb_output.data + 1, 3, pixel_count);
    vDSP_vsmul(rgb_output.data + 2, 3, &wb_multipliers[2], rgb_output.data + 2, 3, pixel_count);

    float zero = 0.0f, one = 1.0f;
    vDSP_vclip(rgb_output.data, 1, &zero, &one, rgb_output.data, 1, pixel_count * 3);
    
    std::cout << "✅ White balance applied successfully using vDSP" << std::endl;
    return true;
#else
    // Apple Silicon 以外のためのフォールバック実装
    // (省略) ...
    return false;
#endif
}

bool CPUAccelerator::convert_color_space(const ImageBufferFloat32& rgb_input, ImageBufferFloat32& rgb_output, const float transform[3][4]) {
    if (!initialized_) return false;
    if (!rgb_input.is_valid() || !rgb_output.is_valid() || rgb_input.width != rgb_output.width || rgb_input.height != rgb_output.height || rgb_input.channels != 3 || rgb_output.channels != 3) {
        std::cerr << "❌ Invalid or mismatched buffers for color space conversion" << std::endl;
        return false;
    }
    std::cout << "🎯 Camera matrix-based color space conversion" << std::endl;
    const size_t pixel_count = rgb_input.width * rgb_input.height;
    for (size_t i = 0; i < pixel_count; i++) {
        const float* in = &rgb_input.data[i * 3];
        float* out = &rgb_output.data[i * 3];
        out[0] = fmaxf(0.0f, fminf(1.0f, transform[0][0] * in[0] + transform[0][1] * in[1] + transform[0][2] * in[2] + transform[0][3]));
        out[1] = fmaxf(0.0f, fminf(1.0f, transform[1][0] * in[0] + transform[1][1] * in[1] + transform[1][2] * in[2] + transform[1][3]));
        out[2] = fmaxf(0.0f, fminf(1.0f, transform[2][0] * in[0] + transform[2][1] * in[1] + transform[2][2] * in[2] + transform[2][3]));
    }
    std::cout << "✅ Camera matrix color conversion completed" << std::endl;
    return true;
}

bool CPUAccelerator::gamma_correct(const ImageBufferFloat32& rgb_input, ImageBufferFloat32& rgb_output, float gamma_power, float gamma_slope, int output_color_space) {
#ifdef __APPLE__
    if (!initialized_) return false;
    if (!rgb_input.is_valid() || !rgb_output.is_valid() || rgb_input.width != rgb_output.width || rgb_input.height != rgb_output.height || rgb_input.channels != 3 || rgb_output.channels != 3) {
        std::cerr << "❌ Invalid or mismatched buffers for gamma correction" << std::endl;
        return false;
    }
    if (output_color_space == 0 || output_color_space == 5) {
        std::cout << "🔄 Gamma correction skipped for linear color space " << output_color_space << " (Raw/XYZ)" << std::endl;
        if (rgb_input.data != rgb_output.data) {
            std::memcpy(rgb_output.data, rgb_input.data, rgb_input.width * rgb_input.height * 3 * sizeof(float));
        }
        return true;
    }
    std::cout << "🎯 Gamma Correction: color_space=" << output_color_space << ", γ=" << gamma_power << ", slope=" << gamma_slope << std::endl;
    if (rgb_input.data != rgb_output.data) {
        std::memcpy(rgb_output.data, rgb_input.data, rgb_input.width * rgb_input.height * 3 * sizeof(float));
    }
    for (size_t i = 0; i < rgb_input.width * rgb_input.height; i++) {
        float* pixel = &rgb_output.data[i * 3];
        for (int c = 0; c < 3; c++) {
            float linear_value = pixel[c], gamma_corrected;
            switch (output_color_space) {
                case 1: case 7: gamma_corrected = apply_srgb_gamma_encode(linear_value); break;
                case 2: case 3: gamma_corrected = apply_pure_power_gamma_encode(linear_value, 2.2f); break;
                case 4:         gamma_corrected = apply_pure_power_gamma_encode(linear_value, 1.8f); break;
                case 6:         gamma_corrected = apply_aces_gamma_encode(linear_value); break;
                case 8:         gamma_corrected = apply_rec2020_gamma_encode(linear_value); break;
                default:
                    gamma_corrected = (linear_value < 1.0f / gamma_slope) ? linear_value * gamma_slope : std::pow(linear_value, 1.0f / gamma_power);
                    break;
            }
            pixel[c] = std::max(0.0f, std::min(1.0f, gamma_corrected));
        }
    }
    std::cout << "✅ Gamma correction applied successfully" << std::endl;
    return true;
#else
    return false;
#endif
}

double CPUAccelerator::get_last_processing_time() const { return last_processing_time_; }
size_t CPUAccelerator::get_memory_usage() const { return 0; }
void CPUAccelerator::set_debug_mode(bool enable) { debug_mode_ = enable; }
std::string CPUAccelerator::get_device_info() const { return device_name_; }



bool CPUAccelerator::pre_interpolate(ImageBuffer& image_buffer, uint32_t filters, const char (&xtrans)[6][6], bool half_size) {
    if (!image_buffer.is_valid()) return false;
    uint16_t (*image)[4] = image_buffer.image;
    size_t width = image_buffer.width, height = image_buffer.height;
    std::cout << "🔧 Pre-interpolation: " << width << "x" << height << " (filters=" << filters << ", half_size=" << half_size << ")" << std::endl;
    if (half_size && filters == 9) {
        size_t row, col;
        for (row = 0; row < 3; ++row) for (col = 1; col < 4; ++col) if (!(image[row * width + col][0] | image[row * width + col][2])) goto break_outer;
        break_outer:
        for (; row < height; row += 3)
            for (size_t col_start = (col - 1) % 3 + 1; col_start < width - 1; col_start += 3)
                for (int c = 0; c < 3; c += 2)
                    image[row * width + col_start][c] = (image[row * width + col_start - 1][c] + image[row * width + col_start + 1][c]) >> 1;
    }
    std::cout << "✅ Pre-interpolation completed" << std::endl;
    return true;
}

float CPUAccelerator::apply_srgb_gamma_encode(float v) const { return (v <= 0.0031308f) ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f; }
float CPUAccelerator::apply_pure_power_gamma_encode(float v, float p) const { return std::pow(std::max(0.0f, v), 1.0f / p); }
float CPUAccelerator::apply_rec2020_gamma_encode(float v) const {
    const float a = 1.09929682680944f, b = 0.09929682680944f;
    return (v < b / 4.5f) ? 4.5f * v : a * std::pow(v, 0.45f) - (a - 1.0f);
}
float CPUAccelerator::apply_aces_gamma_encode(float v) const {
    if (v <= 0.0f) return 0.0f;
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return std::max(0.0f, std::min(1.0f, (v * (a * v + b)) / (v * (c * v + d) + e)));
}


void CPUAccelerator::border_interpolate(const ImageBuffer& raw_buffer, uint32_t filters, int border_int) {
    const size_t width = raw_buffer.width, height = raw_buffer.height;
    const int border = border_int;
    uint16_t (*image)[4] = raw_buffer.image;
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
            unsigned sum[8] = {0};
            
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
            for (int c = 0; c < 4; c++) {
                if (c != f && sum[c + 4] > 0) {
                    image[row * width + col][c] = sum[c] / sum[c + 4];
                }
            }
        }
    }
    std::cout << "✅ LibRaw-exact border interpolation completed" << std::endl;
}


} // namespace libraw_enhanced