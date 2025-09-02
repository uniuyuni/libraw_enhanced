#include "shader_types.h"
#include "shader_common.h"
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// CLIPマクロの定義
#define CLIP(x) (x)

// ヘルパー関数の定義
inline int fcol_xtrans(int row, int col, thread char (&xtrans)[6][6]) {
    return xtrans[(row + 6) % 6][(col + 6) % 6];
}

inline int isgreen(int row, int col, thread char (&xtrans)[6][6]) {
    return (xtrans[(row) % 3][(col) % 3] & 1);
}

inline float raw_buffer(int row, int col, const device uint16_t* raw_data, constant XTransParams& params) {
    int pos = row * params.width + col;
    //if (pos < 0 || pos >= (int)(params.width * params.height)) return 0.0f;
    return (float)raw_data[pos] / params.maximum_value;
}

inline float raw_buffer_hex(int row, int col, short hex, const device uint16_t* raw_data, constant XTransParams& params) {
    int pos = row * params.width + col + hex;
    //if (pos < 0 || pos >= (int)(params.width * params.height)) return 0.0f;
    return (float)raw_data[pos] / params.maximum_value;
}

inline void vconvertrgbrgbrgbrgb2rrrrggggbbbb(const device float *src,
                                                    thread float4 &rv,
                                                    thread float4 &gv,
                                                    thread float4 &bv) {
    // RGB tripletsから4つのピクセル分のR、G、B成分を分離
    rv = float4(src[0], src[3], src[6], src[9]);
    gv = float4(src[1], src[4], src[7], src[10]);
    bv = float4(src[2], src[5], src[8], src[11]);
}

#define fabsf fabs

// X-Trans 3-pass demosaic kernel
kernel void demosaic_xtrans_3pass(
    const device uint16_t* raw_data [[buffer(0)]],
    device float* rgb_data [[buffer(1)]],
    const device float* cbrt_lut [[buffer(2)]],
    const device short* allhex_data [[buffer(3)]],
    const device uint2& sg_coords [[buffer(4)]],
    const device float* xyz_cam [[buffer(5)]],
    device XTrans3passTile* tile_data [[buffer(6)]],
    constant XTransParams& params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgsize [[threads_per_threadgroup]])
{
    const uint width = params.width;
    const uint height = params.height;
    const bool use_cielab = params.use_cielab != 0;
    
    const int ts = XTRANS_3PASS_TS;
    const int passes = XTRANS_3PASS_PASSES;
    const int ndir = XTRANS_3PASS_nDIRS;
    const int sgrow = sg_coords.x;
    const int sgcol = sg_coords.y;
    
    const short dir[4] = { 1, ts, ts + 1, ts - 1 };

    // xtransデータをレジスタへロード
    thread char xtrans[6][6];
    for (int i = 0; i < 6; i++) {
        *(thread char3*)(&xtrans[i][0]) = *(constant char3*)(&params.xtrans[i][0]);
        *(thread char3*)(&xtrans[i][3]) = *(constant char3*)(&params.xtrans[i][3]);
    }

    // RightShift配列の計算
    thread int RightShift[3];
    for(int row = 0; row < 3; row++) {
        int greencount = 0;
        for(int col = 0; col < 3; col++) {
            greencount += isgreen(row, col, xtrans);
        }
        RightShift[row] = (greencount == 2);
    }
    
    // タイル座標計算
    const int tile_x = tgid.x;
    const int tile_y = tgid.y;
    const uint tile_w = (width + (ts - 16) - 1) / (ts - 16);
    int top = tile_y * (ts - 16) + 3;
    int left = tile_x * (ts - 16) + 3;
    int mrow = min(top + ts, (int)height - 3);
    int mcol = min(left + ts, (int)width - 3);

    // 共有メモリ初期化
    const device short (*allhex)[3][3][8] = (const device short (*)[3][3][8])allhex_data;
    device XTrans3passTile* shared_data = tile_data + (tile_y * tile_w + tile_x);
    device float (*rgb)[ts][ts][3] = shared_data->rgb;
    device float (*lab)[ts - 8][ts - 8] = shared_data->lab;
    device float (*drv)[ts - 10][ts - 10] = shared_data->drv;
    device uint8_t (*homo)[ts][ts] = shared_data->homo;
    device uint8_t (*homosum)[ts][ts] = shared_data->homosum;
    device uint8_t (*homosummax)[ts] = shared_data->homosummax;
    device s_minmaxgreen (*greenminmaxtile)[ts / 2] = shared_data->greenminmax;
    // threadgroup_barrier(mem_flags::mem_device);

    // Set greenmin and greenmax to the minimum and maximum allowed values:
    for (int row = top; row < mrow; row++) {
        // find first non-green pixel
        int leftstart = left;

        for(; leftstart < mcol; leftstart++)
            if(!isgreen(row, leftstart, xtrans)) {
                break;
            }

        int coloffset = (RightShift[row % 3] == 1 ? 3 : 1 + (fcol_xtrans(row, leftstart + 1, xtrans) & 1));

        float minval = MAXFLOAT;
        float maxval = 0.f;

        if(coloffset == 3) {
            const device short *hex = allhex[0][row % 3][leftstart % 3];

            for (int col = leftstart; col < mcol; col += coloffset) {
                minval = MAXFLOAT;
                maxval = 0.f;

                for(int c = 0; c < 6; c++) {
                    float val = raw_buffer_hex(row, col, hex[c], raw_data, params);

                    minval = minval < val ? minval : val;
                    maxval = maxval > val ? maxval : val;
                }

                greenminmaxtile[row - top][(col - left) >> 1].min = minval;
                greenminmaxtile[row - top][(col - left) >> 1].max = maxval;
            }
        } else {
            int col = leftstart;

            if(coloffset == 2) {
                minval = MAXFLOAT;
                maxval = 0.f;
                const device short *hex = allhex[0][row % 3][col % 3];

                for(int c = 0; c < 6; c++) {
                    float val = raw_buffer_hex(row, col, hex[c], raw_data, params);

                    minval = minval < val ? minval : val;
                    maxval = maxval > val ? maxval : val;
                }

                greenminmaxtile[row - top][(col - left) >> 1].min = minval;
                greenminmaxtile[row - top][(col - left) >> 1].max = maxval;
                col += 2;
            }

            const device short *hex = allhex[0][row % 3][col % 3];

            for (; col < mcol - 1; col += 3) {
                minval = MAXFLOAT;
                maxval = 0.f;

                for(int c = 0; c < 6; c++) {
                    float val = raw_buffer_hex(row, col, hex[c], raw_data, params);

                    minval = minval < val ? minval : val;
                    maxval = maxval > val ? maxval : val;
                }

                greenminmaxtile[row - top][(col - left) >> 1].min = minval;
                greenminmaxtile[row - top][(col - left) >> 1].max = maxval;
                greenminmaxtile[row - top][(col + 1 - left) >> 1].min = minval;
                greenminmaxtile[row - top][(col + 1 - left) >> 1].max = maxval;
            }

            if(col < mcol) {
                minval = MAXFLOAT;
                maxval = 0.f;

                for(int c = 0; c < 6; c++) {
                    float val = raw_buffer_hex(row, col, hex[c], raw_data, params);

                    minval = minval < val ? minval : val;
                    maxval = maxval > val ? maxval : val;
                }

                greenminmaxtile[row - top][(col - left) >> 1].min = minval;
                greenminmaxtile[row - top][(col - left) >> 1].max = maxval;
            }
        }
    }
    // threadgroup_barrier(mem_flags::mem_device);

    //memset(rgb, 0, ts * ts * 3 * sizeof(float));
    device packed_float4* rgb_p = (device packed_float4*)rgb;
    for (uint i = 0; i < ts * ts * 3 / 4; i++) {
        rgb_p[i] = 0.0;
    }
    // threadgroup_barrier(mem_flags::mem_device);

    for (int row = top; row < mrow; row++) {
        for (int col = left; col < mcol; col++) {
            rgb[0][row - top][col - left][fcol_xtrans(row, col, xtrans)] = raw_buffer(row, col, raw_data, params);
        }
    }
    // threadgroup_barrier(mem_flags::mem_device);

    for(int c = 0; c < 3; c++) {
        //memcpy (rgb[c + 1], rgb[0], sizeof * rgb);
        device packed_float4 *rgb_p = (device packed_float4*)&rgb[c + 1];
        device packed_float4 *rgb_s = (device packed_float4*)&rgb[0];
        for (uint i = 0; i < sizeof(*rgb) / sizeof(packed_float4); ++i) {
            rgb_p[i] = rgb_s[i];
        }
    }
    // threadgroup_barrier(mem_flags::mem_device);

    // Interpolate green horizontally, vertically, and along both diagonals:
    // std::cout << "[DEBUG] Interpolate green horizontally, vertically, and along both diagonals:" << std::endl;
    for (int row = top; row < mrow; row++) {
        // find first non-green pixel
        int leftstart = left;

        for(; leftstart < mcol; leftstart++)
            if(!isgreen(row, leftstart, xtrans)) {
                break;
            }

        int coloffset = (RightShift[row % 3] == 1 ? 3 : 1 + (fcol_xtrans(row, leftstart + 1, xtrans) & 1));

        if(coloffset == 3) {
            const device short *hex = allhex[0][row % 3][leftstart % 3];

            for (int col = leftstart; col < mcol; col += coloffset) {
                float color[4];
                color[0] = 0.6796875f * (raw_buffer_hex(row, col, hex[1], raw_data, params) + raw_buffer_hex(row, col, hex[0], raw_data, params)) -
                            0.1796875f * (raw_buffer_hex(row, col, 2 * hex[1], raw_data, params) + raw_buffer_hex(row, col, 2 * hex[0], raw_data, params));
                color[1] = 0.87109375f * raw_buffer_hex(row, col, hex[3], raw_data, params) + raw_buffer_hex(row, col, hex[2], raw_data, params) * 0.12890625f +
                            0.359375f * (raw_buffer_hex(row, col, 0, raw_data, params) - raw_buffer_hex(row, col, -hex[2], raw_data, params));

                for(int c = 0; c < 2; c++)
                    color[2 + c] = 0.640625f * raw_buffer_hex(row, col, hex[4 + c], raw_data, params) + 0.359375f * raw_buffer_hex(row, col, -2 * hex[4 + c], raw_data, params) + 0.12890625f *
                                    (2.f * raw_buffer_hex(row, col, 0, raw_data, params) - raw_buffer_hex(row, col, 3 * hex[4 + c], raw_data, params) - raw_buffer_hex(row, col, -3 * hex[4 + c], raw_data, params));

                for(int c = 0; c < 4; c++) {
                    rgb[c][row - top][col - left][1] = LIM(color[c], greenminmaxtile[row - top][(col - left) >> 1].min, greenminmaxtile[row - top][(col - left) >> 1].max);
                }
            }
        } else {
            const device short *hexmod[2] = {
                allhex[0][row % 3][leftstart % 3],
                allhex[0][row % 3][(leftstart + coloffset) % 3],
            };

            for (int col = leftstart, hexindex = 0; col < mcol; col += coloffset, coloffset ^= 3, hexindex ^= 1) {
                const device short *hex = hexmod[hexindex];
                float color[4];
                color[0] = 0.6796875f * (raw_buffer_hex(row, col, hex[1], raw_data, params) + raw_buffer_hex(row, col, hex[0], raw_data, params)) -
                            0.1796875f * (raw_buffer_hex(row, col, 2 * hex[1], raw_data, params) + raw_buffer_hex(row, col, 2 * hex[0], raw_data, params));
                color[1] = 0.87109375f *  raw_buffer_hex(row, col, hex[3], raw_data, params) + raw_buffer_hex(row, col, hex[2], raw_data, params) * 0.12890625f +
                            0.359375f * (raw_buffer_hex(row, col, 0, raw_data, params) - raw_buffer_hex(row, col, -hex[2], raw_data, params));

                for(int c = 0; c < 2; c++) {
                    color[2 + c] = 0.640625f * raw_buffer_hex(row, col, hex[4 + c], raw_data, params) + 0.359375f * raw_buffer_hex(row, col, -2 * hex[4 + c], raw_data, params) + 0.12890625f *
                                    (2.f * raw_buffer_hex(row, col, 0, raw_data, params) - raw_buffer_hex(row, col, 3 * hex[4 + c], raw_data, params) - raw_buffer_hex(row, col, -3 * hex[4 + c], raw_data, params));
                }

                for(int c = 0; c < 4; c++) {
                    rgb[c ^ 1][row - top][col - left][1] = LIM(color[c], greenminmaxtile[row - top][(col - left) >> 1].min, greenminmaxtile[row - top][(col - left) >> 1].max);
                }
            }
        }
    }
    // threadgroup_barrier(mem_flags::mem_device);

    for (int pass = 0; pass < passes; pass++) {
        if (pass == 1) {
            //memcpy (rgb += 4, rgb_buffer_tile.data(), 4 * sizeof * rgb);
            rgb += 4;
            device packed_float4* rgb_p = (device packed_float4*)rgb;
            device packed_float4* rgb_s = (device packed_float4*)shared_data->rgb;
            for (uint i = 0; i < 4 * sizeof(*rgb) / sizeof(packed_float4); ++i) {
                rgb_p[i] = rgb_s[i];
            }
        }
        // threadgroup_barrier(mem_flags::mem_device);

        // Recalculate green from interpolated values of closer pixels:
        //std::cout << "[DEBUG] Recalculate green from interpolated values of closer pixels: " << std::endl;
        if (pass) {
            for (int row = top + 2; row < mrow - 2; row++) {
                int leftstart = left + 2;

                for(; leftstart < mcol - 2; leftstart++)
                    if(!isgreen(row, leftstart, xtrans)) {
                        break;
                    }

                int coloffset = (RightShift[row % 3] == 1 ? 3 : 1 + (fcol_xtrans(row, leftstart + 1, xtrans) & 1));

                if(coloffset == 3) {
                    int f = fcol_xtrans(row, leftstart, xtrans);
                    const device short *hex = allhex[1][row % 3][leftstart % 3];

                    for (int col = leftstart; col < mcol - 2; col += coloffset, f ^= 2) {
                        for (int d = 3; d < 6; d++) {
                            device float (*rix)[3] = &rgb[(d - 2)][row - top][col - left];
                            float val = 0.33333333f * (rix[-2 * hex[d]][1] + 2 * (rix[hex[d]][1] - rix[hex[d]][f])
                                                        - rix[-2 * hex[d]][f]) + rix[0][f];
                            rix[0][1] = LIM(val, greenminmaxtile[row - top][(col - left) >> 1].min, greenminmaxtile[row - top][(col - left) >> 1].max);
                        }
                    }
                } else {
                    int f = fcol_xtrans(row, leftstart, xtrans);
                    const device short *hexmod[2] = {
                        allhex[1][row % 3][leftstart % 3],
                        allhex[1][row % 3][(leftstart + coloffset) % 3],
                    };

                    for (int col = leftstart, hexindex = 0; col < mcol - 2; col += coloffset, coloffset ^= 3, f = f ^ (coloffset & 2), hexindex ^= 1 ) {
                        const device short *hex = hexmod[hexindex];

                        for (int d = 3; d < 6; d++) {
                            device float (*rix)[3] = &rgb[(d - 2) ^ 1][row - top][col - left];
                            float val = 0.33333333f * (rix[-2 * hex[d]][1] + 2 * (rix[hex[d]][1] - rix[hex[d]][f])
                                                        - rix[-2 * hex[d]][f]) + rix[0][f];
                            rix[0][1] = LIM(val, greenminmaxtile[row - top][(col - left) >> 1].min, greenminmaxtile[row - top][(col - left) >> 1].max);
                        }
                    }
                }
            }
        }
        // threadgroup_barrier(mem_flags::mem_device);

        // Interpolate red and blue values for solitary green pixels:
        //std::cout << "[DEBUG] Interpolate red and blue values for solitary green pixels:" << std::endl;
        int sgstartcol = (left - sgcol + 4) / 3 * 3 + sgcol;
        thread float color[3][6];

        for (int row = (top - sgrow + 4) / 3 * 3 + sgrow; row < mrow - 2; row += 3) {
            for (int col = sgstartcol, h = fcol_xtrans(row, col + 1, xtrans); col < mcol - 2; col += 3, h ^= 2) {
                device float (*rix)[3] = &rgb[0][row - top][col - left];
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
        // threadgroup_barrier(mem_flags::mem_device);

        // Interpolate red for blue pixels and vice versa:
        //std::cout << "[DEBUG] Interpolate red for blue pixels and vice versa: " << std::endl;
        for (int row = top + 3; row < mrow - 3; row++) {
            int leftstart = left + 3;

            for(; leftstart < mcol - 1; leftstart++)
                if(!isgreen(row, leftstart, xtrans)) {
                    break;
                }

            int coloffset = (RightShift[row % 3] == 1 ? 3 : 1);
            int c = ((row - sgrow) % 3) ? ts : 1;
            int h = 3 * (c ^ ts ^ 1);

            if(coloffset == 3) {
                int f = 2 - fcol_xtrans(row, leftstart, xtrans);

                for (int col = leftstart; col < mcol - 3; col += coloffset, f ^= 2) {
                    device float (*rix)[3] = &rgb[0][row - top][col - left];

                    for (int d = 0; d < 4; d++, rix += ts * ts) {
                        int i = d > 1 || ((d ^ c) & 1) ||
                                ((fabsf(rix[0][1] - rix[c][1]) + fabsf(rix[0][1] - rix[-c][1])) < 2.f * (fabsf(rix[0][1] - rix[h][1]) + fabsf(rix[0][1] - rix[-h][1]))) ? c : h;

                        rix[0][f] = CLIP(rix[0][1] + 0.5f * (rix[i][f] + rix[-i][f] - rix[i][1] - rix[-i][1]));
                    }
                }
            } else {
                coloffset = fcol_xtrans(row, leftstart + 1, xtrans) == 1 ? 2 : 1;
                int f = 2 - fcol_xtrans(row, leftstart, xtrans);

                for (int col = leftstart; col < mcol - 3; col += coloffset, coloffset ^= 3, f = f ^ (coloffset & 2) ) {
                    device float (*rix)[3] = &rgb[0][row - top][col - left];

                    for (int d = 0; d < 4; d++, rix += ts * ts) {
                        int i = d > 1 || ((d ^ c) & 1) ||
                            ((fabsf(rix[0][1] - rix[c][1]) + fabsf(rix[0][1] - rix[-c][1])) < 2.f * (fabsf(rix[0][1] - rix[h][1]) + fabsf(rix[0][1] - rix[-h][1]))) ? c : h;

                        rix[0][f] = CLIP(rix[0][1] + 0.5f * (rix[i][f] + rix[-i][f] - rix[i][1] - rix[-i][1]));
                    }
                }
            }
        }
        // threadgroup_barrier(mem_flags::mem_device);

        // Fill in red and blue for 2x2 blocks of green:
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

        int coloffsetstart = 2 - (fcol_xtrans(topstart, leftstart + 1, xtrans) & 1);

        for (int row = topstart; row < mrow - 2; row++) {
            if ((row - sgrow) % 3) {
                device short *hexmod[2];
                hexmod[0] = (device short*)allhex[1][row % 3][leftstart % 3];
                hexmod[1] = (device short*)allhex[1][row % 3][(leftstart + coloffsetstart) % 3];

                for (int col = leftstart, coloffset = coloffsetstart, hexindex = 0; col < mcol - 2; col += coloffset, coloffset ^= 3, hexindex ^= 1) {
                    device float (*rix)[3] = &rgb[0][row - top][col - left];
                    device short *hex = hexmod[hexindex];

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
    // threadgroup_barrier(mem_flags::mem_device);

// end of multipass part

    rgb = shared_data->rgb;
    mrow -= top;
    mcol -= left;    

    if(false) {
        // Convert to CIELab and differentiate in all directions:
        //std::cout << "[DEBUG] Convert to CIELab and differentiate in all directions:" << std::endl;
        // Original dcraw algorithm uses CIELab as perceptual space
        // (presumably coming from original AHD) and converts taking
        // camera matrix into account.  We use this in RT.

    } else {
        // For 1-pass demosaic we use YPbPr which requires much
        // less code and is nearly indistinguishable. It assumes the
        // camera RGB is roughly linear.
        //std::cout << "[DEBUG] For 1-pass demosaic we use YPbPr which requires much" << std::endl;
#if 1
        float4 zd2627v = float4(0.2627f);
        float4 zd6780v = float4(0.6780f);
        float4 zd0593v = float4(0.0593f);
        float4 zd56433v = float4(0.56433f);
        float4 zd67815v = float4(0.67815f);
#endif
        for (int d = 0; d < ndir; d++) {
            device float (*yuv)[ts - 8][ts - 8] = lab; // we use the lab buffer, which has the same dimensions

            for (int row = 4; row < mrow - 4; row++) {
                int col = 4;            
#if 1
                for (; col < mcol - 7; col += 4) {
                    // 4ピクセル分のRGBデータを読み込み
                    thread float4 redv, greenv, bluev;
                    vconvertrgbrgbrgbrgb2rrrrggggbbbb(rgb[d][row][col], redv, greenv, bluev);

                    // BT.2020変換
                    float4 yv = zd2627v * redv + zd6780v * greenv + zd0593v * bluev;
                    float4 u_comp = (bluev - yv) * zd56433v;
                    float4 v_comp = (redv - yv) * zd67815v;

                    // 結果の書き込み
                    *(device float4*)(&yuv[0][row - 4][col - 4]) = yv;
                    *(device float4*)(&yuv[1][row - 4][col - 4]) = u_comp;
                    *(device float4*)(&yuv[2][row - 4][col - 4]) = v_comp;
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
                    device float *y = &yuv[0][row - 4][col - 4];
                    device float *u = &yuv[1][row - 4][col - 4];
                    device float *v = &yuv[2][row - 4][col - 4];
                    drv[d][row - 5][col - 5] = SQR(2 * y[0] - y[f] - y[-f])
                                                + SQR(2 * u[0] - u[f] - u[-f])
                                                + SQR(2 * v[0] - v[f] - v[-f]);
                }
            }
        }
    }
    // threadgroup_barrier(mem_flags::mem_device);

    // Build homogeneity maps from the derivatives:
    //std::cout << "[DEBUG] Build homogeneity maps from the derivatives:" << std::endl;
    for (int row = 6; row < mrow - 6; row++) {
        int col = 6;
#if 1
        for (; col < mcol - 9; col += 4) {
            float4 tr1v = min(*(device const packed_float4*)(&drv[0][row - 5][col - 5]), *(device const packed_float4*)(&drv[1][row - 5][col - 5]));
            float4 tr2v = min(*(device const packed_float4*)(&drv[2][row - 5][col - 5]), *(device const packed_float4*)(&drv[3][row - 5][col - 5]));

            if(ndir > 4) {
                float4 tr3v = min(*(device const packed_float4*)(&drv[4][row - 5][col - 5]), *(device const packed_float4*)(&drv[5][row - 5][col - 5]));
                float4 tr4v = min(*(device const packed_float4*)(&drv[6][row - 5][col - 5]), *(device const packed_float4*)(&drv[7][row - 5][col - 5]));
                tr1v = min(tr1v, tr3v);
                tr1v = min(tr1v, tr4v);
            }

            tr1v = min(tr1v, tr2v);
            tr1v = tr1v * float4(8.f);

            for (int d = 0; d < ndir; d++) {
                //uint8_t tempstore[16];
                float4 tempv = float4(0.f);

                for (int v = -1; v <= 1; v++) {
                    for (int h = -1; h <= 1; h++) {
                        float4 drv_val = *(device const packed_float4*)(&drv[d][row + v - 5][col + h - 5]);
                        tempv += float4(drv_val <= tr1v);
                    }
                }

                *(device packed_uchar4*)(&homo[d][row][col]) = uchar4(tempv.x, tempv.y, tempv.z, tempv.w);
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
    // threadgroup_barrier(mem_flags::mem_device);

    if (height - top < ts + 4) {
        mrow = height - top + 2;
    }

    if (width - left < ts + 4) {
        mcol = width - left + 2;
    }

    // Build 5x5 sum of homogeneity maps
    //std::cout << "[DEBUG] Build 5x5 sum of homogeneity maps" << std::endl;
    for(int d = 0; d < ndir; d++) {
        for (int row = min(top, 8); row < mrow - 8; row++) {
            int col = min(left, 8);
#if 1
            int endcol = row < mrow - 9 ? mcol - 8 : mcol - 23;

            // crunching 16 values at once is faster than summing up column sums
            for (; col < endcol; col += 4) {
                uchar4 v5sum = uchar4(0);
                
                // 5x5畳み込み
                for (int v = -2; v <= 2; v++) {
                    for (int h = -2; h <= 2; h++) {
                        v5sum += *(device packed_uchar4*)(&homo[d][row + v][col + h]);
                        //v5sum = min(v5sum, uchar4(255));
                    }
                }
                
                // 結果保存
                *(device packed_uchar4*)(&homosum[d][row][col]) = v5sum;
            }
#endif
            if(col < mcol - 8) {
                int v5sum[5] = {0};

                for(int v = -2; v <= 2; v++)
                    for(int h = -2; h <= 2; h++) {
                        v5sum[2 + h] += homo[d][row + v][col + h];
                        //v5sum[2 + h] = min(v5sum[2 + h], 255);
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
    // threadgroup_barrier(mem_flags::mem_device);

    // calculate maximum of homogeneity maps per pixel. Vectorized calculation is a tiny bit faster than on the fly calculation in next step
    //std::cout << "[DEBUG] calculate maximum of homogeneity maps per pixel. Vectorized calculation is a tiny bit faster than on the fly calculation in next step" << std::endl;
    for (int row = min(top, 8); row < mrow - 8; row++) {
        int col = min(left, 8);
#if 1
        // 4要素のSIMD処理
        int endcol = row < mrow - 9 ? mcol - 8 : mcol - 23;

        for (; col < endcol; col += 4) {
            uchar4 maxval1 = max(uchar4(*(device const packed_uchar4*)(&homosum[0][row][col])),
                                 uchar4(*(device const packed_uchar4*)(&homosum[1][row][col])));
            uchar4 maxval2 = max(uchar4(*(device const packed_uchar4*)(&homosum[2][row][col])),
                                 uchar4(*(device const packed_uchar4*)(&homosum[3][row][col])));

            if (ndir > 4) {
                uchar4 maxval3 = max(uchar4(*(device const packed_uchar4*)(&homosum[4][row][col])),
                                     uchar4(*(device const packed_uchar4*)(&homosum[5][row][col])));
                uchar4 maxval4 = max(uchar4(*(device const packed_uchar4*)(&homosum[6][row][col])),
                                     uchar4(*(device const packed_uchar4*)(&homosum[7][row][col])));
                maxval1 = max(maxval1, maxval3);
                maxval1 = max(maxval1, maxval4);
            }            
            maxval1 = max(maxval1, maxval2);
            
            uchar4 subv = (maxval1 >> 3) & uchar4(0x1F);
            *(device packed_uchar4*)(&homosummax[row][col]) = select(uchar4(0), maxval1 - subv, maxval1 > subv);
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
    // threadgroup_barrier(mem_flags::mem_device);

    /* Average the most homogeneous pixels for the final result: */
    //std::cout << "[DEBUG] Average the most homogeneous pixels for the final result:" << std::endl;
    uint8_t hm[8] = {0,};
    for (int row = min(top, 8); row < mrow - 8; row++) {
        for (int col = min(left, 8); col < mcol - 8; col++) {

            for (int d = 0; d < 4; d++) {
                hm[d] = homosum[d][row][col];
            }

            for (int d = 4; d < ndir; d++) {
                hm[d] = homosum[d][row][col];

                if (hm[d - 4] < hm[d]) {
                    hm[d - 4] = 0;
                } else
                if (hm[d - 4] > hm[d]) {
                    hm[d] = 0;
                }
            }

            float3 avg = float3(0.f);
            float count = 0.f;

            uint8_t maxval = homosummax[row][col];

            for (int d = 0; d < ndir; d++) {
                if (hm[d] >= maxval) {
                    avg += *(device packed_float3*)(&rgb[d][row][col]);
                    count += 1.f;
                }
            }
            int idx = ((row + top) * width + (col + left)) * 3;
            *(device packed_float3*)(&rgb_data[idx]) =  max(0.f, avg / count);
        }
    }
}