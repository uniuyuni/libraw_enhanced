//
// shader_types.h
// C++ と Metal Shader で共有するデータ構造の定義
//
#ifndef shader_types_h
#define shader_types_h

#include <simd/simd.h>
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define UINT2 uint2
#else
#define UINT2 vector_uint2
#endif

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t border_width;
    float maximum_value;
    uint32_t filters;
    UINT2 grid_size;
    float clip_pt;      // Dynamic clipping threshold
    float clip_pt8;     // Dynamic highlight clipping threshold
} BayerParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t border_width;
    float maximum_value;
    char xtrans[6][6];
    uint32_t use_cielab;     // 0=YPbPr, 1=CIELab
} XTransParams;

struct s_minmaxgreen {
    float min;
    float max;
};
typedef struct {
    float rgb[8][114][114][3];  // 8方向のRGBデータ
    float lab[3][106][106];     // Labデータ (ts-8)
    float drv[8][104][104];     // 微分データ (ts-10)
    uint8_t homo[8][114][114];  // 均質性マップ
    uint8_t homosum[8][114][114]; // 均質性和
    uint8_t homosummax[114][114]; // 最大均質性
    s_minmaxgreen greenminmax[114][57]; // greenminmaxデータ (ts x ts/2)
} XTransTileData;

typedef struct {
    uint32_t width;
    uint32_t height;
    float multipliers[4];    // RGBG white balance multipliers
} WhiteBalanceParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    float transform[3][4];   // 3x4 color transformation matrix
} ColorSpaceParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    float gamma_power;       // Gamma power value
    float gamma_slope;       // Gamma slope value
    uint32_t output_color_space; // Color space identifier
} GammaParams;

#endif /* shader_types_h */
