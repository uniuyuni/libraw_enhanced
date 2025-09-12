//
// shader_types.h
// C++ と Metal Shader で共有するデータ構造の定義
//
#ifndef shader_types_h
#define shader_types_h

#include <simd/simd.h>
#ifdef __METAL_VERSION__
#include <metal_stdlib>

// Metal シェーダー最適化設定
#ifdef METAL_OPTIMIZED
    // 高速数学演算プラグマ
    #pragma clang fp reassociate(on)
    #pragma clang fp contract(fast)
    using namespace metal;
    // metal::fastは曖昧性を引き起こすため個別に明示
    // using namespace metal::fast;
#endif

#define UINT2 uint2
#define FLOAT4 float4
#else
#define UINT2 vector_uint2
#define FLOAT4 vector_float4
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

#define XTRANS_3PASS_TS 114
#define XTRANS_3PASS_PASSES 3
#define XTRANS_3PASS_nDIRS (4 << (XTRANS_3PASS_PASSES > 1)) 
struct s_minmaxgreen {
    float min;
    float max;
};
typedef struct {
    float rgb[XTRANS_3PASS_nDIRS][XTRANS_3PASS_TS][XTRANS_3PASS_TS][3];  // 8方向のRGBデータ
    float lab[3][XTRANS_3PASS_TS-8][XTRANS_3PASS_TS-8];     // Labデータ (ts-8)
    float drv[XTRANS_3PASS_nDIRS][XTRANS_3PASS_TS-10][XTRANS_3PASS_TS-10];     // 微分データ (ts-10)
    uint8_t homo[XTRANS_3PASS_nDIRS][XTRANS_3PASS_TS][XTRANS_3PASS_TS];  // 均質性マップ
    uint8_t homosum[XTRANS_3PASS_nDIRS][XTRANS_3PASS_TS][XTRANS_3PASS_TS]; // 均質性和
    uint8_t homosummax[XTRANS_3PASS_TS][XTRANS_3PASS_TS]; // 最大均質性
    s_minmaxgreen greenminmax[XTRANS_3PASS_TS][XTRANS_3PASS_TS/2]; // greenminmaxデータ (ts x ts/2)
} XTrans3passTile;

typedef struct {
    uint32_t width;
    uint32_t height;
    float multipliers[4];    // RGBG white balance multipliers
    uint32_t filters;
    char xtrans[6][6];
} WhiteBalanceParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    FLOAT4 transform[3];
    //float transform[3][4];   // 3x4 color transformation matrix
} ColorSpaceParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    float gamma_power;       // Gamma power value
    float gamma_slope;       // Gamma slope value
    uint32_t output_color_space; // Color space identifier
} GammaParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    float after_scale;
} ToneMappingParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    float threshold;
    float strength;
    float target_contrast;
    float max_local_std; // metalが書き込むワーク 0.f 初期化
} EnhanceMicroContrastParams;

#endif /* shader_types_h */
