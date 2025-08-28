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
} XTransParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t filters;
    uint32_t border;
    uint32_t maximum_value;
} BorderParams;

typedef struct  {
    uint32_t width;
    uint32_t height;
    uint32_t filters;
    uint32_t nr_width;
    uint32_t nr_height;
    uint32_t nr_margin;
    matrix_float3x3 yuv_cam;
    uint32_t channel_maximum[3];
    uint32_t channel_minimum[3];
    uint16_t maximum_value;
} AAHDParams;


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
