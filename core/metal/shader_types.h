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
} DemosaicBayerParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t border_width;
    float maximum_value;
    char xtrans[6][6];
    uint32_t use_cielab;     // 0=YPbPr, 1=CIELab
} DemosaicXTransParams;

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
} ApplyWhiteBalanceParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    FLOAT4 transform[3];
    //float transform[3][4];   // 3x4 color transformation matrix
} ConvertColorSpaceParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    float gamma_power;       // Gamma power value
    float gamma_slope;       // Gamma slope value
    uint32_t output_color_space; // Color space identifier
} GammaCorrectParams;

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
    uint32_t use_mask;   // 1: read soft mask buffer; 0: derive a 0/1 weight from threshold
} EnhanceMicroContrastParams;

// Axial-CA: image preparation (normalise by max(G), split planes, compute G²,
// R·G, B·G in one pass).
typedef struct {
    uint32_t width;
    uint32_t height;
    float    inv_norm;     // 1.0 / max(G)
} AxialCaPrepareParams;

// Axial-CA: per-pixel (a, b) regression coefficients from box-mean inputs.
typedef struct {
    uint32_t width;
    uint32_t height;
    float    epsilon;
} AxialCaAbParams;

// Axial-CA: final blend of original and filtered R/B by edge-gated strength.
typedef struct {
    uint32_t width;
    uint32_t height;
    float    strength;
    float    norm_ref;     // = max(G); used to de-normalise the working planes
    float    edge_lo;      // smoothstep lower bound on smoothed |grad G|
    float    edge_hi;      // smoothstep upper bound
} AxialCaBlendParams;

// Detail-preserving tone-map: per-pixel kernel parameters.
typedef struct {
    uint32_t width;
    uint32_t height;
    float    eps;        // guided-filter regularisation
    float    edge0;      // smoothstep lower bound (gate on `base`)
    float    edge1;      // smoothstep upper bound
} DetailTonemapParams;

// Lateral-CA: per-level Lucas-Kanade estimation kernel parameters.
typedef struct {
    uint32_t W;             // current level plane dimensions
    uint32_t H;
    uint32_t map_w;         // shift-map dimensions (same at every level)
    uint32_t map_h;
    uint32_t max_iterations;
    float    max_shift;     // per-level pixel clamp
    float    cell_w_f;      // W / map_w (float; cells anchored at level)
    float    cell_h_f;      // H / map_h
} LateralCaLkParams;

// Lateral-CA: 2x downsample of a planar R32Float texture.
typedef struct {
    uint32_t W_in;
    uint32_t H_in;
    uint32_t W_out;
    uint32_t H_out;
} LateralCaDownsampleParams;

// Lateral-CA: final per-pixel apply of bilinear-sampled R/B shifts.
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t map_w;
    uint32_t map_h;
    float    cell_size;     // full-resolution cell side (level-0 pixels)
    float    clamp_shift;   // absolute clamp on per-pixel sampled shift
} LateralCaApplyParams;

// Defringe: shared shape/control parameters used by every defringe kernel.
typedef struct {
    uint32_t width;
    uint32_t height;
    float    edge_threshold;
    float    chroma_threshold;
    float    strength;
    float    green_strength_scale;
    float    sensitivity;          // = clamp(sqrt(max(strength, 1)), 1, 2.5)
    float    inv_sensitivity;
    float    mask_cutoff;          // 0.02 / sensitivity
    float    max_guide;            // populated by the CPU pre-pass (max(G))
    float    scene_highlight_gate; // smoothstep(0.55, 0.75, max_guide)
    uint32_t enable_green_defringe; // 0/1
    float    ratio_eps;            // log() numerical regularisation
} DefringeParams;

#endif /* shader_types_h */
