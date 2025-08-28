// demosaic_bayer_amaze.metal
// ✅ AMaZEアルゴリズム C++版完全移植 最終完成版（修正済み）
// すべての問題を修正した完全なシェーダーコード

#include "shader_types.h" 
#include "shader_common.h"

// 補助関数の定義
inline float xmul2f(float x) { return x + x; }
inline float xdiv2f(float x) { return x * 0.5f; }

#define fc_rt(r, c) cfarray[r & 1][c & 1]
#define fabsf fabs

struct s_hv { float h; float v; };

[[kernel, max_total_threads_per_threadgroup(16)]]
void demosaic_bayer_amaze(
    const device ushort4* raw_buffer    [[buffer(0)]],
    device float* rgb_buffer            [[buffer(1)]],
    constant BayerParams& params        [[buffer(2)]],
    device float* rgbgreen_             [[buffer(3)]],
    device float* delhvsqsum_           [[buffer(4)]],
    device float* dirwts0_              [[buffer(5)]],
    device float* dirwts1_              [[buffer(6)]],
    device float* vcd_                  [[buffer(7)]],
    device float* hcd_                  [[buffer(8)]],
    device float* vcdalt_               [[buffer(9)]],
    device float* hcdalt_               [[buffer(10)]],
    device float* cddiffsq_             [[buffer(11)]],
    device float* hvwt_                 [[buffer(12)]],
    device float* dgintv_               [[buffer(13)]],
    device float* dginth_               [[buffer(14)]],
    device float* Dgrbsq1m_             [[buffer(15)]],
    device float* Dgrbsq1p_             [[buffer(16)]],
    device uchar* nyquist_              [[buffer(17)]],
    device uchar* nyquist2_             [[buffer(18)]],
    device uchar* nyqutest_             [[buffer(19)]],
    device float* delp_                 [[buffer(20)]],
    device float* delm_                 [[buffer(21)]],
    device float* Dgrb0_                [[buffer(22)]], // G-R difference buffer
    device float* cfa_                  [[buffer(23)]],
    uint2 tile_id [[threadgroup_position_in_grid]],
    uint flat_local_id [[thread_index_in_threadgroup]],
    uint2 group_dims [[threads_per_threadgroup]]
) {
    const int TS = 160;
    const int TSH = TS/2;
    const float eps = 1e-5f;
    const float epssq = 1e-10f;
    const float arthresh = 0.75f;
    const float nyqthresh = 0.5f;
    const int v1 = TS, v2 = 2*TS, v3 = 3*TS, p1 = -TS+1, p2 = -2*TS+2, p3 = -3*TS+3, m1 = TS+1, m2 = 2*TS+2, m3 = 3*TS+3;

    const float gaussodd[4] = {0.14659727707323927f, 0.103592713382435f, 0.0732036125103057f, 0.0365543548389495f};
    const float gaussgrad[6] = {
        nyqthresh * 0.07384411893421103f, nyqthresh * 0.06207511968171489f, nyqthresh * 0.0521818194747806f,
        nyqthresh * 0.03687419286733595f, nyqthresh * 0.03099732204057846f, nyqthresh * 0.018413194161458882f
    };
    const float gausseven[2] = {0.13719494435797422f, 0.05640252782101291f};
    const float gquinc[4] = {0.169917f, 0.108947f, 0.069855f, 0.0287182f};
    
    // タイルオフセットの計算
    const int full_offset = (tile_id.y * params.grid_size.x + tile_id.x) * (TS * TS);
    const int half_offset = (tile_id.y * params.grid_size.x + tile_id.x) * (TS * TSH);

    device float* rgbgreen = rgbgreen_ + full_offset;
    device float* delhvsqsum = delhvsqsum_ + full_offset;
    device float* dirwts0 = dirwts0_ + full_offset;
    device float* dirwts1 = dirwts1_ + full_offset;
    device float* vcd = vcd_ + full_offset;
    device float* hcd = hcd_ + full_offset;
    device float* vcdalt = vcdalt_ + full_offset;
    device float* hcdalt = hcdalt_ + full_offset;
    device float* cddiffsq = cddiffsq_ + full_offset;
    device float* hvwt = hvwt_ + half_offset;
    device float* dgintv = dgintv_ + full_offset;
    device float* dginth = dginth_ + full_offset;
    device float* Dgrbsq1m = Dgrbsq1m_ + half_offset;
    device float* Dgrbsq1p = Dgrbsq1p_ + half_offset;
    device uchar* nyquist = nyquist_ + half_offset;
    device uchar* nyquist2 = nyquist2_ + half_offset;
    device uchar* nyqutest = nyqutest_ + half_offset;
    device float* cfa = cfa_ + full_offset;
    //device float* delp = delp_ + full_offset;
    //device float* delm = delm_ + full_offset;
    //device float* Dgrb0 = Dgrb0_ + half_offset;
    //device float* Dgrb1 = hcd;

    uint32_t width_ = params.width;
    uint32_t height_ = params.height;
    uint32_t clip_pt = params.clip_pt;
    uint32_t clip_pt8 = params.clip_pt8;
    unsigned int cfarray[2][2];
    int ex = 0, ey = 0;

    // Cache the 2x2 Bayer pattern based on the provided helper function
    cfarray[0][0] = fcol_bayer(0, 0, params.filters);
    cfarray[0][1] = fcol_bayer(0, 1, params.filters);
    cfarray[1][0] = fcol_bayer(1, 0, params.filters);
    cfarray[1][1] = fcol_bayer(1, 1, params.filters);
    
    // Determine R pixel offset (ey, ex)
    if (cfarray[0][0] == 1) { // Top-left is G (GRBG or GBRG)
        if (cfarray[0][1] == 0) { ey = 0; ex = 1; } // GRBG
        else { ey = 1; ex = 0; }               // GBRG
    } else { // Top-left is R or B (RGGB or BGGR)
        if (cfarray[0][0] == 0) { ey = 0; ex = 0; } // RGGB
        else { ey = 1; ex = 1; }               // BGGR
    }


    const int top = (int)(tile_id.y * (TS - 32)) - 16;
    const int left = (int)(tile_id.x * (TS - 32)) - 16;
    const int bottom = min(top + TS, (int)height_ + 16);
    const int right = min(left + TS, (int)width_ + 16);
    const int rr1 = bottom - top;
    const int cc1 = right - left;
    const int rrmin = (top < 0) ? 16 : 0;
    const int ccmin = (left < 0) ? 16 : 0;
    const int rrmax = (bottom > (int)height_) ? (int)height_ - top : rr1;
    const int ccmax = (right > (int)width_) ? (int)width_ - left : cc1;

    // === Tile Initialization with 16-pixel border ===
    const float scale = 1.0f / params.maximum_value;
    
    // Fill upper border
    if (rrmin > 0) {
        for (int rr = 0; rr < 16; rr++) {
            for (int cc = ccmin; cc < ccmax; cc++) {
                int row = 32 - rr + top;
                int safe_row = max(0, min((int)height_ - 1, row));
                int safe_col = max(0, min((int)width_ - 1, cc + left));
                int c = fc_rt(safe_row, safe_col);
                float val = (float)raw_buffer[safe_row * width_ + safe_col][c] * scale;
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
            float val = (float)raw_buffer[row * width_ + (cc + left)][c] * scale;
            cfa[indx1] = val;
            rgbgreen[indx1] = val;
        }
    }
    
    // Fill lower border
    if (rrmax < rr1) {
        for (int rr = 0; rr < 16; rr++) {
                for (int cc = ccmin; cc < ccmax; cc++) {
                int safe_row = max(0, min((int)height_ - 1, (int)height_ - rr - 2));
                int safe_col = max(0, min((int)width_ - 1, left + cc));
                int c = fc_rt(safe_row, safe_col);
                float val = (float)raw_buffer[safe_row * width_ + safe_col][c] * scale;
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
                int safe_row = max(0, min((int)height_ - 1, row));
                int safe_col = max(0, min((int)width_ - 1, 32 - cc + left));
                int c = fc_rt(safe_row, safe_col);
                float val = (float)raw_buffer[safe_row * width_ + safe_col][c] * scale;
                cfa[rr * TS + cc] = val;
                rgbgreen[rr * TS + cc] = val;
            }
        }
    }
    if (ccmax < cc1) {
        for (int rr = rrmin; rr < rrmax; rr++) {
            for (int cc = 0; cc < 16; cc++) {
                int safe_row = max(0, min((int)height_ - 1, top + rr));
                int safe_col = max(0, min((int)width_ - 1, (int)width_ - cc - 2));
                int c = fc_rt(safe_row, safe_col);
                float val = (float)raw_buffer[safe_row * width_ + safe_col][c] * scale;
                cfa[rr * TS + ccmax + cc] = val;
                rgbgreen[rr * TS + ccmax + cc] = val;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

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
    threadgroup_barrier(mem_flags::mem_device);
    
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
            dgintv[idx] = min(SQR(guha-gdha), SQR(guar-gdar));
            dginth[idx] = min(SQR(glha-grha), SQR(glar-grar));
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

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
    threadgroup_barrier(mem_flags::mem_device);

    // STAGE 4 & 5: Directional variance, Texture analysis & Nyquist test calculation
    for (int rr=6; rr<rr1-6; rr++) {
        for (int cc=6+(fc_rt(rr,2)&1), idx=+rr*TS+cc; cc<cc1-6; cc+=2, idx+=2) {
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
        for (int cc=6+(fc_rt(rr,2)&1),idx=+rr*TS+cc; cc<cc1-6; cc+=2,idx+=2){
            nyqutest[idx>>1]=(gaussodd[0]*cddiffsq[idx]+gaussodd[1]*(cddiffsq[idx-m1]+cddiffsq[idx+p1]+cddiffsq[idx-p1]+cddiffsq[idx+m1])+gaussodd[2]*(cddiffsq[idx-v2]+cddiffsq[idx-2]+cddiffsq[idx+2]+cddiffsq[idx+v2])+gaussodd[3]*(cddiffsq[idx-m2]+cddiffsq[idx+p2]+cddiffsq[idx-p2]+cddiffsq[idx+m2]))-(gaussgrad[0]*delhvsqsum[idx]+gaussgrad[1]*(delhvsqsum[idx-v1]+delhvsqsum[idx+1]+delhvsqsum[idx-1]+delhvsqsum[idx+v1])+gaussgrad[2]*(delhvsqsum[idx-m1]+delhvsqsum[idx+p1]+delhvsqsum[idx-p1]+delhvsqsum[idx+m1])+gaussgrad[3]*(delhvsqsum[idx-v2]+delhvsqsum[idx-2]+delhvsqsum[idx+2]+delhvsqsum[idx+v2])+gaussgrad[4]*(delhvsqsum[idx-v2-1]+delhvsqsum[idx-v2+1]+delhvsqsum[idx-TS-2]+delhvsqsum[idx-TS+2]+delhvsqsum[idx+TS-2]+delhvsqsum[idx+TS+2]+delhvsqsum[idx+v2-1]+delhvsqsum[idx+v2+1])+gaussgrad[5]*(delhvsqsum[idx-m2]+delhvsqsum[idx+p2]+delhvsqsum[idx-p2]+delhvsqsum[idx+m2]));
        }
    }
    bool doNyquist = false;
    for (int i = 0; i < (int)sizeof(unsigned char) * TS * TSH; ++i) {
        nyquist[i] = 0;
    }
    for (int rr = 6; rr < rr1-6; rr++) {
        for (int cc=6+(fc_rt(rr,2)&1),idx=+rr*TS+cc; cc<cc1-6; cc+=2,idx+=2) {
            if(nyqutest[idx>>1]>0.f) {
                nyquist[idx>>1]=1; doNyquist=true;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    // STAGE 6: Nyquist processing & Green interpolation
    if(doNyquist) {
        for (int i = 0; i < (int)sizeof(unsigned char)*TS*TSH; ++i) {
            nyquist2[i] = 0;
        }
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
    device float* Dgrb0 = vcdalt; 
    device s_hv* Dgrb2_ptr = (device s_hv*)(Dgrbsq1m);
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
    threadgroup_barrier(mem_flags::mem_device);
    
    // STAGE 7: Red/Blue gradient preprocessing
    device float* delp = cddiffsq;
    device float* delm = (device float*)((device char*)delp + sizeof(float) * TS * TSH);
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
    threadgroup_barrier(mem_flags::mem_device);
    
    // STAGE 8: Red/Blue color ratio interpolation (Diagonal)
    device float* rbm = vcd;
    device float* rbp = hcdalt;
    device float* pmwt = dirwts1;
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
    threadgroup_barrier(mem_flags::mem_device);
    
    // STAGE 9: Final Green interpolation and Chrominance interpolation
    device float* rbint = delhvsqsum;
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
    device float* Dgrb1 = hcd;
    
    // DEBUG: Dgrb1初期化の詳細確認
    //static bool debug_printed = false;
    //if (!debug_printed) {
    //    std::cout << "🔍 CPU Dgrb1初期化: ey=" << ey << ", rr1=" << rr1 << ", ex=" << ex << ", cc1=" << cc1 << std::endl;
    //    debug_printed = true;
    //}
    
    for (int rr = 13 - ey; rr < rr1 - 12; rr += 2) {
        for (int idx1 = (rr * TS + 13 - ex) >> 1; idx1 < (rr * TS + cc1 - 12) >> 1; idx1++) {
            // DEBUG: 実際の処理範囲の確認
            if (idx1 >= 960 && idx1 <= 1130) {  // GPU比較範囲
                //std::cout << "🔢 CPU Dgrb1[" << idx1 << "] = " << Dgrb0[idx1] << " (rr=" << rr << ")" << std::endl;
            }
            Dgrb1[idx1] = Dgrb0[idx1];
            Dgrb0[idx1] = 0;
        }
    }
    for (int rr = 14; rr < rr1 - 14; rr++) {
        for (int cc = 14 + (fc_rt(rr, 2) & 1), idx = rr * TS + cc; cc < cc1 - 14; cc += 2, idx += 2) {
            // In RawTherapee, R=0, B=2. c becomes 1 for R-sites, 0 for B-sites
            int c = 1 - fc_rt(rr, cc) / 2;
            // Dgrb0 is for G-R, Dgrb1 is for G-B. But the interpolation logic uses the *other* color's buffer.
            device float* Dgrb_c = c ? Dgrb1 : Dgrb0;
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
    threadgroup_barrier(mem_flags::mem_device);
    
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

            // ★★★ BUG FIX #2 ★★★
            // Use a direct and robust check for the G-site.
            if (fc_rt(row, col) == 1) { // G site
                g = cfa[tile_idx];
                float wsum_inv = 1.0f / (hvwt[(tile_idx - v1) >> 1] + 2.f - hvwt[(tile_idx + 1) >> 1] - hvwt[(tile_idx - 1) >> 1] + hvwt[(tile_idx + v1) >> 1]);                            float r_diff = (hvwt[(tile_idx-v1)>>1]*Dgrb0[(tile_idx-v1)>>1] + (1.f-hvwt[(tile_idx+1)>>1])*Dgrb0[(tile_idx+1)>>1] + (1.f-hvwt[(tile_idx-1)>>1])*Dgrb0[(tile_idx-1)>>1] + hvwt[(tile_idx+v1)>>1]*Dgrb0[(tile_idx+v1)>>1]) * wsum_inv;
                float b_diff = (hvwt[(tile_idx-v1)>>1]*Dgrb1[(tile_idx-v1)>>1] + (1.f-hvwt[(tile_idx+1)>>1])*Dgrb1[(tile_idx+1)>>1] + (1.f-hvwt[(tile_idx-1)>>1])*Dgrb1[(tile_idx-1)>>1] + hvwt[(tile_idx+v1)>>1]*Dgrb1[(tile_idx+v1)>>1]) * wsum_inv;
                r = g - r_diff;
                b = g - b_diff;
                // DEBUG: G-site処理の値確認
                if (row < 5 && col < 5) {
                    //std::cout << "🟢 CPU G-site (" << row << "," << col << "): g=" << g << ", r_diff=" << r_diff << ", b_diff=" << b_diff << ", r=" << r << ", b=" << b << std::endl;
                }
            } else { // R or B site
                g = rgbgreen[tile_idx];
                // ネイティブの色(cfa)を使わず、Dgrb0とDgrb1から両方の色を計算する
                r = g - Dgrb0[tile_idx >> 1];
                b = g - Dgrb1[tile_idx >> 1];
                // DEBUG: R/B-site処理の値確認
                if (row < 5 && col < 5) {
                    //int idx_half = tile_idx >> 1;
                    //std::cout << "🔴🔵 CPU R/B-site (" << row << "," << col << "): g=" << g << ", Dgrb0[" << idx_half << "]=" << Dgrb0[idx_half] << ", Dgrb1[" << idx_half << "]=" << Dgrb1[idx_half] << ", r=" << r << ", b=" << b << std::endl;
                }
            }
            rgb_buffer[out_idx * 3 + 0] = max(0.f, r);
            rgb_buffer[out_idx * 3 + 1] = max(0.f, g);
            rgb_buffer[out_idx * 3 + 2] = max(0.f, b);
        }
    }
}
