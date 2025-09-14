//
// shader_common.h
// LibRaw Enhanced - Common Metal definitions and utilities
//

#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#include <simd/simd.h>
#define CONSTANT constant
#define STATIC_CONSTANT constant
using namespace metal;
#else
#include <cstdint>
#include <algorithm> // Required for std::min and std::max
#define CONSTANT const
#define STATIC_CONSTANT static const
using namespace std;
#endif

STATIC_CONSTANT uint32_t bayer_to3[4] = {0, 1, 2, 1}; // Maps Bayer filter to RGBG

inline uint32_t fcol_bayer_native(uint32_t row, uint32_t col, uint32_t filters) {
    return (filters >> ((((row) << 1 & 14) | ((col) & 1)) << 1)) & 3;
}

inline uint32_t fcol_bayer(uint32_t row, uint32_t col, uint32_t filters) {
    return bayer_to3[fcol_bayer_native(row, col, filters)];
}

inline uint32_t fcol_xtrans(uint32_t row, uint32_t col, CONSTANT char xtrans[6][6]) {
    return xtrans[(row + 6) % 6][(col + 6) % 6];
}

template<typename T, typename U, typename V>
inline T ulim_generic(T val, U upper, V lower) {
    if (val > upper) return static_cast<T>(upper);
    if (val < lower) return static_cast<T>(lower);
    return val;
}

template<typename T>
inline T SQR(T x) {
    return x * x;
}

template<typename T>
inline T LIM(T val, T low, T high) {
    return max(low, min(val, high));
}

template<typename T>
inline T median(T a, T b, T c) {
    if (a > b) { float t = a; a = b; b = t; }
    if (b > c) { float t = b; b = c; c = t; }
    if (a > b) { float t = a; a = b; b = t; }
    return b;
//    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

#endif // SHADER_COMMON_H