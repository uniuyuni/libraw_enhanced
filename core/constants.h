//
// constants.h
// LibRaw Enhanced - Constants and Enums
// rawpy/LibRaw互換のEnum値定義（C++版）
//

#pragma once

namespace libraw_enhanced {

// Demosaic Algorithm Constants (rawpy/LibRaw compatible)
enum class DemosaicAlgorithm : int {
    Linear = 0,                    // Linear/Bilinear interpolation (LibRaw quality=0)
    VNG = 1,                      // Variable Number of Gradients (LibRaw quality=1)
    PPG = 2,                      // Patterned Pixel Grouping (LibRaw quality=2)
    AHD = 3,                      // Adaptive Homogeneity-Directed (LibRaw quality=3)
    DCB = 4,                      // DCB (Dave Coffin's method) (LibRaw quality=4)
    ModifiedAHD = 5,              // Modified AHD (requires GPL2 pack)
    AFD = 6,                      // AFD (Adaptive Filtered Demosaicing) (requires GPL2 pack)
    VCD = 7,                      // VCD (Variable Color Demosaicing) (requires GPL2 pack)
    MixedVCDModifiedAHD = 8,      // Mixed VCD and Modified AHD (requires GPL2 pack)
    LMMSE = 9,                    // LMMSE (Linear Minimum Mean Square Error) (requires GPL2 pack)
    AMaZE = 10,                   // AMaZE (Aliasing minimization and zipper elimination) (requires GPL3 pack)
    DHT = 11,                     // DHT interpolation (LibRaw quality=11)
    AAHD = 12,                    // AAHD (Modified AHD variant) (LibRaw quality=12)
    
    // LibRaw Enhanced extensions
    Adaptive = 100,               // 適応的デモザイク
    MLEnhanced = 101             // 機械学習強化
};

// Color Space Constants (rawpy/LibRaw compatible)
enum class ColorSpace : int {
    Raw = 0,                     // LIBRAW_COLORSPACE_NotFound
    sRGB = 1,                   // LIBRAW_COLORSPACE_sRGB
    AdobeRGB = 2,               // LIBRAW_COLORSPACE_AdobeRGB
    WideGamutRGB = 3,           // LIBRAW_COLORSPACE_WideGamutRGB
    ProPhotoRGB = 4,            // LIBRAW_COLORSPACE_ProPhotoRGB
    XYZ = 5,                    // rawpy互換値（LibRawには直接対応なし）
    ACES = 6,                   // rawpy拡張
    P3D65 = 7,                  // rawpy拡張
    Rec2020 = 8                 // rawpy拡張
};


// Human-readable algorithm names
inline const char* get_algorithm_name(DemosaicAlgorithm algo) {
    switch (algo) {
        case DemosaicAlgorithm::Linear: return "Linear";
        case DemosaicAlgorithm::VNG: return "VNG";
        case DemosaicAlgorithm::PPG: return "PPG";
        case DemosaicAlgorithm::AHD: return "AHD";
        case DemosaicAlgorithm::DCB: return "DCB";
        case DemosaicAlgorithm::ModifiedAHD: return "ModifiedAHD";
        case DemosaicAlgorithm::AFD: return "AFD";
        case DemosaicAlgorithm::VCD: return "VCD";
        case DemosaicAlgorithm::MixedVCDModifiedAHD: return "MixedVCDModifiedAHD";
        case DemosaicAlgorithm::LMMSE: return "LMMSE";
        case DemosaicAlgorithm::AMaZE: return "AMaZE";
        case DemosaicAlgorithm::DHT: return "DHT";
        case DemosaicAlgorithm::AAHD: return "AAHD";
        case DemosaicAlgorithm::Adaptive: return "Adaptive";
        case DemosaicAlgorithm::MLEnhanced: return "MLEnhanced";
        default: return "Unknown";
    }
}

inline const char* get_colorspace_name(ColorSpace cs) {
    switch (cs) {
        case ColorSpace::Raw: return "Raw";
        case ColorSpace::sRGB: return "sRGB";
        case ColorSpace::AdobeRGB: return "AdobeRGB";
        case ColorSpace::WideGamutRGB: return "WideGamutRGB";
        case ColorSpace::ProPhotoRGB: return "ProPhotoRGB";
        case ColorSpace::XYZ: return "XYZ";
        case ColorSpace::ACES: return "ACES";
        case ColorSpace::P3D65: return "P3D65";
        case ColorSpace::Rec2020: return "Rec2020";
        default: return "Unknown";
    }
}

} // namespace libraw_enhanced