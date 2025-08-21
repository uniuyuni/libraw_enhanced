//
// camera_matrices.cpp
// LibRaw Enhanced - Camera Color Matrix Management  
// Direct integration with LibRaw's adobe_coeff database
//

#include "camera_matrices.h"
#include <memory>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <string>
#include <libraw/libraw.h>
#include <libraw/libraw_const.h>

namespace libraw_enhanced {

// Helper function to convert LibRaw make string to make index
// Uses LibRaw's cameramakeridx2maker function by reverse lookup
unsigned get_libraw_make_index(const char* make) {
    if (!make) return LIBRAW_CAMERAMAKER_Unknown;
    
    // Create a temporary LibRaw instance to access the cameramakeridx2maker function
    static LibRaw temp_libraw;
    
    // Case-insensitive string matching
    auto case_contains = [](const char* text, const char* substring) -> bool {
        if (!text || !substring) return false;
        std::string text_lower(text);
        std::string sub_lower(substring);
        std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
        std::transform(sub_lower.begin(), sub_lower.end(), sub_lower.begin(), ::tolower);
        return text_lower.find(sub_lower) != std::string::npos;
    };
    
    // Search through all possible maker indices using LibRaw's own function
    for (unsigned idx = LIBRAW_CAMERAMAKER_Unknown + 1; idx < 200; idx++) {
        const char* maker_name = temp_libraw.cameramakeridx2maker(idx);
        if (maker_name) {
            if (case_contains(make, maker_name) || case_contains(maker_name, make)) {
                return idx;
            }
        }
    }
    
    return LIBRAW_CAMERAMAKER_Unknown; // Unknown maker
}

class CameraMatrixManager::Impl {
public:
    LibRaw libraw_instance;
    
    Impl() {
        // LibRaw camera matrix database initialized
    }
    
    ~Impl() = default;
    
    ColorTransformMatrix compute_transform(const char* make, const char* model, int output_color_space) {
        ColorTransformMatrix result;
        
        try {
            // Get LibRaw make index
            unsigned make_idx = get_libraw_make_index(make);
            
            // Setup minimal LibRaw instance state for adobe_coeff to work
            libraw_instance.imgdata.idata.colors = 3; // RGB
            libraw_instance.imgdata.color.black = 0;
            libraw_instance.imgdata.color.maximum = 0xffff;
            memset(libraw_instance.imgdata.color.cblack, 0, sizeof(libraw_instance.imgdata.color.cblack));
            
            // Call LibRaw's adobe_coeff to get cam_xyz matrix
            // adobe_coeff populates imgdata.color.cam_xyz from internal database
            int matrix_found = libraw_instance.adobe_coeff(make_idx, model, 0);
            
            if (matrix_found > 0) {
                // Convert cam_xyz from float[4][3] to double[4][3] for cam_xyz_coeff
                double cam_xyz_double[4][3];
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 3; j++) {
                        cam_xyz_double[i][j] = (double)libraw_instance.imgdata.color.cam_xyz[i][j];
                    }
                }
                
                // Now use LibRaw's cam_xyz_coeff to compute rgb_cam properly
                // This is the correct LibRaw pipeline that handles color science properly
                cam_xyz_coeff(result.transform, cam_xyz_double, 3);
                
                // Apply output color space conversion if not sRGB
                if (output_color_space != 1) {
                    apply_output_colorspace_transform(result.transform, output_color_space);
                }
                
                result.valid = true;
                
                // Optional: Log successful matrix retrieval for debugging
                std::cout << "Found camera matrix for " << make << " " << model << " (cam_xyz_coeff)" << std::endl;
                
            } else {
                result.valid = false;
                std::cout << "⚠️ No camera matrix found for " << make << " " << model << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error computing camera matrix: " << e.what() << std::endl;
            result.valid = false;
        }
        
        return result;
    }
    
private:
    // XYZ to RGB transformation matrix (from LibRaw_constants::xyz_rgb)
    static constexpr double xyz_rgb[3][3] = {
        {0.4124564, 0.3575761, 0.1804375},
        {0.2126729, 0.7151522, 0.0721750}, 
        {0.0193339, 0.1191920, 0.9503041}
    };
    
    void pseudoinverse(double in[4][3], double out[4][3], int size) {
        double work[3][6], num;
        int i, j, k;

        for (i = 0; i < 3; i++) {
            for (j = 0; j < 6; j++)
                work[i][j] = (j == i + 3) ? 1.0 : 0.0;
            for (j = 0; j < 3; j++)
                for (k = 0; k < size && k < 4; k++)
                    work[i][j] += in[k][i] * in[k][j];
        }
        
        for (i = 0; i < 3; i++) {
            num = work[i][i];
            for (j = 0; j < 6; j++)
                if (fabs(num) > 0.00001)
                    work[i][j] /= num;
            for (k = 0; k < 3; k++) {
                if (k == i)
                    continue;
                num = work[k][i];
                for (j = 0; j < 6; j++)
                    work[k][j] -= work[i][j] * num;
            }
        }
        
        for (i = 0; i < size && i < 4; i++)
            for (j = 0; j < 3; j++)
                for (out[i][j] = k = 0; k < 3; k++)
                    out[i][j] += work[j][k + 3] * in[i][k];
    }
    
    // LibRaw cam_xyz_coeff function implementation  
    void cam_xyz_coeff(float rgb_cam[3][4], double cam_xyz[4][3], int colors) {
        double cam_rgb[4][3], inverse[4][3], num;
        int i, j, k;

        // Multiply out XYZ colorspace  
        for (i = 0; i < colors && i < 4; i++) {
            for (j = 0; j < 3; j++) {
                cam_rgb[i][j] = 0.0;
                for (k = 0; k < 3; k++)
                    cam_rgb[i][j] += cam_xyz[i][k] * xyz_rgb[k][j];
            }
        }

        // Normalize cam_rgb so that cam_rgb * (1,1,1) is (1,1,1,1)
        for (i = 0; i < colors && i < 4; i++) {
            for (num = j = 0; j < 3; j++)
                num += cam_rgb[i][j];
            if (num > 0.00001) {
                for (j = 0; j < 3; j++)
                    cam_rgb[i][j] /= num;
            } else {
                for (j = 0; j < 3; j++)
                    cam_rgb[i][j] = 0.0;
            }
        }
        
        // Compute pseudoinverse
        pseudoinverse(cam_rgb, inverse, colors);
        
        // Set rgb_cam matrix
        for (i = 0; i < 3; i++)
            for (j = 0; j < colors && j < 4; j++)
                rgb_cam[i][j] = (float)inverse[j][i];
    }

    void apply_output_colorspace_transform(float transform[3][4], int output_color_space) {
        // Complete color space matrices (XYZ->RGB conversion matrices)
        // Based on constants.py ColorSpace definitions (0-8)
        static const double output_matrices[][3][3] = {
            // 0: Raw (identity - linear passthrough)
            {{1,0,0}, {0,1,0}, {0,0,1}},
            
            // 1: sRGB (ITU-R BT.709 / IEC 61966-2-1)
            {{3.2404542, -1.5371385, -0.4985314},
             {-0.9692660, 1.8760108, 0.0415560},
             {0.0556434, -0.2040259, 1.0572252}},
            
            // 2: Adobe RGB (1998)
            {{2.0413690, -0.5649464, -0.3446944},
             {-0.9692660, 1.8760108, 0.0415560},
             {0.0134474, -0.1183897, 1.0154096}},
            
            // 3: Wide Gamut RGB
            {{1.4628067, -0.1840623, -0.2743606},
             {-0.5217933, 1.4472381, 0.0677227},
             {0.0349342, -0.0968930, 1.2884099}},
            
            // 4: ProPhoto RGB (ROMM RGB)
            {{1.3459433, -0.2556075, -0.0511118},
             {-0.5445989, 1.5081673, 0.0205351},
             {0.0000000, 0.0000000, 1.2118128}},
            
            // 5: XYZ (identity - linear XYZ passthrough)
            {{1,0,0}, {0,1,0}, {0,0,1}},
            
            // 6: ACES AP1 (Academy Color Encoding System)
            {{1.7166511, -0.3556708, -0.2533663},
             {-0.6666844, 1.6164812, 0.0157685},
             {0.0176399, -0.0427706, 0.9421031}},
            
            // 7: Display P3 (DCI-P3 D65)
            {{2.4934969, -0.9313836, -0.4027108},
             {-0.8294890, 1.7626641, 0.0236247},
             {0.0358458, -0.0761724, 0.9568845}},
            
            // 8: Rec.2020 (ITU-R BT.2020)
            {{1.7166511, -0.3556708, -0.2533663},
             {-0.6666844, 1.6164812, 0.0157685},
             {0.0176399, -0.0427706, 0.9421031}}
        };
        
        if (output_color_space < 0 || output_color_space > 8) {
            output_color_space = 1; // Default to sRGB
        }
        
        // Check for linear color spaces that should skip matrix transformation
        if (output_color_space == 0 || output_color_space == 5) {
            // Raw (0) and XYZ (5) - linear passthrough, no additional matrix needed
            return;
        }
        
        // Apply output color space transformation: out_cam = output_matrix * rgb_cam
        float temp[3][4];
        memcpy(temp, transform, sizeof(temp));
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                transform[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    transform[i][j] += output_matrices[output_color_space][i][k] * temp[k][j];
                }
            }
        }
    }
};

// CameraMatrixManager implementation
CameraMatrixManager::CameraMatrixManager() : pimpl_(std::make_unique<Impl>()) {}

CameraMatrixManager::~CameraMatrixManager() = default;

ColorTransformMatrix CameraMatrixManager::get_color_transform(const char* camera_make,
                                                             const char* camera_model,
                                                             int output_color_space) {
    return pimpl_->compute_transform(camera_make, camera_model, output_color_space);
}

bool CameraMatrixManager::is_camera_supported(const char* camera_make, const char* camera_model) {
    ColorTransformMatrix result = get_color_transform(camera_make, camera_model, 1);
    return result.valid;
}

ColorTransformMatrix CameraMatrixManager::get_fallback_transform(int output_color_space) {
    // Generic fallback matrix - simple identity transform
    ColorTransformMatrix result;
    
    // Identity matrix for RGB channels
    result.transform[0][0] = 1.0f; result.transform[0][1] = 0.0f; result.transform[0][2] = 0.0f; result.transform[0][3] = 0.0f;
    result.transform[1][0] = 0.0f; result.transform[1][1] = 1.0f; result.transform[1][2] = 0.0f; result.transform[1][3] = 0.0f;
    result.transform[2][0] = 0.0f; result.transform[2][1] = 0.0f; result.transform[2][2] = 1.0f; result.transform[2][3] = 0.0f;
    
    result.valid = true;
    return result;
}

// Convenience function
ColorTransformMatrix compute_camera_transform(const char* make, const char* model, int color_space) {
    static CameraMatrixManager manager; // Singleton instance
    return manager.get_color_transform(make, model, color_space);
}

} // namespace libraw_enhanced