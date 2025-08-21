//
// camera_matrices.h  
// LibRaw Enhanced - Camera Color Matrix Management
// Direct integration with LibRaw's internal color matrix database
//

#pragma once

#include <string>
#include <cstring>

// Forward declaration for LibRaw integration
class LibRaw;

namespace libraw_enhanced {

// Camera matrix computation result
struct ColorTransformMatrix {
    float transform[3][4];  // Final camera->output color space matrix
    bool valid;
    
    ColorTransformMatrix() : valid(false) {
        memset(transform, 0, sizeof(transform));
    }

    void set_default() {
        transform[0][0] = 1.0f; transform[0][1] = 0.0f; transform[0][2] = 0.0f; transform[0][3] = 0.0f;
        transform[1][0] = 0.0f; transform[1][1] = 1.0f; transform[1][2] = 0.0f; transform[1][3] = 0.0f;
        transform[2][0] = 0.0f; transform[2][1] = 0.0f; transform[2][2] = 1.0f; transform[2][3] = 0.0f;
        valid = true;
    }
};

// Main interface for camera matrix operations
class CameraMatrixManager {
public:
    CameraMatrixManager();
    ~CameraMatrixManager();
    
    // Compute color transform matrix for given camera and output color space
    // Uses LibRaw's internal adobe_coeff() function directly
    ColorTransformMatrix get_color_transform(const char* camera_make,
                                           const char* camera_model,
                                           int output_color_space = 1);  // 1=sRGB
    
    // Check if camera is supported in LibRaw database
    bool is_camera_supported(const char* camera_make, const char* camera_model);
    
    // Get fallback matrix for unsupported cameras
    ColorTransformMatrix get_fallback_transform(int output_color_space = 1);
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Convenience functions for common operations
ColorTransformMatrix compute_camera_transform(const char* make, const char* model, int color_space = 1);

} // namespace libraw_enhanced