#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <map>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace libraw_enhanced {

#ifdef METAL_ACCELERATION_AVAILABLE
// Forward declarations for common types (defined in accelerator.h)
struct ProcessingParams;
struct ImageBuffer;
class LibRawGPUCallbacks;
#endif


// 処理時間計測構造体
struct ProcessingTimes {
    double total_time = 0.0;
    double file_load_time = 0.0;
    double unpack_time = 0.0;
    double demosaic_time = 0.0;
    double gpu_demosaic_time = 0.0;
    double color_correction_time = 0.0;
    double gamma_correction_time = 0.0;
    double memory_copy_time = 0.0;
    double postprocess_time = 0.0;
    bool gpu_used = false;
    std::string demosaic_algorithm_name;
    
    ProcessingTimes() = default;
};

// 処理済み画像データ構造体
struct ProcessedImageData {
    size_t width = 0;
    size_t height = 0;
    size_t channels = 0;
    size_t bits_per_sample = 0;
    int error_code = 0;
    std::vector<uint8_t> data;
    ProcessingTimes timing_info;  // 追加: 処理時間情報
    
    ProcessedImageData() = default;
    
    // データサイズの計算
    size_t get_data_size() const {
        return width * height * channels * (bits_per_sample / 8);
    }
    
    // データの有効性チェック
    bool is_valid() const {
        return width > 0 && height > 0 && channels > 0 && 
               bits_per_sample > 0 && error_code == 0 && 
               !data.empty();
    }
};

// 画像メタデータ構造体
struct ImageInfo {
    size_t width = 0;
    size_t height = 0; 
    size_t raw_width = 0;
    size_t raw_height = 0;
    int colors = 3;
    float cam_mul[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float pre_mul[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::string camera_make;
    std::string camera_model;
    
    ImageInfo() = default;
};

// LibRawWrapper クラス宣言
class LibRawWrapper {
public:
    LibRawWrapper();
    ~LibRawWrapper();
    
    // ファイル操作
    int load_file(const std::string& filename);
    int load_buffer(const std::vector<uint8_t>& buffer);
    int unpack();
    int process();
    
    // 画像データ取得
    ProcessedImageData get_processed_image();
    ImageInfo get_image_info();
    std::vector<uint16_t> get_raw_image();
    py::array_t<uint16_t> get_raw_image_as_numpy();
    
    // メタデータアクセス
    std::string get_camera_make() const;
    std::string get_camera_model() const;
    
    // rawpy互換processing method (Python dict will be converted to ProcessingParams)
    ProcessedImageData process_with_dict(const std::map<std::string, float>& float_params,
                                         const std::map<std::string, int>& int_params,
                                         const std::map<std::string, bool>& bool_params,
                                         const std::map<std::string, std::string>& string_params);
    
    // 設定メソッド
    void set_debug_mode(bool enable);
    void close();
    
#ifdef METAL_ACCELERATION_AVAILABLE
    // Metal関連メソッド（実装はlibraw_wrapper.cppで定義）
    void set_processing_params(const ProcessingParams& params);
    void enable_gpu_acceleration(bool enable);
    bool is_gpu_available() const;
    std::string get_metal_device_info() const;
    
    // REMOVED: void enable_custom_pipeline(bool enable); - unused custom pipeline feature
#endif

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

#ifdef METAL_ACCELERATION_AVAILABLE
// rawpy完全互換パラメータ変換関数（実装はlibraw_wrapper.cppで定義）
ProcessingParams create_params_from_rawpy_args(
    // Basic parameters
    bool use_camera_wb = true,
    bool half_size = false,
    bool four_color_rgb = false,
    int output_bps = 16,
    int user_flip = -1,
    
    // Demosaicing parameters
    int demosaic_algorithm = 1,
    int dcb_iterations = 0,
    bool dcb_enhance = false,
    
    // Noise reduction parameters
    int fbdd_noise_reduction = 0,
    float noise_thr = 0.0f,
    int median_filter_passes = 0,
    
    // White balance parameters
    bool use_auto_wb = false,
    const std::array<float, 4>& user_wb = {1.0f, 1.0f, 1.0f, 1.0f},
    
    // Color and output parameters
    int output_color = 1,
    
    // Brightness and exposure parameters
    float bright = 1.0f,
    bool no_auto_bright = false,
    float auto_bright_thr = 0.01f,
    float adjust_maximum_thr = 0.75f,
    
    // Highlight processing
    int highlight_mode = 0,
    
    // Exposure correction parameters
    float exp_shift = 1.0f,
    float exp_preserve_highlights = 0.0f,
    
    // Gamma and scaling
    const std::pair<float, float>& gamma = {2.222f, 4.5f},
    bool no_auto_scale = false,
    
    // Color correction parameters
    float chromatic_aberration_red = 1.0f,
    float chromatic_aberration_blue = 1.0f,
    
    // User adjustments
    int user_black = -1,
    int user_sat = -1,
    
    // File-based corrections
    const std::string& bad_pixels_path = "",
    
    // LibRaw Enhanced extensions
    bool use_gpu_acceleration = true
);

// Platform detection functions
bool is_apple_silicon();
// REMOVED: bool is_gpu_available(); - use LibRawWrapper::is_gpu_available() instead
std::vector<std::string> get_metal_device_list();

#endif

} // namespace libraw_enhanced