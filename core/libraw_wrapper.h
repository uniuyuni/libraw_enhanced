#pragma once

#include <array>
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace libraw_enhanced {

#ifdef __arm64__
// Forward declarations for common types (defined in accelerator.h)
struct ProcessingParams;
#endif

// 処理時間計測構造体
struct ProcessingTimes {
  double total_time = 0.0;
  double file_load_time = 0.0;
  double unpack_time = 0.0;
  double demosaic_time = 0.0;
  double color_correction_time = 0.0;
  double gamma_correction_time = 0.0;
  double memory_copy_time = 0.0;
  double postprocess_time = 0.0;
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
  float *data = nullptr;
  // std::vector<uint8_t> data;
  ProcessingTimes timing_info;        // 追加: 処理時間情報
  std::array<float, 12> color_matrix; // 追加: カメラカラーマトリックス (3x4)

  ProcessedImageData() { color_matrix.fill(0.0f); }

  // データサイズの計算
  size_t get_data_size() const {
    return width * height * channels * (bits_per_sample / 8);
  }

  // データの有効性チェック
  bool is_valid() const {
    return width > 0 && height > 0 && channels > 0 && bits_per_sample > 0 &&
           error_code == 0 && data != nullptr;
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
  std::string color_desc;
  bool is_xtrans = false;

  ImageInfo() = default;
};

struct MaximumResult {
  float data_maximum;
  float maximum;
};

// LibRawWrapper クラス宣言
class LibRawWrapper {
public:
  LibRawWrapper();
  ~LibRawWrapper();

  // ファイル操作
  int load_file(const std::string &filename);
  int load_buffer(const std::vector<uint8_t> &buffer);
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

  // rawpy互換processing method (Python dict will be converted to
  // ProcessingParams)
  ProcessedImageData
  process_with_dict(const std::map<std::string, float> &float_params,
                    const std::map<std::string, int> &int_params,
                    const std::map<std::string, bool> &bool_params,
                    const std::map<std::string, std::string> &string_params);

  // 設定メソッド
  void close();

#ifdef __arm64__
  // Metal関連メソッド（実装はlibraw_wrapper.cppで定義）
  void set_processing_params(const ProcessingParams &params);
  void set_gpu_acceleration(bool enable);
  std::string get_device_info() const;

  // Standalone image processing methods (accept numpy float32 array, return new
  // array)
  float get_threshold() const;
  py::array_t<float> recover_highlights_numpy(py::array_t<float> image,
                                              float threshold = -1.f);
  py::array_t<float> tone_mapping_numpy(py::array_t<float> image,
                                        float after_scale = 1.f);
  py::array_t<float>
  enhance_micro_contrast_numpy(py::array_t<float> image, float threshold = -1.f,
                               float strength = 8.f,
                               float target_contrast = 0.06f);
#endif

private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

#ifdef __arm64__
// rawpy完全互換パラメータ変換関数（実装はlibraw_wrapper.cppで定義）
ProcessingParams create_params_from_rawpy_args(
    // Basic parameters
    bool use_camera_wb = true, bool half_size = false,
    bool four_color_rgb = false, int output_bps = 16, int user_flip = -1,

    // Demosaicing parameters
    int demosaic_algorithm = 1, int dcb_iterations = 0,
    bool dcb_enhance = false,

    // Noise reduction parameters
    int fbdd_noise_reduction = 0, float noise_thr = 0.0f,
    int median_filter_passes = 0,

    // White balance parameters
    bool use_auto_wb = false,
    const std::array<float, 4> &user_wb = {1.0f, 1.0f, 1.0f, 1.0f},

    // Color and output parameters
    int output_color = 1,

    // Brightness and exposure parameters
    float bright = 1.0f, bool no_auto_bright = false,
    float auto_bright_thr = 0.01f, float adjust_maximum_thr = 0.75f,

    // Highlight processing
    int highlight_mode = 0,

    // Exposure correction parameters
    float exp_shift = 1.0f, float exp_preserve_highlights = 0.0f,

    // Gamma and scaling
    const std::pair<float, float> &gamma = {0.f, 0.f}, //{2.222f, 4.5f},
    bool no_auto_scale = false,

    // Color correction parameters
    float chromatic_aberration_red = 1.0f,
    float chromatic_aberration_blue = 1.0f,

    // User adjustments
    int user_black = -1, int user_sat = -1,

    // File-based corrections
    const std::string &bad_pixels_path = "",

    // LibRaw Enhanced extensions
    bool use_gpu_acceleration = false, bool preprocess = false);

// Platform detection functions
bool is_apple_silicon();
std::vector<std::string> get_device_list();

#endif

} // namespace libraw_enhanced