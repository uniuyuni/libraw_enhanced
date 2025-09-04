#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "libraw_wrapper.h"

#ifdef __arm64__
#include "accelerator.h"
#include "cpu_accelerator.h"
#include "camera_matrices.h"
#endif

namespace py = pybind11;
using namespace libraw_enhanced;

PYBIND11_MODULE(_core, m) {
    m.doc() = "LibRaw Enhanced Core Module with Metal acceleration";
    
    // バージョン情報
    m.attr("__version__") = VERSION_INFO;
    
    // ProcessingTimes構造体のバインディング
    py::class_<ProcessingTimes>(m, "ProcessingTimes")
        .def(py::init<>())
        .def_readonly("total_time", &ProcessingTimes::total_time)
        .def_readonly("file_load_time", &ProcessingTimes::file_load_time)
        .def_readonly("unpack_time", &ProcessingTimes::unpack_time)
        .def_readonly("demosaic_time", &ProcessingTimes::demosaic_time)
        .def_readonly("color_correction_time", &ProcessingTimes::color_correction_time)
        .def_readonly("gamma_correction_time", &ProcessingTimes::gamma_correction_time)
        .def_readonly("memory_copy_time", &ProcessingTimes::memory_copy_time)
        .def_readonly("postprocess_time", &ProcessingTimes::postprocess_time)
        .def_readonly("demosaic_algorithm_name", &ProcessingTimes::demosaic_algorithm_name);
    
#ifdef __arm64__
    // ProcessingParams構造体のバインディング (完全なrawpy互換)
    py::class_<ProcessingParams>(m, "ProcessingParams")
        .def(py::init<>())
        // Basic processing parameters
        .def_readwrite("use_camera_wb", &ProcessingParams::use_camera_wb)
        .def_readwrite("half_size", &ProcessingParams::half_size)
        .def_readwrite("four_color_rgb", &ProcessingParams::four_color_rgb)
        .def_readwrite("output_bps", &ProcessingParams::output_bps)
        .def_readwrite("user_flip", &ProcessingParams::user_flip)
        
        // Demosaicing parameters
        .def_readwrite("demosaic_algorithm", &ProcessingParams::demosaic_algorithm)
        .def_readwrite("dcb_iterations", &ProcessingParams::dcb_iterations)
        .def_readwrite("dcb_enhance", &ProcessingParams::dcb_enhance)
        
        // Noise reduction parameters
        .def_readwrite("fbdd_noise_reduction", &ProcessingParams::fbdd_noise_reduction)
        .def_readwrite("noise_thr", &ProcessingParams::noise_thr)
        .def_readwrite("median_filter_passes", &ProcessingParams::median_filter_passes)
        
        // White balance parameters
        .def_readwrite("use_auto_wb", &ProcessingParams::use_auto_wb)
        .def_property("user_wb",
            [](ProcessingParams &self) -> py::array_t<float> {
                return py::array_t<float>(4, self.user_wb);
            },
            [](ProcessingParams &self, py::array_t<float> input) {
                py::buffer_info buf = input.request();
                if (buf.size != 4) {
                    throw std::runtime_error("user_wb must have 4 elements");
                }
                std::copy_n(static_cast<float*>(buf.ptr), 4, self.user_wb);
            })
        
        // Color space and output parameters
        .def_readwrite("output_color_space", &ProcessingParams::output_color_space)
        
        // Brightness and exposure parameters  
        .def_readwrite("bright", &ProcessingParams::bright)
        .def_readwrite("no_auto_bright", &ProcessingParams::no_auto_bright)
        .def_readwrite("auto_bright_thr", &ProcessingParams::auto_bright_thr)
        .def_readwrite("adjust_maximum_thr", &ProcessingParams::adjust_maximum_thr)
        
        // Highlight processing
        .def_readwrite("highlight_mode", &ProcessingParams::highlight_mode)
        
        // Exposure correction parameters
        .def_readwrite("exp_shift", &ProcessingParams::exp_shift)
        .def_readwrite("exp_preserve_highlights", &ProcessingParams::exp_preserve_highlights)
        
        // Gamma correction parameters
        .def_readwrite("gamma_power", &ProcessingParams::gamma_power)
        .def_readwrite("gamma_slope", &ProcessingParams::gamma_slope)
        .def_readwrite("no_auto_scale", &ProcessingParams::no_auto_scale)
        
        // Color correction parameters
        .def_readwrite("chromatic_aberration_red", &ProcessingParams::chromatic_aberration_red)
        .def_readwrite("chromatic_aberration_blue", &ProcessingParams::chromatic_aberration_blue)
        
        // User adjustments
        .def_readwrite("user_black", &ProcessingParams::user_black)
        .def_readwrite("user_sat", &ProcessingParams::user_sat)
        
        // File-based corrections
        .def_readwrite("bad_pixels_path", &ProcessingParams::bad_pixels_path)
        
        // Metal-specific settings
        .def_readwrite("use_gpu_acceleration", &ProcessingParams::use_gpu_acceleration)
        .def_property("color_matrix",
            [](ProcessingParams &self) -> py::array_t<float> {
                return py::array_t<float>(9, self.color_matrix);
            },
            [](ProcessingParams &self, py::array_t<float> input) {
                py::buffer_info buf = input.request();
                if (buf.size != 9) {
                    throw std::runtime_error("Color matrix must have 9 elements");
                }
                std::copy_n(static_cast<float*>(buf.ptr), 9, self.color_matrix);
            }
        );

    // ImageBuffer構造体のバインディング  
    py::class_<ImageBuffer>(m, "ImageBuffer")
        .def(py::init<>())
        .def_readonly("width", &ImageBuffer::width)
        .def_readonly("height", &ImageBuffer::height)
        .def_readonly("channels", &ImageBuffer::channels)
        .def_property_readonly("image",
            [](ImageBuffer &self) -> py::object {
                if (self.image == nullptr) {
                    return py::none();
                }
                // Return image data as array
                size_t total_size = self.height * self.width * self.channels;
                return py::array_t<uint16_t>(total_size, reinterpret_cast<uint16_t*>(self.image));
            });

    // ImageInfo構造体のバインディング
    py::class_<ImageInfo>(m, "ImageInfo")
        .def(py::init<>())
        .def_readonly("width", &ImageInfo::width)
        .def_readonly("height", &ImageInfo::height)
        .def_readonly("raw_width", &ImageInfo::raw_width)
        .def_readonly("raw_height", &ImageInfo::raw_height)
        .def_readonly("colors", &ImageInfo::colors)
        .def_readonly("camera_make", &ImageInfo::camera_make)
        .def_readonly("camera_model", &ImageInfo::camera_model)
        .def_property_readonly("cam_mul",
            [](ImageInfo &self) -> py::array_t<float> {
                return py::array_t<float>(4, self.cam_mul);
            })
        .def_property_readonly("pre_mul",
            [](ImageInfo &self) -> py::array_t<float> {
                return py::array_t<float>(4, self.pre_mul);
            });

    // ImageBufferFloat32構造体のバインディング
    py::class_<ImageBufferFloat>(m, "ImageBufferFloat")
        .def(py::init<>())
        .def_readonly("width", &ImageBufferFloat::width)
        .def_readonly("height", &ImageBufferFloat::height)
        .def_readonly("channels", &ImageBufferFloat::channels)
        .def("is_valid", &ImageBufferFloat::is_valid)
        .def_property_readonly("data",
            [](ImageBufferFloat &self) -> py::array_t<float> {
                if (!self.image || self.width == 0 || self.height == 0 || self.channels == 0) {
                    return py::array_t<float>();
                }
                size_t total_pixels = self.width * self.height;
                return py::array_t<float>(
                    {total_pixels, self.channels},
                    {sizeof(float) * self.channels, sizeof(float)},
                    reinterpret_cast<float*>(self.image),
                    py::cast(self) // Keep object alive
                );
            });

    // Camera matrix functions
    py::class_<ColorTransformMatrix>(m, "ColorTransformMatrix")
        .def(py::init<>())
        .def_readonly("valid", &ColorTransformMatrix::valid)
        .def_property_readonly("transform", [](const ColorTransformMatrix& self) -> py::array_t<float> {
            return py::array_t<float>({3, 4}, {4*sizeof(float), sizeof(float)}, &self.transform[0][0]);
        });
        
    m.def("compute_camera_transform", &compute_camera_transform,
          py::arg("make"), py::arg("model"), py::arg("color_space") = 1,
          "Compute camera transformation matrix for given camera and color space");
#else
    // Stub implementation for non-Metal platforms
    m.def("compute_camera_transform", [](const char* make, const char* model, int color_space) -> py::object {
        return py::none();
    }, py::arg("make"), py::arg("model"), py::arg("color_space") = 1);
#endif

    // ProcessedImageData構造体のバインディング
    py::class_<ProcessedImageData>(m, "ProcessedImageData")
        .def(py::init<>())
        .def_readonly("width", &ProcessedImageData::width)
        .def_readonly("height", &ProcessedImageData::height)
        .def_readonly("channels", &ProcessedImageData::channels)
        .def_readonly("bits_per_sample", &ProcessedImageData::bits_per_sample)
        .def_readonly("error_code", &ProcessedImageData::error_code)
        .def_readonly("timing_info", &ProcessedImageData::timing_info)
        .def("get_data_size", &ProcessedImageData::get_data_size)
        .def("is_valid", &ProcessedImageData::is_valid)
        .def("to_numpy", [](ProcessedImageData &self) -> py::array {
            if (!self.is_valid()) {
                throw std::runtime_error("ProcessedImageData is invalid");
            }
            
            size_t height = self.height;
            size_t width = self.width; 
            size_t channels = self.channels;
            
            if(self.bits_per_sample == 8) {
                // Return as uint8 array for 8-bit data  
                py::array_t<uint8_t> array_8({height, width, channels});
                float* in_ptr = self.data;
                uint8_t* out_ptr = static_cast<uint8_t*>(array_8.request().ptr);

                for (size_t i = 0; i < height * width * channels; i++) {
                    *out_ptr++ = *in_ptr++ * 255.0f + 0.5f;
                }
                return array_8;
            
            } else if(self.bits_per_sample == 16) {
                // Return as uint16 array for 16-bit data
                py::array_t<uint16_t> array_16({height, width, channels});
                float* in_ptr = self.data;
                uint16_t* out_ptr = static_cast<uint16_t*>(array_16.request().ptr);

                for (size_t i = 0; i < height * width * channels; i++) {
                    *out_ptr++ = *in_ptr++ * 65535.0f + 0.5f;
                }
                return array_16;

            } else {
                // Return as float32 array for 32-bit data
                return py::array_t<float>(
                    {height, width, channels},
                    {sizeof(float) * width * channels, sizeof(float) * channels, sizeof(float)},
                    self.data
                );
            }
        }, "Convert to numpy array with proper shape and dtype")
        .def_property_readonly("data", 
            [](ProcessedImageData &self) -> py::array_t<float> {
                return py::array_t<float>(
                    {self.width, self.height, self.channels},
                    self.data,
                    py::cast(self)
                );
            }
        );

    // LibRawWrapper クラスのバインディング
    py::class_<LibRawWrapper>(m, "LibRawWrapper")
        .def(py::init<>())
        .def("load_file", &LibRawWrapper::load_file)
        .def("load_buffer", &LibRawWrapper::load_buffer)
        .def("unpack", &LibRawWrapper::unpack)
        .def("process", &LibRawWrapper::process)
        .def("get_processed_image", &LibRawWrapper::get_processed_image)
        .def("get_image_info", &LibRawWrapper::get_image_info)
        .def("get_raw_image_as_numpy", [](LibRawWrapper& self) {
            return self.get_raw_image_as_numpy();
        })
        .def("get_camera_make", &LibRawWrapper::get_camera_make)
        .def("get_camera_model", &LibRawWrapper::get_camera_model)
        .def("process_with_dict", &LibRawWrapper::process_with_dict)
        .def("close", &LibRawWrapper::close)
        .def("set_debug_mode", &LibRawWrapper::set_debug_mode)
#ifdef __arm64__
        .def("set_processing_params", 
            [](LibRawWrapper& self, const ProcessingParams& params) {
                self.set_processing_params(params);
            })
        .def("enable_gpu_acceleration", &LibRawWrapper::enable_gpu_acceleration)
        .def("is_gpu_available", &LibRawWrapper::is_gpu_available)
        .def("get_metal_device_info", &LibRawWrapper::get_metal_device_info)
        // REMOVED: .def("enable_custom_pipeline", &LibRawWrapper::enable_custom_pipeline, - unused custom pipeline feature
#endif
        ;

    // ヘルパー関数のバインディング
#ifdef __arm64__
    m.def("is_apple_silicon", &is_apple_silicon, "Check if running on Apple Silicon");
    // REMOVED: Global is_gpu_available function - use LibRawWrapper.is_gpu_available() instead
    m.def("is_gpu_available", []() { 
        // Return basic Apple Silicon check instead of creating GPU instances
        return is_apple_silicon();
    }, "Check if GPU might be available (basic check)");
    m.def("get_metal_device_list", &get_metal_device_list, "Get list of Metal devices");
    
    m.def("create_params_from_rawpy_args", &create_params_from_rawpy_args,
          "Create ProcessingParams from complete rawpy-compatible arguments",
          // Basic parameters
          py::arg("use_camera_wb") = true,
          py::arg("half_size") = false,
          py::arg("four_color_rgb") = false,
          py::arg("output_bps") = 16,
          py::arg("user_flip") = -1,
          
          // Demosaicing parameters
          py::arg("demosaic_algorithm") = 1,
          py::arg("dcb_iterations") = 0,
          py::arg("dcb_enhance") = false,
          
          // Noise reduction parameters
          py::arg("fbdd_noise_reduction") = 0,
          py::arg("noise_thr") = 0.0f,
          py::arg("median_filter_passes") = 0,
          
          // White balance parameters
          py::arg("use_auto_wb") = false,
          py::arg("user_wb") = std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f},
          
          // Color and output parameters
          py::arg("output_color") = 1,
          
          // Brightness and exposure parameters
          py::arg("bright") = 1.0f,
          py::arg("no_auto_bright") = false,
          py::arg("auto_bright_thr") = 0.01f,
          py::arg("adjust_maximum_thr") = 0.75f,
          
          // Highlight processing
          py::arg("highlight_mode") = 0,
          
          // Exposure correction parameters
          py::arg("exp_shift") = 1.0f,
          py::arg("exp_preserve_highlights") = 0.0f,
          
          // Gamma and scaling
          py::arg("gamma") = std::make_pair(2.222f, 4.5f),
          py::arg("no_auto_scale") = false,
          
          // Color correction parameters
          py::arg("chromatic_aberration_red") = 1.0f,
          py::arg("chromatic_aberration_blue") = 1.0f,
          
          // User adjustments
          py::arg("user_black") = -1,
          py::arg("user_sat") = -1,
          
          // File-based corrections
          py::arg("bad_pixels_path") = std::string(""),
          
          // LibRaw Enhanced extensions
          py::arg("use_gpu_acceleration") = false
    );
#else
    // Metal非対応環境用のスタブ関数
    m.def("is_apple_silicon", []() { return false; }, "Check if running on Apple Silicon (always false on non-Apple platforms)");
    m.def("is_gpu_available", []() { return false; }, "Check if GPU is available (always false on non-Apple platforms)");
    m.def("get_metal_device_list", []() { return std::vector<std::string>(); }, "Get list of Metal devices (empty on non-Apple platforms)");
#endif

    // 定数のバインディング
    py::module constants = m.def_submodule("constants", "Constants and enums");
    
    // LibRAW定数 (libraw/libraw_const.h から)
    constants.attr("LIBRAW_SUCCESS") = py::int_(0);
    constants.attr("LIBRAW_UNSPECIFIED_ERROR") = py::int_(-1);
    constants.attr("LIBRAW_FILE_UNSUPPORTED") = py::int_(-2);
    constants.attr("LIBRAW_REQUEST_FOR_NONEXISTENT_IMAGE") = py::int_(-3);
    constants.attr("LIBRAW_OUT_OF_ORDER_CALL") = py::int_(-4);
    constants.attr("LIBRAW_NO_THUMBNAIL") = py::int_(-5);
    constants.attr("LIBRAW_UNSUPPORTED_THUMBNAIL") = py::int_(-6);
    constants.attr("LIBRAW_INPUT_CLOSED") = py::int_(-7);
    constants.attr("LIBRAW_INSUFFICIENT_MEMORY") = py::int_(-100007);
    constants.attr("LIBRAW_DATA_ERROR") = py::int_(-100008);
    constants.attr("LIBRAW_IO_ERROR") = py::int_(-100009);
    constants.attr("LIBRAW_CANCELLED_BY_CALLBACK") = py::int_(-100010);
    constants.attr("LIBRAW_BAD_CROP") = py::int_(-100011);
    
    // カラースペース定数
    constants.attr("COLORSPACE_RAW") = py::int_(0);
    constants.attr("COLORSPACE_SRGB") = py::int_(1);
    constants.attr("COLORSPACE_ADOBE_RGB") = py::int_(2);
    constants.attr("COLORSPACE_WIDE_GAMUT_RGB") = py::int_(3);
    constants.attr("COLORSPACE_PROPHOTO_RGB") = py::int_(4);
    constants.attr("COLORSPACE_XYZ") = py::int_(5);
    
    // ハイライト処理モード
    constants.attr("HIGHLIGHT_CLIP") = py::int_(0);
    constants.attr("HIGHLIGHT_UNCLIP") = py::int_(1);
    constants.attr("HIGHLIGHT_BLEND") = py::int_(2);
    constants.attr("HIGHLIGHT_REBUILD") = py::int_(3);
    
    // デモザイクアルゴリズム
    constants.attr("DEMOSAIC_LINEAR") = py::int_(0);
    constants.attr("DEMOSAIC_VNG") = py::int_(1);
    constants.attr("DEMOSAIC_PPG") = py::int_(2);
    constants.attr("DEMOSAIC_AHD") = py::int_(3);
    constants.attr("DEMOSAIC_DCB") = py::int_(4);
    constants.attr("DEMOSAIC_DHT") = py::int_(11);
    constants.attr("DEMOSAIC_AAHD") = py::int_(12);
    
    // ユーティリティ関数
    m.def("get_version_info", []() {
        return std::string(VERSION_INFO);
    }, "Get version information");
    
    m.def("get_build_info", []() {
        py::dict info;
        info["version"] = VERSION_INFO;
        info["platform"] = py::str(
#ifdef APPLE_PLATFORM
            "Apple"
#else
            "Generic"
#endif
        );
        info["metal_support"] = py::bool_(
#ifdef __arm64__
            true
#else
            false
#endif
        );
        info["apple_silicon"] = py::bool_(
#ifdef __arm64__
            true
#else
            false
#endif
        );
        return info;
    }, "Get build configuration information");

    // 例外処理
    py::register_exception<std::runtime_error>(m, "LibRawError");
    py::register_exception<std::invalid_argument>(m, "LibRawInvalidArgument");
    py::register_exception<std::out_of_range>(m, "LibRawOutOfRange");
}