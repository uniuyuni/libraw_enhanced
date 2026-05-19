#include "libraw_wrapper.h"
#include "metal/constants.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

// LibRaw ヘッダー
#include <libraw/libraw.h>

#ifdef __arm64__
#include "accelerator.h"
#include "camera_matrices.h"
#endif
#include "metal/shader_common.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace libraw_enhanced {

// ---------------------------------------------------------------------------
// box_filter_normalized — separable sliding-window box blur (float, planar)
//
// #pragma float_control(precise, on) is REQUIRED here.
//
// The build uses -Ofast which enables -fassociative-math, permitting the
// compiler to reorder floating-point operations for SIMD. The sliding-window
// accumulator (sum += / sum -=) is order-sensitive: different SIMD groupings
// of the same sum produce slightly different results due to FP rounding. The
// grouping depends on the data-pointer alignment at runtime; because heap
// allocations are ASLR-randomised between process runs, std::vector<float>
// buffers land at different alignments each run, causing different FP sums
// and therefore different output pixels. Forcing precise FP semantics for
// this function eliminates that run-to-run variation while leaving -Ofast in
// effect everywhere else.
// ---------------------------------------------------------------------------
#pragma float_control(precise, on, push)
static void box_filter_normalized(const std::vector<float> &src,
                                  std::vector<float> &dst,
                                  std::vector<float> &tmp,
                                  size_t width, size_t height, int radius) {
  if (src.size() != width * height || dst.size() != src.size() ||
      tmp.size() != src.size() || width == 0 || height == 0) {
    return;
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t y = 0; y < height; ++y) {
    const size_t row = y * width;
    float sum = 0.0f;
    const size_t first_right = std::min(width - 1, static_cast<size_t>(radius));
    for (size_t x = 0; x <= first_right; ++x) {
      sum += src[row + x];
    }

    for (size_t x = 0; x < width; ++x) {
      const size_t left = x > static_cast<size_t>(radius)
                              ? x - static_cast<size_t>(radius)
                              : 0;
      const size_t right = std::min(width - 1, x + static_cast<size_t>(radius));
      tmp[row + x] = sum / static_cast<float>(right - left + 1);

      const size_t remove_x = x >= static_cast<size_t>(radius)
                                  ? x - static_cast<size_t>(radius)
                                  : width;
      const size_t add_x = x + static_cast<size_t>(radius) + 1;
      if (remove_x < width) {
        sum -= src[row + remove_x];
      }
      if (add_x < width) {
        sum += src[row + add_x];
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t x = 0; x < width; ++x) {
    float sum = 0.0f;
    const size_t first_bottom = std::min(height - 1, static_cast<size_t>(radius));
    for (size_t y = 0; y <= first_bottom; ++y) {
      sum += tmp[y * width + x];
    }

    for (size_t y = 0; y < height; ++y) {
      const size_t top = y > static_cast<size_t>(radius)
                             ? y - static_cast<size_t>(radius)
                             : 0;
      const size_t bottom = std::min(height - 1, y + static_cast<size_t>(radius));
      dst[y * width + x] = sum / static_cast<float>(bottom - top + 1);

      const size_t remove_y = y >= static_cast<size_t>(radius)
                                  ? y - static_cast<size_t>(radius)
                                  : height;
      const size_t add_y = y + static_cast<size_t>(radius) + 1;
      if (remove_y < height) {
        sum -= tmp[remove_y * width + x];
      }
      if (add_y < height) {
        sum += tmp[add_y * width + x];
      }
    }
  }
}
#pragma float_control(pop)

class LibRawWrapper::Impl {
public:
  LibRaw processor;
  ProcessingTimes timing_info; // 処理時間情報

#ifdef __arm64__
  std::unique_ptr<Accelerator> accelerator;

  ProcessingParams current_params;
  std::array<float, 12> last_color_matrix;
#endif

  //===============================================================
  // 高精度タイマーユーティリティ
  //===============================================================
  std::chrono::high_resolution_clock::time_point timer_start;

  void start_timer() {
    timer_start = std::chrono::high_resolution_clock::now();
  }

  double get_elapsed_time() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - timer_start);
    return duration.count() / 1000000.0; // 秒単位で返す
  }

  //===============================================================
  // ゼロを無くす
  //===============================================================

  void remove_zeroes() {
    auto &imgdata = processor.imgdata;

    // === メイン処理: 全ピクセルをスキャンしてゼロ値を補間 ===

    for (unsigned row = 0; row < imgdata.sizes.height; row++) {
      for (unsigned col = 0; col < imgdata.sizes.width; col++) {

        // 現在のピクセル値を取得
        unsigned short &current_pixel =
            imgdata.image[((row) >> imgdata.rawdata.ioparams.shrink) *
                              imgdata.sizes.iwidth +
                          ((col) >> imgdata.rawdata.ioparams.shrink)]
                         [fcol_bayer_native(row, col, imgdata.idata.filters)];

        // ゼロピクセルが見つかった場合のみ処理
        if (current_pixel == 0) {

          // --- 周辺5x5領域から同色ピクセルを探して平均値を計算 ---

          unsigned int pixel_sum = 0;   // 有効ピクセル値の合計
          unsigned int valid_count = 0; // 有効ピクセルの個数

          // 中心から半径2ピクセルの5x5領域をスキャン
          for (int search_row = (int)row - 2; search_row <= (int)row + 2;
               search_row++) {
            for (int search_col = (int)col - 2; search_col <= (int)col + 2;
                 search_col++) {

              // --- 境界チェック ---
              if (search_row < 0 || search_row >= (int)imgdata.sizes.height ||
                  search_col < 0 || search_col >= (int)imgdata.sizes.width) {
                continue; // 画像範囲外はスキップ
              }

              // --- 同色ピクセルかつ非ゼロの場合のみ使用 ---
              unsigned short neighbor_pixel =
                  imgdata
                      .image[((search_row) >> imgdata.rawdata.ioparams.shrink) *
                                 imgdata.sizes.iwidth +
                             ((search_col) >> imgdata.rawdata.ioparams.shrink)]
                            [fcol_bayer_native(search_row, search_col,
                                               imgdata.idata.filters)];

              // 条件チェック:
              // 1. 同じ色チャンネル (R,G,B,G2)
              // 2. ゼロでない値
              if (fcol_bayer_native(search_row, search_col,
                                    imgdata.idata.filters) ==
                      fcol_bayer_native(row, col, imgdata.idata.filters) &&
                  neighbor_pixel != 0) {
                pixel_sum += neighbor_pixel;
                valid_count++;
              }
            }
          }

          // --- 補間値の計算と適用 ---
          if (valid_count > 0) {
            // 平均値を計算してゼロピクセルを置き換え
            current_pixel = pixel_sum / valid_count;
          }
          // 注意: 周辺にも同色の有効ピクセルがない場合は0のまま残る
        }
      }
    }
  }

  //===============================================================
  // ブラックレベル調整
  //===============================================================

  void adjust_bl() {
    // === ステップ1: ユーザー指定のブラックレベルを適用 ===
    bool user_values_applied = apply_user_black_levels();

    // === ステップ2: 2D配列形式のブラックレベルを処理 ===
    if (has_2d_black_level_pattern()) {
      process_2d_black_level_pattern();
    }

    // === ステップ3: 基本ブラックレベル配列から共通部分を抽出 ===
    extract_common_black_level_from_basic_array();

    // === ステップ4: 2D配列部分から共通部分を抽出 ===
    extract_common_black_level_from_2d_array();

    // === ステップ5: 最終的な調整 ===
    finalize_black_levels();
  }

  // ユーザー指定のブラックレベル値を適用
  bool apply_user_black_levels() {
    auto &imgdata = processor.imgdata;
    bool applied = false;

    // 全体のブラックレベルがユーザー指定されている場合
    if (imgdata.params.user_black >= 0) {
      imgdata.color.black = imgdata.params.user_black;
      applied = true;
    }

    // 色別のブラックレベルがユーザー指定されている場合
    for (int i = 0; i < 4; i++) {
      if (imgdata.params.user_cblack[i] > -1000000) {
        imgdata.color.cblack[i] = imgdata.params.user_cblack[i];
        applied = true;
      }
    }

    // ユーザー指定値が適用された場合、2D配列パターンをリセット
    if (applied) {
      imgdata.color.cblack[4] = 0; // 2D配列の幅
      imgdata.color.cblack[5] = 0; // 2D配列の高さ
    }

    return applied;
  }

  // 2D配列形式のブラックレベルパターンがあるかチェック
  bool has_2d_black_level_pattern() {
    auto &imgdata = processor.imgdata;
    return (imgdata.idata.filters > 1000 &&
            (imgdata.color.cblack[4] + 1) / 2 == 1 &&
            (imgdata.color.cblack[5] + 1) / 2 == 1);
  }

  // Bayerパターン用の2D配列処理
  void process_2d_black_level_pattern() {
    auto &imgdata = processor.imgdata;
    // 各位置の色を特定
    int color_at_position[4];
    int green_count = 0;
    int last_green_pos = -1;

    for (int pos = 0; pos < 4; pos++) {
      color_at_position[pos] = fcol_bayer_native(
          pos / 2, pos % 2, imgdata.idata.filters); // FC: Filter Color関数

      if (color_at_position[pos] == 1) { // Green
        green_count++;
        last_green_pos = pos;
      }
    }

    // 複数のGreenがある場合、最後のものを別色として扱う
    if (green_count > 1 && last_green_pos >= 0) {
      color_at_position[last_green_pos] = 3; // 第2のGreen
    }

    // 2D配列の値を対応する色のブラックレベルに加算
    for (int pos = 0; pos < 4; pos++) {
      int row = pos / 2;
      int col = pos % 2;
      int array_index =
          6 + (row % imgdata.color.cblack[4]) * imgdata.color.cblack[5] +
          (col % imgdata.color.cblack[5]);

      imgdata.color.cblack[color_at_position[pos]] +=
          imgdata.color.cblack[array_index];
    }

    // 2D配列パターンを無効化
    imgdata.color.cblack[4] = 0;
    imgdata.color.cblack[5] = 0;
  }

  // Fuji RAF DNG形式の特別処理
  void process_fuji_raf_format() {
    auto &imgdata = processor.imgdata;
    // 単純に全色に同じ値を加算
    for (int c = 0; c < 4; c++) {
      imgdata.color.cblack[c] += imgdata.color.cblack[6];
    }

    imgdata.color.cblack[4] = 0;
    imgdata.color.cblack[5] = 0;
  }

  // 基本ブラックレベル配列（cblack[0-3]）から共通部分を抽出
  void extract_common_black_level_from_basic_array() {
    auto &imgdata = processor.imgdata;
    // Fuji RAF DNG形式の特別処理
    if (imgdata.idata.filters <= 1000 && imgdata.color.cblack[4] == 1 &&
        imgdata.color.cblack[5] == 1) {
      process_fuji_raf_format();
    }

    // 4色の最小値を見つける
    int min_black = imgdata.color.cblack[3];
    for (int c = 0; c < 3; c++) {
      if (min_black > imgdata.color.cblack[c]) {
        min_black = imgdata.color.cblack[c];
      }
    }

    // 各色から最小値を引いて、共通部分をblackに移す
    for (int c = 0; c < 4; c++) {
      imgdata.color.cblack[c] -= min_black;
    }
    imgdata.color.black += min_black;
  }

  // 2D配列部分（cblack[6+]）から共通部分を抽出
  void extract_common_black_level_from_2d_array() {
    auto &imgdata = processor.imgdata;
    if (!imgdata.color.cblack[4] || !imgdata.color.cblack[5]) {
      return; // 2D配列がない場合は何もしない
    }

    int array_size = imgdata.color.cblack[4] * imgdata.color.cblack[5];

    // 2D配列の最小値を見つける
    int min_value = imgdata.color.cblack[6];
    for (int i = 1; i < array_size; i++) {
      if (min_value > imgdata.color.cblack[6 + i]) {
        min_value = imgdata.color.cblack[6 + i];
      }
    }

    // 各要素から最小値を引く
    int non_zero_count = 0;
    for (int i = 0; i < array_size; i++) {
      imgdata.color.cblack[6 + i] -= min_value;
      if (imgdata.color.cblack[6 + i] != 0) {
        non_zero_count++;
      }
    }

    // 共通部分をblackに移す
    imgdata.color.black += min_value;

    // すべてが0になった場合、2D配列を無効化
    if (non_zero_count == 0) {
      imgdata.color.cblack[4] = 0;
      imgdata.color.cblack[5] = 0;
    }
  }

  // 最終的なブラックレベルの調整
  void finalize_black_levels() {
    auto &imgdata = processor.imgdata;
    // 各色のブラックレベルに共通ブラックレベルを加算
    for (int c = 0; c < 4; c++) {
      imgdata.color.cblack[c] += imgdata.color.black;
    }
  }

  //===============================================================
  // LibRaw-compatible black level correction (internal function)
  //===============================================================
  void apply_black_level_correction(ImageBuffer &raw_buffer) {
    std::cout << "📋 Apply black level subtraction: "
              << processor.imgdata.color.black << std::endl;

    // Extract black level data from LibRaw
    int cblack[4];
    for (int i = 0; i < 4; i++) {
      cblack[i] = processor.imgdata.color.cblack[i];
    }

    size_t total_pixels = raw_buffer.width * raw_buffer.height;

    // Check if we have positional black level data (cblack[4] and cblack[5])
    if (processor.imgdata.color.cblack[4] &&
        processor.imgdata.color.cblack[5]) {
      // Complex black level with position-dependent correction
      std::cout << "📋 Using position-dependent black level correction"
                << std::endl;

      for (size_t q = 0; q < total_pixels; q++) {
        for (int c = 0; c < 4; c++) {
          int val = raw_buffer.image[q][c];

          // Position-dependent black level correction (LibRaw formula)
          int row = q / raw_buffer.width;
          int col = q % raw_buffer.width;
          int pos_black =
              processor.imgdata.color
                  .cblack[6 +
                          (row % processor.imgdata.color.cblack[4]) *
                              processor.imgdata.color.cblack[5] +
                          (col % processor.imgdata.color.cblack[5])];

          val -= pos_black;
          val -= cblack[c];

          // Clamp to valid range
          if (val < 0)
            val = 0;
          if (val > (int)processor.imgdata.color.maximum)
            val = processor.imgdata.color.maximum;

          raw_buffer.image[q][c] = val;
        }
      }

    } else if (cblack[0] || cblack[1] || cblack[2] || cblack[3]) {
      // Simple per-channel black level correction
      std::cout << "📋 Using per-channel black level correction: " << cblack[0]
                << "," << cblack[1] << "," << cblack[2] << "," << cblack[3]
                << std::endl;

      for (size_t q = 0; q < total_pixels; q++) {
        for (int c = 0; c < 4; c++) {
          int val = raw_buffer.image[q][c];
          val -= cblack[c];

          // Clamp to valid range
          if (val < 0)
            val = 0;
          if (val > (int)processor.imgdata.color.maximum)
            val = processor.imgdata.color.maximum;

          raw_buffer.image[q][c] = val;
        }
      }

    } else {
      // Fallback: simple global black level
      std::cout << "📋 Using global black level: "
                << processor.imgdata.color.black << std::endl;

      for (size_t q = 0; q < total_pixels; q++) {
        for (int c = 0; c < 4; c++) {
          int val = raw_buffer.image[q][c];
          val -= processor.imgdata.color.black;

          // Clamp to valid range
          if (val < 0)
            val = 0;
          if (val > (int)processor.imgdata.color.maximum)
            val = processor.imgdata.color.maximum;

          raw_buffer.image[q][c] = val;
        }
      }
    }

    std::cout << "✅ Black level subtraction completed for " << total_pixels
              << " pixels" << std::endl;
  }

  //===============================================================
  // LibRaw-compatible green matching (internal function)
  //===============================================================
  void apply_green_matching(ImageBuffer &raw_buffer, uint32_t filters) {
    std::cout << "📋 Apply green matching for G1/G2 equilibration" << std::endl;

    // Skip for XTrans sensors (only for Bayer)
    if (filters == FILTERS_XTRANS) {
      std::cout << "📋 Skipping green matching for XTrans sensor" << std::endl;
      return;
    }

    const int margin = 3;
    const float thr = 0.01f;
    const int width = raw_buffer.width;
    const int height = raw_buffer.height;
    const int maximum =
        processor.imgdata.color.maximum; // Dynamic maximum value

    // Find G2 pixel position in Bayer pattern
    int oj = 2, oi = 2;

    // In RGGB Bayer pattern: R(0,0), G1(0,1), G2(1,0), B(1,1)
    // We look for G2 positions which are typically at (1,0) pattern
    // For more robust detection, we assume standard RGGB and start from (1,0)
    oj = 1; // G2 row offset
    oi = 0; // G2 col offset

    // Create working copy of image data
    uint16_t(*img)[4] =
        (uint16_t(*)[4])calloc(height * width, sizeof(uint16_t[4]));
    if (!img) {
      std::cerr << "❌ Failed to allocate memory for green matching"
                << std::endl;
      return;
    }

    // Copy original data
    memcpy(img, raw_buffer.image, height * width * sizeof(uint16_t[4]));

    int processed_pixels = 0;

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = oj + 2; j < height - margin; j += 2) {
      for (int i = oi + 2; i < width - margin; i += 2) {
        // Ensure we don't go out of bounds
        if (j - 1 < 0 || j + 1 >= height || i - 1 < 0 || i + 1 >= width ||
            j - 2 < 0 || j + 2 >= height || i - 2 < 0 || i + 2 >= width) {
          continue;
        }

        // Get surrounding G1 pixels (channel[1])
        int o1_1 = img[(j - 1) * width + i - 1][1];
        int o1_2 = img[(j - 1) * width + i + 1][1];
        int o1_3 = img[(j + 1) * width + i - 1][1];
        int o1_4 = img[(j + 1) * width + i + 1][1];

        // Get surrounding G2 pixels (channel[3])
        int o2_1 = img[(j - 2) * width + i][3];
        int o2_2 = img[(j + 2) * width + i][3];
        int o2_3 = img[j * width + i - 2][3];
        int o2_4 = img[j * width + i + 2][3];

        // Calculate averages
        double m1 = (o1_1 + o1_2 + o1_3 + o1_4) / 4.0;
        double m2 = (o2_1 + o2_2 + o2_3 + o2_4) / 4.0;

        // Calculate consistency (variation) in each group
        double c1 = (abs(o1_1 - o1_2) + abs(o1_1 - o1_3) + abs(o1_1 - o1_4) +
                     abs(o1_2 - o1_3) + abs(o1_3 - o1_4) + abs(o1_2 - o1_4)) /
                    6.0;
        double c2 = (abs(o2_1 - o2_2) + abs(o2_1 - o2_3) + abs(o2_1 - o2_4) +
                     abs(o2_2 - o2_3) + abs(o2_3 - o2_4) + abs(o2_2 - o2_4)) /
                    6.0;

        // Apply correction only in flat areas and non-saturated pixels
        if ((img[j * width + i][3] < maximum * 0.95) && (c1 < maximum * thr) &&
            (c2 < maximum * thr) && (m2 > 0.1)) { // Avoid division by zero

          float correction = raw_buffer.image[j * width + i][3] * m1 / m2;
          raw_buffer.image[j * width + i][3] =
              correction > maximum ? maximum : (uint16_t)correction;
          processed_pixels++;
        }
      }
    }

    free(img);
    std::cout << "✅ Green matching completed: processed " << processed_pixels
              << " G2 pixels" << std::endl;
  }

  //===============================================================
  // カラースケール処理
  //===============================================================

  inline uint32_t cfa_channel_at(size_t row, size_t col) const {
    const auto &idata = processor.imgdata.idata;
    if (idata.filters == FILTERS_XTRANS) {
      return fcol_xtrans(static_cast<int>(row), static_cast<int>(col),
                         idata.xtrans);
    }
    return fcol_bayer_native(static_cast<int>(row), static_cast<int>(col),
                             idata.filters);
  }


  void scale_colors(ImageBuffer &raw_buffer, float scale_mul[4]) {
    auto &imgdata = processor.imgdata;

    // 変数宣言
    unsigned bottom, right, size, row, col, ur, uc, i, x, y, c, sum[8];
    int val;
    double dsum[8], dmin, dmax;
    float fr, fc;
    ushort *img = 0, *pix;

    // ========================================
    // 1. ユーザー指定の乗数設定
    // ========================================
    if (imgdata.params.user_mul[0]) {
      memcpy(imgdata.color.pre_mul, imgdata.params.user_mul,
             sizeof(imgdata.color.pre_mul));
    }

    // ========================================
    // 2. 自動ホワイトバランス計算
    // ========================================
    bool should_use_auto_wb =
        imgdata.params.use_auto_wb ||
        (imgdata.params.use_camera_wb &&
         (imgdata.color.cam_mul[0] <
              -0.5 // LibRaw 0.19以前: cam_mul[0]が-1の時のみ自動に戻る
          || (imgdata.color.cam_mul[0] <=
                  0.00001f // 新しいデフォルト:
                           // cam_mulがメタデータから解析されない場合
              && !(imgdata.rawparams.options &
                   LIBRAW_RAWOPTIONS_CAMERAWB_FALLBACK_TO_DAYLIGHT))));

    if (should_use_auto_wb) {
      // 各チャンネルのpノルムを計算
      double p = 1.0; // 6.0
      double norm[4] = {0.0, 0.0, 0.0, 0.0};
      double count[4] = {0.0, 0.0, 0.0, 0.0};
      uint32_t cc;

      for (int row = 0; row < raw_buffer.height; ++row) {
        for (int col = 0; col < raw_buffer.width; ++col) {
          if (imgdata.idata.filters == FILTERS_XTRANS) {
            cc = fcol_xtrans(row, col, imgdata.idata.xtrans);
          } else {
            cc = fcol_bayer_native(row, col, imgdata.idata.filters);
          }
          size_t idx = row * imgdata.sizes.iwidth + col;
          norm[cc] += std::pow(raw_buffer.image[idx][cc], p);
          count[cc] += 1;
        }
      }
      norm[1] += norm[3];
      count[1] += count[3];

      // p乗平均を計算
      for (cc = 0; cc < 3; ++cc) {
        norm[cc] = std::pow(norm[cc] / count[cc], 1.0 / p);
      }

      // ゲインを計算 (Gチャンネルを基準とする場合)
      imgdata.color.pre_mul[0] = norm[1] / norm[0];
      imgdata.color.pre_mul[1] = 1.0;
      imgdata.color.pre_mul[2] = norm[1] / norm[2];
      imgdata.color.pre_mul[3] = 1.0;
      /*
                  // RGBチャンネルを表すenum
                  enum ColorChannel {
                      RED = 0,
                      GREEN = 1,
                      BLUE = 2
                  };

                  // 画像統計情報を保持する構造体
                  struct ImageStats {
                      double min[3] = {0.0, 0.0, 0.0};
                      double max[3] = {0.0, 0.0, 0.0};
                      double mean[3] = {0.0, 0.0, 0.0};
                      int count[3] = {0, 0, 0};
                  };

                  // 画像統計を計算
                  ImageStats stats;
                  double sum[3] = {0.0, 0.0, 0.0};

                  // 各ピクセルを処理
      #ifdef _OPENMP
      //            #pragma omp parallel for collapse(2)
      #endif
                  for (size_t row = 0; row < imgdata.sizes.height; ++row) {
                      for (size_t col = 0; col < imgdata.sizes.width; ++col) {
                          int pixelIndex = row * imgdata.sizes.width + col;

                          // カラーフィルタ配列からカラーチャンネルを取得
                          int colorIndex;
                          if (imgdata.idata.filters == FILTERS_XTRANS) {
                              colorIndex = fcol_xtrans(row, col,
      imgdata.idata.xtrans); } else { colorIndex = fcol_bayer(row, col,
      imgdata.idata.filters);
                          }

                          unsigned short pixelValue =
      imgdata.image[pixelIndex][colorIndex];

                          // カラーチャンネルに応じて統計情報を更新
                          if (colorIndex >= 0 && colorIndex < 3) {
                              double value = static_cast<double>(pixelValue);

                              // 最小値更新
                              if (stats.count[colorIndex] == 0 || value <
      stats.min[colorIndex]) { stats.min[colorIndex] = value;
                              }

                              // 最大値更新
                              if (stats.count[colorIndex] == 0 || value >
      stats.max[colorIndex]) { stats.max[colorIndex] = value;
                              }

                              // 合計値更新
                              sum[colorIndex] += value;
                              stats.count[colorIndex]++;
                          }
                      }
                  }

                  // 平均値計算
                  for (int i = 0; i < 3; ++i) {
                      if (stats.count[i] > 0) {
                          stats.mean[i] = sum[i] / stats.count[i];
                      }
                  }

                  // グリーンチャンネルを基準としてホワイトバランス係数を計算
                  imgdata.color.pre_mul[RED] =
      static_cast<float>(stats.mean[GREEN] / stats.mean[RED]);
                  imgdata.color.pre_mul[GREEN] = 1.0f; //
      グリーンは基準なので1.0 imgdata.color.pre_mul[BLUE] =
      static_cast<float>(stats.mean[GREEN] / stats.mean[BLUE]);
                  imgdata.color.pre_mul[GREEN+2] = 1.0f;
      */
    }

    // ========================================
    // 3. カメラホワイトバランス処理
    // ========================================
    if (imgdata.params.use_camera_wb && imgdata.color.cam_mul[0] > 0.00001f) {
      memset(sum, 0, sizeof(sum));

      // ホワイトポイントサンプル処理
      for (row = 0; row < 8; row++) {
        for (col = 0; col < 8; col++) {
          c = fcol_bayer_native(row, col, imgdata.idata.filters);
          if ((val = imgdata.color.white[row][col] - imgdata.color.cblack[c]) >
              0) {
            sum[c] += val;
          }
          sum[c + 4]++;
        }
      }

      if (imgdata.color.as_shot_wb_applied) {
        // Nikon sRAW: カメラWBが既に適用済み
        imgdata.color.pre_mul[0] = imgdata.color.pre_mul[1] =
            imgdata.color.pre_mul[2] = imgdata.color.pre_mul[3] = 1.0;
      } else if (sum[0] && sum[1] && sum[2] && sum[3]) {
        // 全色のデータがある場合
        for (c = 0; c < 4; c++) {
          imgdata.color.pre_mul[c] = (float)sum[c + 4] / sum[c];
        }
      } else if (imgdata.color.cam_mul[0] > 0.00001f &&
                 imgdata.color.cam_mul[2] > 0.00001f) {
        // カメラ乗数を直接使用
        memcpy(imgdata.color.pre_mul, imgdata.color.cam_mul,
               sizeof(imgdata.color.pre_mul));
      } else {
        // 警告: カメラWBが不正
        imgdata.process_warnings |= LIBRAW_WARN_BAD_CAMERA_WB;
      }
    }

    // ========================================
    // 4. Nikon sRAW特別処理（昼光設定）
    // ========================================
    bool is_nikon_sraw_daylight =
        imgdata.color.as_shot_wb_applied && !imgdata.params.use_camera_wb &&
        !imgdata.params.use_auto_wb && imgdata.color.cam_mul[0] > 0.00001f &&
        imgdata.color.cam_mul[1] > 0.00001f &&
        imgdata.color.cam_mul[2] > 0.00001f;

    if (is_nikon_sraw_daylight) {
      for (c = 0; c < 3; c++) {
        imgdata.color.pre_mul[c] /= imgdata.color.cam_mul[c];
      }
    }

    // ========================================
    // 5. pre_mul値の正規化
    // ========================================
    if (imgdata.color.pre_mul[1] == 0) {
      imgdata.color.pre_mul[1] = 1;
    }
    if (imgdata.color.pre_mul[3] == 0) {
      imgdata.color.pre_mul[3] =
          imgdata.idata.colors < 4 ? imgdata.color.pre_mul[1] : 1;
    }

    // ========================================
    // 6. ウェーブレットノイズ除去（オプション）
    // ========================================
    /*
            if (threshold) {
                wavelet_denoise();
            }
    */
    // ========================================
    // 7. スケーリング係数計算
    // ========================================
    imgdata.color.maximum -= imgdata.color.black;

    if (!should_use_auto_wb) {

      // 最小・最大乗数値を検索
      for (dmin = std::numeric_limits<double>::max(), c = 0; c < 4; c++) {
        if (dmin > imgdata.color.pre_mul[c]) {
          dmin = imgdata.color.pre_mul[c];
        }
      }

      // スケーリング乗数計算
      if (dmin > 0.00001 && imgdata.color.maximum > 0) {
        for (c = 0; c < 4; c++) {
          // scale_mul[c] = (imgdata.color.pre_mul[c] /= dmax) * 65535.0 /
          // imgdata.color.maximum;
          scale_mul[c] = imgdata.color.pre_mul[c] / dmin;
        }
      } else {
        for (c = 0; c < 4; c++) {
          scale_mul[c] = 1.0;
        }
      }
    } else {
      for (c = 0; c < 4; c++) {
        scale_mul[c] = imgdata.color.pre_mul[c];
      }
    }

    // ========================================
    // 8. ブラックレベル調整
    // ========================================
    if (imgdata.idata.filters > 1000 &&
        (imgdata.color.cblack[4] + 1) / 2 == 1 &&
        (imgdata.color.cblack[5] + 1) / 2 == 1) {
      for (c = 0; c < 4; c++) {
        imgdata.color
            .cblack[fcol_bayer_native(c / 2, c % 2, imgdata.idata.filters)] +=
            imgdata.color.cblack[6 +
                                 c / 2 % imgdata.color.cblack[4] *
                                     imgdata.color.cblack[5] +
                                 c % 2 % imgdata.color.cblack[5]];
      }
      imgdata.color.cblack[4] = imgdata.color.cblack[5] = 0;
    }

    // ========================================
    // 9. 倍率色収差補正（R/Bチャンネル）
    // ========================================
    //
    // The previous auto-estimation path for RAW-space radial CA has been
    // retired (it suffered from a sparse-bilinear bias on CFA planes and
    // produced visible radial artifacts on real images — e.g. sky banding
    // and magenta→green redistribution on X-Pro2 Yokohama at scale ≈0.998).
    //
    // Lateral CA correction now lives in post-demosaic dense RGB via the
    // Lucas-Kanade channel registration step
    // (ProcessingParams::lateral_ca_correction).
    //
    // The legacy per-channel radial scaling loop below remains for callers
    // that explicitly pass non-unity chromatic_aberration values.  When
    // chromatic_aberration is left at the default (1.0, 1.0) this block is
    // a no-op.
    size = imgdata.sizes.iheight * imgdata.sizes.iwidth;

    if ((imgdata.params.aber[0] != 1.0f || imgdata.params.aber[2] != 1.0f) &&
        imgdata.idata.colors == 3 && raw_buffer.is_valid() &&
        raw_buffer.width > 1 && raw_buffer.height > 1) {
      const float center_y = static_cast<float>(raw_buffer.height) * 0.5f;
      const float center_x = static_cast<float>(raw_buffer.width) * 0.5f;

      for (c = 0; c < 4; c += 2) {
        const float scale = imgdata.params.aber[c];
        if (std::fabs(scale - 1.0f) < 1e-6f || scale <= 0.0f) {
          continue;
        }

        std::vector<ushort> channel_plane(size);
        for (i = 0; i < size; i++) {
          channel_plane[i] = raw_buffer.image[i][c];
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int row_i = 0; row_i < static_cast<int>(raw_buffer.height); row_i++) {
          const size_t row_local = static_cast<size_t>(row_i);
          for (size_t col_local = 0; col_local < raw_buffer.width; col_local++) {
            const float src_col_f =
                (static_cast<float>(col_local) - center_x) * scale + center_x;
            const float src_row_local =
                (static_cast<float>(row_local) - center_y) * scale + center_y;
            const unsigned ur_local = static_cast<unsigned>(src_row_local);
            if (ur_local >= raw_buffer.height - 1) {
              continue;
            }
            const float fr_local = src_row_local - static_cast<float>(ur_local);
            const unsigned uc_local = static_cast<unsigned>(src_col_f);
            if (uc_local >= raw_buffer.width - 1) {
              continue;
            }
            const float fc_local = src_col_f - static_cast<float>(uc_local);

            const ushort *pix_local =
                channel_plane.data() + ur_local * raw_buffer.width + uc_local;
            const float corrected =
                (pix_local[0] * (1.0f - fc_local) + pix_local[1] * fc_local) *
                    (1.0f - fr_local) +
                (pix_local[raw_buffer.width] * (1.0f - fc_local) +
                 pix_local[raw_buffer.width + 1] * fc_local) *
                    fr_local;

            const float clamped =
                std::min(65535.0f, std::max(0.0f, corrected));
            raw_buffer.image[row_local * raw_buffer.width + col_local][c] =
                static_cast<ushort>(clamped);
          }
        }
      }
    }
  }

  //===============================================================
  // LibRaw-compatible adjust_maximum implementation
  //===============================================================

  void adjust_maximum0(const ImageBuffer &raw_buffer, float threshold) {
    std::cout << "📋 Apply adjust_maximum for dynamic maximum value adjustment "
                 "(threshold: "
              << threshold << ")" << std::endl;

    // Early return if threshold is too small (LibRaw compatibility)
    if (threshold < 0.00001f) {
      std::cout << "📋 Skipping adjust_maximum: threshold too small ("
                << threshold << ")" << std::endl;
      return;
    }

    // Use default threshold if too large (LibRaw compatibility)
    float auto_threshold = threshold;
    if (threshold > 0.99999f) {
      auto_threshold = 0.75f; // LIBRAW_DEFAULT_ADJUST_MAXIMUM_THRESHOLD
      std::cout << "📋 Using default threshold: " << auto_threshold
                << std::endl;
    }

    // Calculate data_maximum if not already set (LibRaw compatibility)
    uint16_t real_max = processor.imgdata.color.data_maximum;
    if (real_max == 0 && raw_buffer.image != nullptr) {
      std::cout << "📋 Calculating data_maximum by scanning image data..."
                << std::endl;

      size_t total_pixels = raw_buffer.width * raw_buffer.height;
      uint16_t max_value = 0;

      for (size_t i = 0; i < total_pixels; i++) {
        for (int c = 0; c < 4; c++) {
          uint16_t val = raw_buffer.image[i][c];
          if (val > max_value) {
            max_value = val;
          }
        }
      }

      real_max = max_value;
      processor.imgdata.color.data_maximum = real_max;
      std::cout << "📋 Calculated data_maximum: " << real_max << std::endl;
    }

    uint16_t current_max = processor.imgdata.color.maximum;
    std::cout << "📋 Current maximum: " << current_max
              << ", data_maximum: " << real_max << std::endl;

    // Apply LibRaw's adjust_maximum logic
    if (real_max > 0 && real_max < current_max &&
        real_max > current_max * auto_threshold) {

      processor.imgdata.color.maximum = real_max;
      std::cout << "✅ Adjusted maximum value: " << current_max << " → "
                << real_max << " (threshold: " << auto_threshold << ")"
                << std::endl;
    } else {
      std::cout << "📋 No adjustment needed - conditions not met" << std::endl;
      std::cout << "   real_max > 0: " << (real_max > 0) << std::endl;
      std::cout << "   real_max < current_max: " << (real_max < current_max)
                << std::endl;
      std::cout << "   real_max > current_max * threshold: "
                << (real_max > current_max * auto_threshold) << " (" << real_max
                << " > " << (current_max * auto_threshold) << ")" << std::endl;
    }
  }

  MaximumResult adjust_maximum(const ImageBufferFloat &rgb_buffer,
                               float threshold) {
    std::cout << "📋 Apply adjust_maximum for dynamic maximum value adjustment "
                 "(threshold: "
              << threshold << ")" << std::endl;

    MaximumResult result = {
        (float)processor.imgdata.color.data_maximum,
        (float)processor.imgdata.color.maximum,
    };

    // Early return if threshold is too small (LibRaw compatibility)
    if (threshold < 0.00001f) {
      std::cout << "📋 Skipping adjust_maximum: threshold too small ("
                << threshold << ")" << std::endl;
      return result;
    }

    // Use default threshold if too large (LibRaw compatibility)
    float auto_threshold = threshold;
    if (threshold > 0.99999f) {
      auto_threshold = 0.75f; // LIBRAW_DEFAULT_ADJUST_MAXIMUM_THRESHOLD
      std::cout << "📋 Using default threshold: " << auto_threshold
                << std::endl;
    }

    // Calculate data_maximum if not already set (LibRaw compatibility)
    float real_max = result.data_maximum;
    if (real_max == 0.f && rgb_buffer.image != nullptr) {
      std::cout << "📋 Calculating data_maximum by scanning image data..."
                << std::endl;

      size_t total_pixels = rgb_buffer.width * rgb_buffer.height;
      float max_value = 0.f;

#ifdef _OPENMP
#pragma omp parallel for reduction(max: max_value)
#endif
      for (size_t row = 0; row < rgb_buffer.height; ++row) {
        for (size_t col = 0; col < rgb_buffer.width; ++col) {
          size_t idx = row * rgb_buffer.width + col;
          uint32_t c = fcol_bayer(row, col, processor.imgdata.idata.filters);
          float val = rgb_buffer.image[idx][c];
          if (val > max_value) {
            max_value = val;
          }
        }
      }

      real_max = max_value;
      result.data_maximum = real_max;
      processor.imgdata.color.data_maximum = real_max;
      std::cout << "📋 Calculated data_maximum: " << real_max << std::endl;
    }

    float current_max = result.maximum;
    std::cout << "📋 Current maximum: " << current_max
              << ", data_maximum: " << real_max << std::endl;

    // Apply LibRaw's adjust_maximum logic
    if (real_max > 0 && real_max < current_max &&
        real_max > current_max * auto_threshold) {

      result.maximum = real_max;
      processor.imgdata.color.maximum = real_max;
      std::cout << "✅ Adjusted maximum value: " << current_max << " → "
                << real_max << " (threshold: " << auto_threshold << ")"
                << std::endl;
    } else {
      std::cout << "📋 No adjustment needed - conditions not met" << std::endl;
      std::cout << "   real_max > 0: " << (real_max > 0) << std::endl;
      std::cout << "   real_max < current_max: " << (real_max < current_max)
                << std::endl;
      std::cout << "   real_max > current_max * threshold: "
                << (real_max > current_max * auto_threshold) << " (" << real_max
                << " > " << (current_max * auto_threshold) << ")" << std::endl;
    }

    return result;
  }

  //===============================================================
  // LibRaw recover_highlights equivalent for float32 processing
  //===============================================================

  bool recover_highlights(ImageBufferFloat &rgb_buffer,
                          float saturation_threshold,
                          float *highlight_mask_out = nullptr,
                          const float *wb_rgb = nullptr) {
    std::cout << "🔧 Starting highlight recovery... sat: "
              << saturation_threshold << std::endl;

    const size_t width = rgb_buffer.width;
    const size_t height = rgb_buffer.height;
    const size_t channels = rgb_buffer.channels;

    // Soft highlight mask, two factors combined:
    //
    //   1) smoothstep of the *green* channel around the sensor saturation
    //      point.  Demosaic normalises by `maximum_result.maximum`, so a
    //      pixel where the raw green hit sensor saturation lands at exactly
    //      1.0 here.  R and B are inflated by WB multipliers (often
    //      1.5–2.0×) so `max(R,G,B) >= 1.0` would catch bright reds/blues
    //      that are nowhere near saturation; G has WB_G ≈ 1 and is the
    //      cleanest proxy for "this pixel hit the sensor ceiling".
    //      `saturation_threshold` (= maximum/data_maximum) is *not* a stable
    //      anchor — it varies with WB skew across cameras/scenes — so we use
    //      fixed edges in the normalised post-demosaic space.
    //
    //   2) Neutrality weight = 1 − smoothstep(chroma_n, t_lo, t_hi).
    //      When `wb_rgb` is supplied we virtually undo the per-channel WB
    //      multipliers before computing chroma — that gives a tight
    //      raw-space measurement where a *true* neutral white reads
    //      chroma_n ≈ 0 regardless of WB skew (post-WB whites would
    //      otherwise sit at chroma ≈ 0.3–0.5 just from the WB stretch and
    //      overlap with genuinely tinted highlights).  Edges of (0.10, 0.30)
    //      keep slight tints (chroma_n < 0.1, e.g. tiny gel cast) at full
    //      strength and cut off cleanly past chroma_n ≈ 0.25.
    //      When `wb_rgb` is null we fall back to the previous post-WB
    //      chroma with the looser (0.50, 0.70) edges — used by the
    //      standalone `recover_highlights_numpy` path which doesn't thread
    //      WB through.
    //
    // Computed from the *input* pixel values so the mask captures the
    // original highlight extent before this function rewrites any channels.
    const float mask_t_hi = 1.00f;
    const float mask_t_lo = 0.95f;
    const float mask_inv  = 1.f / std::max(1e-6f, mask_t_hi - mask_t_lo);

    const float chroma_t_lo = (wb_rgb != nullptr) ? 0.20f : 0.50f;
    const float chroma_t_hi = (wb_rgb != nullptr) ? 0.45f : 0.70f;
    const float chroma_inv  = 1.f / std::max(1e-6f, chroma_t_hi - chroma_t_lo);

    const float inv_wb_r = (wb_rgb != nullptr) ? 1.f / std::max(1e-6f, wb_rgb[0]) : 1.f;
    const float inv_wb_g = (wb_rgb != nullptr) ? 1.f / std::max(1e-6f, wb_rgb[1]) : 1.f;
    const float inv_wb_b = (wb_rgb != nullptr) ? 1.f / std::max(1e-6f, wb_rgb[2]) : 1.f;

    // ハイライト部のR/G, B/G比を求める
    float grf = 0.f, gbf = 0.f, count = 0.f;
    std::deque<size_t>
        highlight; // ついでにハイライト処理するピクセルインデクスを保持
    for (size_t idx = 0; idx < width * height; ++idx) {
      float *pixel = rgb_buffer.image[idx];

      if (highlight_mask_out != nullptr) {
        // Green-channel smoothstep weight.
        float t = (pixel[1] - mask_t_lo) * mask_inv;
        if (t < 0.f) t = 0.f; else if (t > 1.f) t = 1.f;
        const float w_g = t * t * (3.f - 2.f * t);

        // WB-undone triplet (purely a chroma-measurement transform; the
        // actual pixel values stay untouched and continue downstream).
        // When wb_rgb is null the *_n values equal the input and the
        // looser edges above keep the post-WB chroma behaviour.
        const float r_n = pixel[0] * inv_wb_r;
        const float g_n = pixel[1] * inv_wb_g;
        const float b_n = pixel[2] * inv_wb_b;
        const float mx = std::max(r_n, std::max(g_n, b_n));
        const float mn = std::min(r_n, std::min(g_n, b_n));
        const float chroma = (mx > 1e-6f) ? (mx - mn) / mx : 0.f;
        float ct = (chroma - chroma_t_lo) * chroma_inv;
        if (ct < 0.f) ct = 0.f; else if (ct > 1.f) ct = 1.f;
        const float neutrality = 1.f - ct * ct * (3.f - 2.f * ct);

        highlight_mask_out[idx] = w_g * neutrality;
      }

      if (pixel[0] >= saturation_threshold &&
          pixel[2] >= saturation_threshold) {
        {
          highlight.push_back(idx); // ハイライト
        }
        if (pixel[0] < 0.95f && pixel[1] >= saturation_threshold &&
            pixel[1] < 0.95f && pixel[2] < 0.95f) {
          // ハイライトだが、白飛びしてないピクセルの比率を平均化する
          grf += pixel[0] / pixel[1];
          gbf += pixel[2] / pixel[1];
          count += 1.f;
        }
      }
    }
    if (count > 0.f) {
      grf /= count;
      gbf /= count;
    } else {
      grf = 1.f;
      gbf = 1.f;
    }

    // highlight処理
    std::deque<size_t>
        white; // ついでに完全白飛び処理するピクセルインデクスを保持
    float max_val = 0.f;
    for (std::deque<size_t>::iterator it = highlight.begin();
         it != highlight.end(); ++it) {
      const size_t idx = *it;
      float *pixel = rgb_buffer.image[idx];

      for (size_t i = 0; i < channels; ++i) {
        if (pixel[i] > max_val) {
          max_val = pixel[i];
        }
      }

      //            if (pixel[0] >= 0.95f || pixel[2] >= 0.95f) {
      //                if (pixel[0] > pixel[2]) {
      //                    pixel[1] = pixel[0] / grf;
      //                    pixel[1] = pixel[2] / gbf;
      //                } else {
      //                    pixel[1] = pixel[2] / gbf;
      //                    pixel[1] = pixel[0] / grf;
      //                }
      //                pixel[1] = (pixel[0] + pixel[2]) * 0.5f;
      pixel[1] = (pixel[0] / grf + pixel[2] / gbf) * 0.5f;
      //            }

      float sp = (pixel[0] < pixel[2]) ? pixel[0] : pixel[2];
      float sl =
          (std::min(sp, 1.f) - saturation_threshold) / (saturation_threshold);
      /*
                  pixel[0] = pixel[0] * (1.f-sl) + (pixel[0] *
         saturation_threshold) * sl; pixel[2] = pixel[2] * (1.f-sl) + (pixel[2]
         * saturation_threshold) * sl; pixel[1] = (pixel[0] + pixel[2]) * 0.5f;
      */
      pixel[0] = pixel[0] * (1.f - sl) + (pixel[1] * grf) * sl;
      pixel[2] = pixel[2] * (1.f - sl) + (pixel[1] * gbf) * sl;
      pixel[1] = (pixel[0] / grf + pixel[2] / gbf) * 0.5f;

      if (pixel[1] >= 0.95f) {
        white.push_back(idx); // 白飛び
      }
    }
    std::cout << "　 Before max value: " << max_val << std::endl;

    // 白飛び部分のピクセルを馴染ませる
    // NOTE: 隣接ピクセルの平均値を事前に全て計算してから書き込む。
    //       in-place で処理すると書き込み順序によって結果が変わり非決定的になるため、
    //       必ず「元の状態」を参照した値を使うこと。
    float(*image)[3] = rgb_buffer.image;

    // Step 1: 全白飛びピクセルの平均値を計算（元の画像を読み取り専用で参照）
    struct PixelCorrection {
      size_t idx;
      float avg[3];
    };
    std::vector<PixelCorrection> corrections;
    corrections.reserve(white.size());

    for (std::deque<size_t>::iterator it = white.begin(); it != white.end();
         ++it) {
      const size_t idx = *it;
      const size_t y = idx / width;
      const size_t x = idx % width;
      if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        continue;
      }

      PixelCorrection pc;
      pc.idx = idx;
      for (uint32_t c = 0; c < channels; ++c) {
        pc.avg[c] = (image[idx - width - 1][c] + image[idx - width + 0][c] +
                     image[idx - width + 1][c] + image[idx - 1][c] +
                     image[idx + 1][c] + image[idx + width - 1][c] +
                     image[idx + width + 0][c] + image[idx + width + 1][c]) *
                    (1.f / 8.f);
      }
      corrections.push_back(pc);
    }

    // Step 2: 計算した平均値を一括書き込み（処理順序に依存しない）
    for (const auto &pc : corrections) {
      for (uint32_t c = 0; c < channels; ++c) {
        image[pc.idx][c] = pc.avg[c];
      }
    }

    // LoCA-specific green compensation on highlight boundaries:
    // if R/B remain dominant after highlight recovery, lift G locally and
    // slightly trim R/B to avoid purple halos while preserving detail.
    size_t loca_compensated = 0;
    for (std::deque<size_t>::iterator it = highlight.begin();
         it != highlight.end(); ++it) {
      const size_t idx = *it;
      const size_t y = idx / width;
      const size_t x = idx % width;
      if (x <= 1 || x >= width - 2 || y <= 1 || y >= height - 2) {
        continue;
      }

      float *p = image[idx];
      const float l = std::max(p[0], std::max(p[1], p[2]));
      if (l < saturation_threshold * 0.90f) {
        continue;
      }

      float local_max = 0.0f;
      float local_min = std::numeric_limits<float>::max();
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const size_t nidx = static_cast<size_t>(static_cast<int>(y) + dy) * width +
                              static_cast<size_t>(static_cast<int>(x) + dx);
          const float *np = image[nidx];
          const float nl = std::max(np[0], std::max(np[1], np[2]));
          local_max = std::max(local_max, nl);
          local_min = std::min(local_min, nl);
        }
      }
      if ((local_max - local_min) < 0.015f) {
        continue;
      }

      const float rb = 0.5f * (p[0] + p[2]);
      if (rb <= p[1] * 1.04f) {
        continue;
      }

      // Estimate local neutral/valid G/RB ratio from nearby highlight region.
      double ratio_sum = 0.0;
      double ratio_w = 0.0;
      for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
          const size_t nidx = static_cast<size_t>(static_cast<int>(y) + dy) * width +
                              static_cast<size_t>(static_cast<int>(x) + dx);
          const float *np = image[nidx];
          const float nrb = 0.5f * (np[0] + np[2]);
          const float nl = std::max(np[0], std::max(np[1], np[2]));
          if (nl < saturation_threshold * 0.75f || nrb <= 1e-6f) {
            continue;
          }
          const float ratio = np[1] / nrb;
          if (ratio < 0.70f || ratio > 1.35f) {
            continue;
          }
          const double w = 1.0 / static_cast<double>(1 + std::abs(dx) + std::abs(dy));
          ratio_sum += ratio * w;
          ratio_w += w;
        }
      }

      const float local_ratio = static_cast<float>(ratio_w > 0.0 ? ratio_sum / ratio_w : 1.0);
      const float desired_g = rb * std::clamp(local_ratio, 0.95f, 1.08f);
      const float g_missing = desired_g - p[1];
      if (g_missing <= 0.0f) {
        continue;
      }

      const float sat_w = std::clamp((l - saturation_threshold) /
                                         std::max(1e-6f, std::max(1.0f, max_val) - saturation_threshold),
                                     0.0f, 1.0f);
      const float edge_w =
          std::clamp((local_max - local_min) / 0.08f, 0.0f, 1.0f);
      const float alpha = std::clamp(0.30f + 0.55f * sat_w * edge_w, 0.0f, 0.90f);

      const float g_boost = alpha * std::min(g_missing, (rb - p[1]) * 0.90f);
      p[1] += g_boost;

      // Mild RB trim to prevent over-magenta ring after G lift.
      const float rb_residual = std::max(0.0f, 0.5f * (p[0] + p[2]) - p[1]);
      if (rb_residual > 0.0f) {
        const float trim = std::min(rb_residual * alpha * 0.45f,
                                    0.5f * (p[0] + p[2]) * 0.12f);
        p[0] = std::max(0.0f, p[0] - trim);
        p[2] = std::max(0.0f, p[2] - trim);
      }
      p[1] = std::max(0.0f, p[1]);
      loca_compensated++;
    }

    std::cout << "✅ Highlight recovery completed. Highlight: "
              << highlight.size() << " pixels.  White: " << white.size()
              << " pixels.  LoCA fixed: " << loca_compensated
              << " pixels." << std::endl;
    return max_val;
  }

  // box_filter_normalized is defined as a free function before the class
  // (see below) to allow #pragma float_control(precise) — that pragma cannot
  // appear inside a class declaration.

  float aces_tone_map_scalar(float x) const {
    static constexpr float a = 2.51f;
    static constexpr float b = 0.03f;
    static constexpr float c = 2.43f;
    static constexpr float d = 0.59f;
    static constexpr float e = 0.14f;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
  }

  void apply_detail_preserving_tonemap(ImageBufferFloat &rgb_buffer) {
    if (!rgb_buffer.is_valid() || rgb_buffer.channels != 3) {
      return;
    }

    const size_t width = rgb_buffer.width;
    const size_t height = rgb_buffer.height;
    const size_t size = width * height;
    if (size == 0) {
      return;
    }

    constexpr int radius = 12;
    constexpr float eps = 0.0025f;
    constexpr float edge0 = 0.55f;
    constexpr float edge1 = 0.90f;

    std::vector<float> guide(size);
    std::vector<float> mean(size);
    std::vector<float> work(size);
    std::vector<float> b(size);
    std::vector<float> tmp(size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      const float *p = rgb_buffer.image[i];
      guide[i] = std::max(0.0f, std::max(p[0], std::max(p[1], p[2])));
      work[i] = guide[i] * guide[i];
    }

    box_filter_normalized(guide, mean, tmp, width, height, radius);
    box_filter_normalized(work, work, tmp, width, height, radius);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      const float var = std::max(0.0f, work[i] - mean[i] * mean[i]);
      const float a = var / (var + eps);
      work[i] = a;
      b[i] = mean[i] * (1.0f - a);
    }

    box_filter_normalized(work, work, tmp, width, height, radius);
    box_filter_normalized(b, b, tmp, width, height, radius);

    long long mapped_pixels = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : mapped_pixels)
#endif
    for (size_t i = 0; i < size; ++i) {
      const float base = std::max(1e-6f, work[i] * guide[i] + b[i]);
      const float mapped_base = aces_tone_map_scalar(base);
      float gain = std::clamp(mapped_base / base, 0.0f, 1.0f);

      float w = (base - edge0) / (edge1 - edge0);
      w = std::clamp(w, 0.0f, 1.0f);
      w = w * w * (3.0f - 2.0f * w);
      if (w <= 0.0f) {
        continue;
      }

      gain = 1.0f + (gain - 1.0f) * w;
      rgb_buffer.image[i][0] *= gain;
      rgb_buffer.image[i][1] *= gain;
      rgb_buffer.image[i][2] *= gain;
      mapped_pixels++;
    }

    std::cout << "✅ Detail-preserving tone map completed. Mapped: "
              << mapped_pixels << " pixels." << std::endl;
  }

  //===============================================================
  // Main RAW to RGB processing pipeline
  //===============================================================

  bool process_raw_to_rgb(ImageBuffer &raw_buffer, ImageBufferFloat &rgb_buffer,
                          const ProcessingParams &params) {
    std::cout << "🎯 Starting unified RAW→RGB processing pipeline" << std::endl;
    std::cout << "📋 Parameters: demosaic=" << params.demosaic_algorithm
              << std::endl;

    auto &imgdata = processor.imgdata;

    // Initialize LibRaw and check for raw data
    if (!accelerator) {
      std::cerr << "❌ Accelerator not initialized" << std::endl;
      return false;
    }

    // Set GPU acceleration flag from processing parameters
    accelerator->set_use_gpu_acceleration(params.use_gpu_acceleration);

    std::cout << "　 data_maximum: " << imgdata.color.data_maximum
              << ", maximum: " << imgdata.color.maximum << std::endl;

    if (imgdata.rawdata.ioparams.zero_is_bad) {
      remove_zeroes();
    }

    bool is_bayer = (imgdata.idata.filters || imgdata.idata.colors == 1);
    int subtract_inline = !imgdata.params.bad_pixels &&
                          !imgdata.params.dark_frame && is_bayer &&
                          !imgdata.rawdata.ioparams.zero_is_bad;

    /*
            if (subtract_inline) {
                adjust_bl();

                imgdata.color.data_maximum = 0;
                imgdata.color.maximum -= imgdata.color.black;
                imgdata.color.cblack[0] = imgdata.color.cblack[1] =
       imgdata.color.cblack[2] = imgdata.color.cblack[3] = 0;
                imgdata.color.black = 0;
            }
    */
    // Apply LibRaw-compatible black level subtraction
    if (!subtract_inline || !imgdata.color.data_maximum) {
      adjust_bl();
      apply_black_level_correction(raw_buffer);
    }

    // Apply adjust_maximum for dynamic maximum value adjustment (must be after
    // black level correction)
    // adjust_maximum0(raw_buffer, params.adjust_maximum_thr);

    // set filters and xtrans
    const uint32_t filters = imgdata.idata.filters;
    const char(&xtrans)[6][6] = imgdata.idata.xtrans;
    std::cout << "🔍 Filters value: 0x" << std::hex << filters
              << " (FILTERS_XTRANS=" << FILTERS_XTRANS << ")" << std::dec << std::endl;

    // Apply green matching for Bayer sensors (after black level, before
    // demosaic)
    apply_green_matching(raw_buffer, filters);

    // rgb_buffer2 is temporary buffer
    AlignedFloatVec rgb_buffer2_data(rgb_buffer.width * rgb_buffer.height *
                                    rgb_buffer.channels);
    ImageBufferFloat rgb_buffer2 = {
        reinterpret_cast<float(*)[3]>(rgb_buffer2_data.data()),
        rgb_buffer.width, rgb_buffer.height, rgb_buffer.channels};

    // Calculate white balance multipliers (same logic as original)
    float effective_wb[4];
    if (!imgdata.params.no_auto_scale) {
      scale_colors(raw_buffer, effective_wb);
    } else {
      effective_wb[0] = 1.0;
      effective_wb[1] = 1.0;
      effective_wb[2] = 1.0;
      effective_wb[3] = 1.0;
    }
    std::cout << "📷 WB: " << effective_wb[0] << ", " << effective_wb[1] << ", "
              << effective_wb[2] << ", " << effective_wb[3] << std::endl;

    // ===========================================================
    // Per-stage timing instrumentation.  Prints "[T] <stage> N ms"
    // for every major step so we can pinpoint where wall-clock
    // time is going (e.g. CA on/off comparisons).
    // ===========================================================
    using _clk = std::chrono::steady_clock;
    auto _t_total0 = _clk::now();
    auto _t_stage = _t_total0;
    auto _stage = [&](const char* name) {
        auto now = _clk::now();
        double ms = std::chrono::duration<double, std::milli>(now - _t_stage).count();
        _t_stage = now;
        std::cout << "[T] " << name << " " << std::fixed
                  << std::setprecision(1) << ms << " ms" << std::endl;
    };

    // Determine CFA type and apply appropriate WB processing
    if (!accelerator->apply_white_balance(raw_buffer, rgb_buffer2, effective_wb,
                                          filters, xtrans)) {
      return false;
    }
    _stage("apply_white_balance");

    libraw_decoder_info_t di;
    processor.get_decoder_info(&di);

    // Apply adjust_maximum for dynamic maximum value adjustment (must be after
    // black level correction)
    MaximumResult maximum_result;
    if (!(di.decoder_flags & LIBRAW_DECODER_FIXEDMAXC)) {
      maximum_result = adjust_maximum(rgb_buffer2, params.adjust_maximum_thr);
    } else {
      maximum_result.data_maximum = imgdata.color.data_maximum;
      maximum_result.maximum = imgdata.color.maximum;
    }
    if (imgdata.params.user_sat > 0) {
      maximum_result.maximum = imgdata.params.user_sat;
    }

    _stage("adjust_maximum");

    // Apply pre-interpolation processing
    if (!accelerator->pre_interpolate(rgb_buffer2, filters, xtrans,
                                      params.half_size)) {
      return false;
    }
    _stage("pre_interpolate");

    // Get camera-specific color transformation matrix
    ColorTransformMatrix camera_matrix = compute_camera_transform(
        imgdata.idata.make, imgdata.idata.model, ColorSpace::XYZ);
    if (!camera_matrix.valid) {
      std::cout << "⚠️ Camera not in database, using fallback matrix"
                << std::endl;
      // Use fallback identity-like matrix for unknown cameras
      camera_matrix.set_default();
    }

    // Store the camera matrix so it can be retrieved via Python API
    std::memcpy(last_color_matrix.data(), camera_matrix.transform,
                12 * sizeof(float));

    if (params.preprocess) {
      std::cout << "📋 Preprocess flag is true. Skipping demosaic and "
                   "returning raw data."
                << std::endl;

      // Before returning, we must normalize the data to 0.0-1.0 range,
      // since the Python bindings expect that and will clamp to 0.0-1.0.
      // 14-bit data -> 16383, 12-bit data -> 4095
      float max_val = 16383.0f; // Default fallback
      if (imgdata.color.raw_bps > 0) {
        max_val = (float)((1 << imgdata.color.raw_bps) - 1);
      } else if (imgdata.color.maximum > 0) {
        max_val = (float)imgdata.color.maximum; // Theoretical max
      }

      size_t num_pixels = rgb_buffer.width * rgb_buffer.height;

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < num_pixels; ++i) {
        rgb_buffer.image[i][0] = rgb_buffer2.image[i][0] / max_val;
        rgb_buffer.image[i][1] = rgb_buffer2.image[i][1] / max_val;
        rgb_buffer.image[i][2] = rgb_buffer2.image[i][2] / max_val;
      }
      return true;
    }

    // Demosaic processing (unified CPU/GPU selection via accelerator)
    // Pass LibRaw cam_mul for dynamic initialGain calculation and maximum_value
    // for precise normalization
    bool demosaic_success = accelerator->demosaic_compute(
        rgb_buffer2, rgb_buffer, params.demosaic_algorithm, filters, xtrans,
        camera_matrix.transform, imgdata.color.cam_mul, maximum_result.maximum);
    if (!demosaic_success) {
      std::cerr << "❌ Demosaic processing failed" << std::endl;
      return false;
    }
    _stage("demosaic");

    float threshold = maximum_result.maximum / maximum_result.data_maximum;

    // Soft highlight mask shared between recover_highlights and
    // enhance_micro_contrast.  Only allocated when both stages can run,
    // i.e. highlight_mode > 3 (the enhance stage gate).  The mask is built
    // from the *pre-recovery* pixel values so it captures the original
    // highlight extent regardless of any later tone-mapping compression;
    // since it is indexed by pixel position the mapping stays valid even
    // when tone_mapping is applied between the two stages.
    AlignedFloatVec highlight_mask_data;
    float *highlight_mask = nullptr;
    if (params.highlight_mode > 3) {
      highlight_mask_data.assign(rgb_buffer.width * rgb_buffer.height, 0.f);
      highlight_mask = highlight_mask_data.data();
    }

    // Highlight recovery — pass the effective WB so the highlight-mask
    // chroma calculation can virtually undo the WB skew and discriminate
    // true neutrals from slightly-tinted highlights in raw space.  The
    // pixel data itself is unaffected; only the mask judgement uses this.
    const float wb_rgb[3] = { effective_wb[0], effective_wb[1], effective_wb[2] };
    if (params.highlight_mode > 2) {
      recover_highlights(rgb_buffer, threshold, highlight_mask, wb_rgb);
      _stage("highlight_recovery");
    }

    // Tone mapping / highlight detail recovery
    float target_contrast = 0.04f;
    if (params.highlight_mode > 5) {
      accelerator->tone_mapping(rgb_buffer, rgb_buffer, 1.f);
      _stage("tone_mapping");
    } else if (params.highlight_mode > 4) {
      if (!accelerator->apply_detail_preserving_tonemap_gpu(rgb_buffer,
                                                            rgb_buffer)) {
        apply_detail_preserving_tonemap(rgb_buffer);
      }
      _stage("detail_preserving_tonemap");
    }

    if (params.highlight_mode > 3) {
      // `threshold` arg is unused when a mask is supplied; pass the old
      // value for safety in case the call ever falls back.
      accelerator->enhance_micro_contrast(rgb_buffer, rgb_buffer,
                                          threshold + 0.3f, 8.f,
                                          target_contrast, highlight_mask);
      _stage("enhance_micro_contrast");
    }

    // Fuji SuperCCD honeycomb sensors require an additional geometric
    // rotation/stretch step after demosaic to match the final orientation.
    if (imgdata.rawdata.ioparams.fuji_width > 0 && imgdata.params.use_fuji_rotate) {
      const int shrink = imgdata.rawdata.ioparams.shrink;
      int fuji_width = (imgdata.rawdata.ioparams.fuji_width - 1 + shrink) >> shrink;
      if (fuji_width > 0) {
        const double step = std::sqrt(0.5);
        const size_t src_w = rgb_buffer.width;
        const size_t src_h = rgb_buffer.height;
        const int wide_i = static_cast<int>(fuji_width / step);
        const int high_i = static_cast<int>((static_cast<int>(src_h) - fuji_width) / step);

        if (wide_i > 1 && high_i > 1) {
          const size_t wide = static_cast<size_t>(wide_i);
          const size_t high = static_cast<size_t>(high_i);
          AlignedFloatVec rotated_data(wide * high * 3, 0.0f);
          float (*rot)[3] = reinterpret_cast<float(*)[3]>(rotated_data.data());

          for (size_t row = 0; row < high; ++row) {
            for (size_t col = 0; col < wide; ++col) {
              const float r = static_cast<float>(
                  fuji_width + (static_cast<double>(row) - static_cast<double>(col)) * step);
              const float c = static_cast<float>(
                  (static_cast<double>(row) + static_cast<double>(col)) * step);
              const size_t ur = static_cast<size_t>(r);
              const size_t uc = static_cast<size_t>(c);
              if (ur >= src_h - 1 || uc >= src_w - 1) {
                continue;
              }

              const float fr = r - static_cast<float>(ur);
              const float fc = c - static_cast<float>(uc);
              const float *p00 = rgb_buffer.image[ur * src_w + uc];
              const float *p01 = rgb_buffer.image[ur * src_w + (uc + 1)];
              const float *p10 = rgb_buffer.image[(ur + 1) * src_w + uc];
              const float *p11 = rgb_buffer.image[(ur + 1) * src_w + (uc + 1)];
              float *dst = rot[row * wide + col];

              for (int ch = 0; ch < 3; ++ch) {
                const float top = p00[ch] * (1.0f - fc) + p01[ch] * fc;
                const float bottom = p10[ch] * (1.0f - fc) + p11[ch] * fc;
                dst[ch] = top * (1.0f - fr) + bottom * fr;
              }
            }
          }

          rgb_buffer_image.swap(rotated_data);
          rgb_buffer.width = wide;
          rgb_buffer.height = high;
          rgb_buffer.image =
              reinterpret_cast<float(*)[3]>(rgb_buffer_image.data());
          std::cout << "✅ Applied Fuji rotate: " << src_w << "x" << src_h
                    << " -> " << rgb_buffer.width << "x" << rgb_buffer.height
                    << std::endl;
        }
      }
    }

    // Lateral chromatic aberration registration in post-demosaic dense RGB.
    // Runs BEFORE defringe so the chroma cleanup step sees properly aligned
    // R/G/B channels.  When this step does its job, the downstream defringe
    // has much less residual lateral fringing to mop up.
    if (params.lateral_ca_correction) {
      std::cout << "🔧 Applying lateral CA registration (cell="
                << params.lateral_ca_cell_size
                << " iter=" << params.lateral_ca_max_iterations
                << " max_shift=" << params.lateral_ca_max_shift
                << " pyramid=" << params.lateral_ca_pyramid_levels << ")"
                << std::endl;
      accelerator->ca_register_lateral(rgb_buffer, rgb_buffer,
                                       params.lateral_ca_cell_size,
                                       params.lateral_ca_max_iterations,
                                       params.lateral_ca_max_shift,
                                       params.lateral_ca_min_confidence,
                                       params.lateral_ca_pyramid_levels);
      _stage("ca_register_lateral");
    }

    // Axial CA cleanup (guided filter).  Runs AFTER lateral registration so
    // R/G/B are aligned, and BEFORE defringe so chroma suppression sees
    // halo-tightened channels.
    if (params.axial_ca_correction) {
      std::cout << "🔧 Applying axial CA cleanup (radius="
                << params.axial_ca_radius
                << " eps=" << params.axial_ca_epsilon
                << " strength=" << params.axial_ca_strength << ")"
                << std::endl;
      accelerator->ca_axial_cleanup(rgb_buffer, rgb_buffer,
                                    params.axial_ca_radius,
                                    params.axial_ca_epsilon,
                                    params.axial_ca_strength);
      _stage("ca_axial_cleanup");
    }

    // Defringe in linear camera RGB before output color-space conversion and
    // gamma. This keeps detection stable for gamma=(1,1) and for different
    // output color spaces.
    if (params.defringe) {
      std::cout << "🔧 Applying linear defringe (radius=" << params.defringe_radius
                << " strength=" << params.defringe_strength
                << " green=" << (params.defringe_green ? "on" : "off")
                << " green_strength=" << params.defringe_green_strength
                << ")" << std::endl;
      accelerator->defringe(rgb_buffer, rgb_buffer,
                            params.defringe_radius,
                            params.defringe_edge_threshold,
                            params.defringe_chroma_threshold,
                            params.defringe_strength,
                            params.defringe_green,
                            params.defringe_green_strength);
      _stage("defringe");
    }

    // Get camera-specific color transformation matrix
    camera_matrix = compute_camera_transform(
        imgdata.idata.make, imgdata.idata.model, params.output_color_space);
    if (!camera_matrix.valid) {
      std::cout << "⚠️ Camera not in database, using fallback matrix"
                << std::endl;
      camera_matrix.set_default();
    }

    // Convert Color space
    if (!accelerator->convert_color_space(rgb_buffer, rgb_buffer,
                                          camera_matrix.transform)) {
      return false;
    }
    _stage("convert_color_space");

    // Gamma correction with color space awareness
    if (!accelerator->gamma_correct(rgb_buffer, rgb_buffer, params.gamma_power,
                                    params.gamma_slope,
                                    params.output_color_space)) {
      return false;
    }
    _stage("gamma_correct");

    {
      double total_ms = std::chrono::duration<double, std::milli>(
          _clk::now() - _t_total0).count();
      std::cout << "[T] === PIPELINE TOTAL " << std::fixed
                << std::setprecision(1) << total_ms << " ms ===" << std::endl;
    }
    std::cout << "✅ Unified RAW→RGB processing pipeline completed successfully"
              << std::endl;
    return true;
  }

  //===============================================================
  // LibRaw raw2image_ex equivalent implementation (excluding subtract_black)
  //===============================================================
  int convert_raw_to_image() {
    std::cout << "🔧 Converting raw data to image..." << std::endl;

    auto &imgdata = processor.imgdata;

    // Step 1: raw2image_start equivalent - initialization
    raw2image_start();

    // Step 2: Handle existing processed image
    if (imgdata.image) {
      std::cout << "ℹ️ Image data already exists, skipping conversion"
                << std::endl;
      return 0;
    }

    // Step 3: Check for raw data availability
    if (!imgdata.rawdata.raw_image && !imgdata.rawdata.color4_image &&
        !imgdata.rawdata.color3_image) {
      std::cerr << "❌ No raw data available for conversion" << std::endl;
      return LIBRAW_REQUEST_FOR_NONEXISTENT_IMAGE;
    }

    // Step 4: Calculate allocation dimensions
    int do_crop = 0;

    // === ステップ1: クロップ処理が必要かチェック ===
    // cropbox[2]とcropbox[3]が設定されている場合（~演算子でビット反転チェック）
    if (~imgdata.params.cropbox[2] && ~imgdata.params.cropbox[3]) {

      // --- クロップ座標の初期化と検証 ---
      int crop[4]; // [left, top, width, height]
      for (int q = 0; q < 4; q++) {
        crop[q] = imgdata.params.cropbox[q];
        if (crop[q] < 0) {
          crop[q] = 0; // 負の値は0にクランプ
        }
      }

      // --- センサータイプ別のアライメント調整 ---

      if (imgdata.rawdata.ioparams.fuji_width &&
          imgdata.idata.filters >= 1000) {
        // === Fujiセンサー（X-Trans以外のBayer）の処理 ===

        // 開始位置を4ピクセル境界に合わせる
        crop[0] = (crop[0] / 4) * 4; // left
        crop[1] = (crop[1] / 4) * 4; // top

        // Fujiの特殊レイアウト処理
        if (!processor.get_internal_data_pointer()->unpacker_data.fuji_layout) {
          // 45度回転レイアウトの場合の幅・高さ補正
          crop[2] *= sqrt(2.0); // width を√2倍
          crop[3] /= sqrt(2.0); // height を√2で割る
        }

        // サイズを4ピクセル境界に合わせる（切り上げ）
        crop[2] = (crop[2] / 4 + 1) * 4; // width
        crop[3] = (crop[3] / 4 + 1) * 4; // height
      } else if (imgdata.idata.filters == 1) {
        // === モノクローム/特殊センサーの処理 ===
        // 16ピクセル境界にアライメント
        crop[0] = (crop[0] / 16) * 16; // left
        crop[1] = (crop[1] / 16) * 16; // top
      } else if (imgdata.idata.filters == FILTERS_XTRANS) {
        // === Fuji X-Transセンサーの処理 ===
        // 6x6パターンの境界に合わせる
        crop[0] = (crop[0] / 6) * 6; // left
        crop[1] = (crop[1] / 6) * 6; // top
      }
      // 通常のBayerセンサー（filters >= 1000）の場合は特別な調整なし

      do_crop = 1; // クロップ実行フラグをセット

      // --- クロップサイズの最終検証と調整 ---

      // 画像境界内に収める
      crop[2] =
          std::min(crop[2], (signed)imgdata.sizes.width - crop[0]); // width制限
      crop[3] = std::min(crop[3],
                         (signed)imgdata.sizes.height - crop[1]); // height制限

      // 無効なクロップサイズの検出
      if (crop[2] <= 0 || crop[3] <= 0) {
        throw LIBRAW_EXCEPTION_BAD_CROP;
      }

      // --- 画像サイズ情報の更新 ---

      // マージン調整（クロップ開始位置分だけマージンを増加）
      imgdata.sizes.left_margin += crop[0];
      imgdata.sizes.top_margin += crop[1];

      // 新しい画像サイズを設定
      imgdata.sizes.width = crop[2];
      imgdata.sizes.height = crop[3];

      // 縮小処理を考慮した最終画像サイズ
      imgdata.sizes.iheight =
          (imgdata.sizes.height + imgdata.rawdata.ioparams.shrink) >>
          imgdata.rawdata.ioparams.shrink; // >> IO.shrink は /2^shrink と同じ
      imgdata.sizes.iwidth =
          (imgdata.sizes.width + imgdata.rawdata.ioparams.shrink) >>
          imgdata.rawdata.ioparams.shrink;

      // --- Bayerフィルターパターンの再計算 ---
      // 通常のBayerセンサー（Fuji以外）でクロップした場合
      if (!imgdata.rawdata.ioparams.fuji_width && imgdata.idata.filters &&
          imgdata.idata.filters >= 1000) {

        int filt, c;

        // 新しいクロップ位置での4x4 Bayerパターンを再計算
        for (filt = c = 0; c < 16; c++) {
          // 4x4グリッドの各位置での色を計算
          int row = (c >> 1) + crop[1]; // 行位置 = (c/2) + top_offset
          int col = (c & 1) + crop[0];  // 列位置 = (c%2) + left_offset

          // FC関数で該当位置の色を取得し、2ビットずつ格納
          filt |= fcol_bayer_native(row, col, imgdata.idata.filters) << (c * 2);
        }

        // 新しいフィルターパターンを設定
        imgdata.idata.filters = filt;
      }
    }

    // === ステップ2: メモリ割り当てサイズの計算 ===

    // デフォルトの割り当てサイズ
    int alloc_width = imgdata.sizes.iwidth;
    int alloc_height = imgdata.sizes.iheight;

    // Fujiセンサーでクロップが実行された場合の特殊計算
    if (imgdata.rawdata.ioparams.fuji_width && do_crop) {

      // --- Fuji特殊レイアウト用のメモリサイズ計算 ---

      // レイアウトタイプに基づく幅の調整
      int IO_fw =
          imgdata.sizes.width >> int(!processor.get_internal_data_pointer()
                                          ->unpacker_data.fuji_layout);
      // fuji_layout == 1 の場合: imgdata.sizes.width >> 0 = imgdata.sizes.width
      // (シフトなし) fuji_layout == 0 の場合: imgdata.sizes.width >> 1 =
      // imgdata.sizes.width / 2

      // Fuji特殊フォーマットでの必要メモリサイズ計算
      int t_alloc_width =
          (imgdata.sizes.height >>
           processor.get_internal_data_pointer()->unpacker_data.fuji_layout) +
          IO_fw;
      // fuji_layout == 1 の場合: imgdata.sizes.height >> 1 + IO_fw
      // fuji_layout == 0 の場合: imgdata.sizes.height >> 0 + IO_fw =
      // imgdata.sizes.height + IO_fw

      int t_alloc_height = t_alloc_width - 1;

      // 縮小処理を考慮した最終的な割り当てサイズ
      alloc_height = (t_alloc_height + imgdata.rawdata.ioparams.shrink) >>
                     imgdata.rawdata.ioparams.shrink;
      alloc_width = (t_alloc_width + imgdata.rawdata.ioparams.shrink) >>
                    imgdata.rawdata.ioparams.shrink;
    }

    // Step 5: Allocate image buffer
    size_t alloc_sz = alloc_width * alloc_height;
    imgdata.image =
        (unsigned short(*)[4])calloc(alloc_sz, sizeof(*imgdata.image));

    if (!imgdata.image) {
      std::cerr << "❌ Failed to allocate image buffer" << std::endl;
      return LIBRAW_UNSUFFICIENT_MEMORY;
    }

    std::cout << "✅ Allocated image buffer (" << alloc_sz << " pixels)"
              << std::endl;

    // Step 6: Copy data based on source type
    if (imgdata.rawdata.color4_image) {
      std::cout << "🔧 Copying from color4_image..." << std::endl;
      copy_color4_image();
    } else if (imgdata.rawdata.color3_image) {
      std::cout << "🔧 Copying from color3_image..." << std::endl;
      copy_color3_image();
    } else if (imgdata.rawdata.raw_image) {
      std::cout << "🔧 Copying from raw_image (Bayer/X-Trans)..." << std::endl;
      copy_bayer_image();
    } else {
      std::cerr << "❌ Unsupported raw data format" << std::endl;
      return LIBRAW_UNSUPPORTED_THUMBNAIL;
    }

    std::cout << "✅ Raw to image conversion completed successfully"
              << std::endl;
    return 0;
  }

  // raw2image_start equivalent - setup and initialization
  void raw2image_start() {
    std::cout << "🔧 raw2image_start: Initializing conversion parameters..."
              << std::endl;

    auto &imgdata = processor.imgdata;

    // Restore metadata from raw data structures
    if (imgdata.rawdata.color.maximum > 0) {
      memcpy(&imgdata.color, &imgdata.rawdata.color, sizeof(imgdata.color));
    }

    // Calculate image dimensions
    auto &S = imgdata.sizes;
    auto &O = imgdata.params;

    // Handle half-size processing
    bool shrink = !imgdata.rawdata.color4_image &&
                  !imgdata.rawdata.color3_image && imgdata.idata.filters &&
                  O.half_size;

    // Calculate final image dimensions
    if (shrink) {
      S.iheight = (S.height + 1) >> 1;
      S.iwidth = (S.width + 1) >> 1;
      std::cout << "   - Half-size mode: " << S.iwidth << "x" << S.iheight
                << std::endl;
    } else {
      S.iheight = S.height;
      S.iwidth = S.width;
      std::cout << "   - Full-size mode: " << S.iwidth << "x" << S.iheight
                << std::endl;
    }

    std::cout << "✅ raw2image_start completed" << std::endl;
  }

  // Copy from 4-channel processed data
  void copy_color4_image() {
    auto &sizes = processor.imgdata.sizes;
    size_t total_pixels = sizes.iwidth * sizes.iheight;

    for (size_t i = 0; i < total_pixels; i++) {
      for (int c = 0; c < 4; c++) {
        processor.imgdata.image[i][c] =
            processor.imgdata.rawdata.color4_image[i][c];
      }
    }

    std::cout << "✅ Copied " << total_pixels << " pixels from color4_image"
              << std::endl;
  }

  // Copy from 3-channel processed data
  void copy_color3_image() {
    auto &sizes = processor.imgdata.sizes;
    size_t total_pixels = sizes.iwidth * sizes.iheight;

    for (size_t i = 0; i < total_pixels; i++) {
      for (int c = 0; c < 3; c++) {
        processor.imgdata.image[i][c] =
            processor.imgdata.rawdata.color3_image[i][c];
      }
      processor.imgdata.image[i][3] = 0; // Alpha channel
    }

    std::cout << "✅ Copied " << total_pixels << " pixels from color3_image"
              << std::endl;
  }

  // Copy from raw Bayer/X-Trans data
  void copy_bayer_image() {
    auto &sizes = processor.imgdata.sizes;
    auto &params = processor.imgdata.params;
    auto &idata = processor.imgdata.idata;

    std::cout << "🔧 Processing Bayer/X-Trans pattern..." << std::endl;
    std::cout << "   - Raw size: " << sizes.raw_width << "x" << sizes.raw_height
              << std::endl;
    std::cout << "   - Output size: " << sizes.iwidth << "x" << sizes.iheight
              << std::endl;
    std::cout << "   - Filters: 0x" << std::hex << idata.filters << std::dec
              << std::endl;

    bool shrink = params.half_size && idata.filters;
    int shrink_factor = shrink ? 1 : 0;

    // Initialize all image data to zero
    size_t total_pixels = sizes.iwidth * sizes.iheight;
    memset(processor.imgdata.image, 0,
           total_pixels * sizeof(*processor.imgdata.image));

    // Handle Foveon X3 sensors (special case)
    if (idata.is_foveon) {
      copy_foveon_image();
      return;
    }

    // Handle special formats (Phase One, Leaf, etc.)
    if (copy_special_formats()) {
      return;
    }

    if (idata.filters == FILTERS_XTRANS) {
      copy_xtrans_image(shrink_factor);
      return;
    }

    copy_bayer_image(shrink_factor);

    std::cout << "✅ Copied Bayer data to image buffer" << std::endl;
  }

  // Handle Foveon X3 sensors
  void copy_foveon_image() {
    auto &sizes = processor.imgdata.sizes;
    size_t total_pixels = sizes.iwidth * sizes.iheight;

    std::cout << "🔧 Processing Foveon X3 sensor..." << std::endl;

    // Foveon has 3 color values per pixel position
    for (size_t i = 0; i < total_pixels; i++) {
      // In Foveon, each layer corresponds to a color
      processor.imgdata.image[i][0] =
          processor.imgdata.rawdata.raw_image[i * 3 + 0]; // Red
      processor.imgdata.image[i][1] =
          processor.imgdata.rawdata.raw_image[i * 3 + 1]; // Green
      processor.imgdata.image[i][2] =
          processor.imgdata.rawdata.raw_image[i * 3 + 2]; // Blue
      processor.imgdata.image[i][3] = 0;                  // Alpha
    }

    std::cout << "✅ Copied Foveon X3 data" << std::endl;
  }

  // Handle special camera formats (Phase One, Leaf, Kodak, etc.)
  bool copy_special_formats() {
    auto &idata = processor.imgdata.idata;

    // Phase One cameras
    if (strstr(idata.make, "Phase One") || strstr(idata.model, "Phase One")) {
      std::cout << "🔧 Processing Phase One format..." << std::endl;
      return copy_phase_one_image();
    }

    // Leaf cameras
    if (strstr(idata.make, "Leaf") || strstr(idata.model, "Leaf")) {
      std::cout << "🔧 Processing Leaf format..." << std::endl;
      return copy_leaf_image();
    }

    // Kodak cameras
    if (strstr(idata.make, "KODAK") || strstr(idata.model, "KODAK")) {
      std::cout << "🔧 Processing Kodak format..." << std::endl;
      return copy_kodak_image();
    }

    // Hasselblad cameras
    /*
            if (strstr(idata.make, "Hasselblad") || strstr(idata.model,
       "Hasselblad")) { std::cout << "🔧 Processing Hasselblad format..." <<
       std::endl; return copy_hasselblad_image();
            }
    */
    return false; // No special format detected
  }

  // Phase One specific processing
  bool copy_phase_one_image() {
    auto &sizes = processor.imgdata.sizes;
    size_t total_pixels = sizes.iwidth * sizes.iheight;

    // Phase One uses specific channel ordering
    for (size_t i = 0; i < total_pixels; i++) {
      int row = i / sizes.iwidth;
      int col = i % sizes.iwidth;

      // Phase One color filter array pattern
      int color_channel = get_phase_one_color(row, col);
      unsigned short val = processor.imgdata.rawdata.raw_image[i];

      processor.imgdata.image[i][color_channel] = val;
    }

    std::cout << "✅ Processed Phase One format" << std::endl;
    return true;
  }

  // Leaf specific processing
  bool copy_leaf_image() {
    auto &sizes = processor.imgdata.sizes;
    size_t total_pixels = sizes.iwidth * sizes.iheight;

    for (size_t i = 0; i < total_pixels; i++) {
      int row = i / sizes.iwidth;
      int col = i % sizes.iwidth;

      int color_channel = get_leaf_color(row, col);
      unsigned short val = processor.imgdata.rawdata.raw_image[i];

      processor.imgdata.image[i][color_channel] = val;
    }

    std::cout << "✅ Processed Leaf format" << std::endl;
    return true;
  }

  // Kodak specific processing
  bool copy_kodak_image() {
    auto &sizes = processor.imgdata.sizes;
    size_t total_pixels = sizes.iwidth * sizes.iheight;

    for (size_t i = 0; i < total_pixels; i++) {
      int row = i / sizes.iwidth;
      int col = i % sizes.iwidth;

      int color_channel = get_kodak_color(row, col);
      unsigned short val = processor.imgdata.rawdata.raw_image[i];

      processor.imgdata.image[i][color_channel] = val;
    }

    std::cout << "✅ Processed Kodak format" << std::endl;
    return true;
  }

  // Hasselblad specific processing
  bool copy_hasselblad_image() {
    auto &sizes = processor.imgdata.sizes;
    size_t total_pixels = sizes.iwidth * sizes.iheight;

    for (size_t i = 0; i < total_pixels; i++) {
      int row = i / sizes.iwidth;
      int col = i % sizes.iwidth;

      int color_channel = get_hasselblad_color(row, col);
      unsigned short val = processor.imgdata.rawdata.raw_image[i];

      processor.imgdata.image[i][color_channel] = val;
    }

    std::cout << "✅ Processed Hasselblad format" << std::endl;
    return true;
  }

  bool copy_xtrans_image(int shrink_factor) {
    auto &sizes = processor.imgdata.sizes;

    // Standard X-Trans processing
    for (int row = 0;
         row < sizes.height && row < sizes.raw_height - sizes.top_margin;
         row++) {
      for (int col = 0;
           col < sizes.width && col < sizes.raw_width - sizes.left_margin;
           col++) {

        // Calculate source pixel position
        int src_row = row + sizes.top_margin;
        int src_col = col + sizes.left_margin;
        int src_idx = src_row * (sizes.raw_width) + src_col;

        // Calculate destination pixel position (with potential shrinking)
        int dst_row = row >> shrink_factor;
        int dst_col = col >> shrink_factor;
        int dst_idx = dst_row * sizes.iwidth + dst_col;

        // Skip if destination is out of bounds
        if (dst_row >= sizes.iheight || dst_col >= sizes.iwidth)
          continue;

        // Get raw pixel value
        unsigned short val = processor.imgdata.rawdata.raw_image[src_idx];

        // Determine color channel using filter pattern
        int color_channel =
            fcol_xtrans(row, col, processor.imgdata.idata.xtrans);

        // Store pixel value in appropriate channel
        processor.imgdata.image[dst_idx][color_channel] = val;
      }
    }
    return true;
  }

  bool copy_bayer_image(int shrink_factor) {
    auto &sizes = processor.imgdata.sizes;
    auto &ioparams = processor.imgdata.rawdata.ioparams;
    const bool is_fuji_honeycomb =
        (ioparams.fuji_width > 0 && processor.imgdata.idata.filters >= 1000);
    const bool fuji_layout =
        processor.get_internal_data_pointer()->unpacker_data.fuji_layout != 0;

    // Fuji CCD honeycomb mapping path (aligned with upstream LibRaw raw2image).
    if (is_fuji_honeycomb) {
      const int copy_rows = std::max(0, sizes.raw_height - sizes.top_margin * 2);
      const int copy_cols =
          ioparams.fuji_width << static_cast<int>(!fuji_layout);

      for (int row = 0; row < copy_rows; row++) {
        for (int col = 0; col < copy_cols; col++) {
          int r, c;
          if (fuji_layout) {
            r = ioparams.fuji_width - 1 - col + (row >> 1);
            c = col + ((row + 1) >> 1);
          } else {
            r = ioparams.fuji_width - 1 + row - (col >> 1);
            c = row + ((col + 1) >> 1);
          }

          const int src_row = row + sizes.top_margin;
          const int src_col = col + sizes.left_margin;
          if (src_row < 0 || src_row >= sizes.raw_height || src_col < 0 ||
              src_col >= sizes.raw_width) {
            continue;
          }
          if (r < 0 || c < 0 || r >= sizes.height || c >= sizes.width) {
            continue;
          }

          const int dst_row = r >> shrink_factor;
          const int dst_col = c >> shrink_factor;
          if (dst_row < 0 || dst_col < 0 || dst_row >= sizes.iheight ||
              dst_col >= sizes.iwidth) {
            continue;
          }

          const int src_idx = src_row * sizes.raw_pitch / 2 + src_col;
          const int dst_idx = dst_row * sizes.iwidth + dst_col;
          const uint16_t val = processor.imgdata.rawdata.raw_image[src_idx];
          const uint32_t color_channel =
              fcol_bayer_native(r, c, processor.imgdata.idata.filters);

          processor.imgdata.image[dst_idx][color_channel] = val;
          if (color_channel == 3) {
            processor.imgdata.image[dst_idx][1] = val;
          }
        }
      }
      return true;
    }

    // Standard Bayer processing.
    for (int row = 0;
         row < sizes.height && row < sizes.raw_height - sizes.top_margin;
         row++) {
      for (int col = 0;
           col < sizes.width && col < sizes.raw_width - sizes.left_margin;
           col++) {

        // Calculate source pixel position
        int src_row = row + sizes.top_margin;
        int src_col = col + sizes.left_margin;
        int src_idx = src_row * sizes.raw_pitch / 2 + src_col;

        // Calculate destination pixel position (with potential shrinking)
        int dst_row = row >> shrink_factor;
        int dst_col = col >> shrink_factor;
        int dst_idx = dst_row * sizes.iwidth + dst_col;

        // Skip if destination is out of bounds
        if (dst_row >= sizes.iheight || dst_col >= sizes.iwidth)
          continue;

        // Get raw pixel value
        uint16_t val = processor.imgdata.rawdata.raw_image[src_idx];

        // Determine color channel using filter pattern
        uint32_t color_channel =
            fcol_bayer_native(row, col, processor.imgdata.idata.filters);
        /*
                        if (val > processor.imgdata.color.cblack[color_channel])
           { val -= processor.imgdata.color.cblack[color_channel]; } else { val
           = 0;
                        }
        */
        // Store pixel value in appropriate channel
        processor.imgdata.image[dst_idx][color_channel] = val;
        if (color_channel == 3) {
          // If G2 channel, copy to G1 for averaging
          processor.imgdata.image[dst_idx][1] = val;
        }
      }
    }
    return true;
  }

  // Phase One color channel determination
  int get_phase_one_color(int row, int col) {
    // Phase One specific Bayer pattern variations
    // Usually RGGB but with specific modifications for Phase One sensors
    return ((row & 1) << 1) | (col & 1); // Simplified for now
  }

  // Leaf color channel determination
  int get_leaf_color(int row, int col) {
    // Leaf cameras typically use standard Bayer patterns
    // RGGB pattern: R=0, G=1, B=2
    if ((row & 1) == 0) {
      return (col & 1) == 0 ? 0 : 1; // R or G
    } else {
      return (col & 1) == 0 ? 1 : 2; // G or B
    }
  }

  // Kodak color channel determination
  int get_kodak_color(int row, int col) {
    // Kodak often uses unique color filter arrays
    // Some use GRBG, others use RGGB variants
    if ((row & 1) == 0) {
      return (col & 1) == 0 ? 1 : 0; // G or R
    } else {
      return (col & 1) == 0 ? 2 : 1; // B or G
    }
  }

  // Hasselblad color channel determination
  int get_hasselblad_color(int row, int col) {
    // Hasselblad typically uses standard Bayer RGGB
    if ((row & 1) == 0) {
      return (col & 1) == 0 ? 0 : 1; // R or G
    } else {
      return (col & 1) == 0 ? 1 : 2; // G or B
    }
  }

  Impl() {
#ifdef __arm64__
    // Metal加速器初期化
    accelerator = std::make_unique<Accelerator>();
    accelerator->initialize();
#else
    std::cout << "Metal acceleration not compiled in" << std::endl;
#endif
  }

  int load_file(const std::string &filename) {
    start_timer();
    int result = processor.open_file(filename.c_str());
    timing_info.file_load_time = get_elapsed_time();

    return result;
  }

  int unpack() {
    start_timer();
    int result = processor.unpack();
    timing_info.unpack_time = get_elapsed_time();

    return result;
  }

  int process() {
    start_timer();

    // CRITICAL: Ensure default parameters are initialized if not already set

    if (current_params.user_wb[0] == 0.0f &&
        current_params.user_wb[1] == 0.0f &&
        current_params.user_wb[2] == 0.0f &&
        current_params.user_wb[3] == 0.0f) {
      std::cout << "⚠️  current_params appears uninitialized, setting defaults"
                << std::endl;
      ProcessingParams defaults;
      set_processing_params(defaults);
    }

    if (accelerator && accelerator->is_available()) {
      std::cout << "🚀 Using unified accelerated pipeline (automatic GPU/CPU "
                   "selection)"
                << std::endl;

      // Step 1: Convert raw sensor data to processed image (LibRaw raw2image_ex
      // equivalent)
      std::cout << "🔧 Converting raw sensor data to processed image format..."
                << std::endl;
      int raw2image_result = convert_raw_to_image();
      if (raw2image_result != 0) {
        return false;
      }

      // Prepare RAW data buffer (now properly initialized)
      ImageBuffer raw_buffer;
      raw_buffer.width =
          processor.imgdata.sizes.iwidth; // Use iwidth (processed width)
      raw_buffer.height =
          processor.imgdata.sizes.iheight; // Use iheight (processed height)
      raw_buffer.channels = 4;
      raw_buffer.image = processor.imgdata.image; // Now guaranteed non-null

      // Prepare output RGB buffer
      rgb_buffer.width =
          processor.imgdata.sizes.iwidth; // Use processed width, not raw width
      rgb_buffer.height = processor.imgdata.sizes
                              .iheight; // Use processed height, not raw height
      rgb_buffer.channels = 3;
      size_t float_elements =
          rgb_buffer.width * rgb_buffer.height * rgb_buffer.channels;
      rgb_buffer_image.resize(
          float_elements); // Resize vector to hold float data
      rgb_buffer.image = reinterpret_cast<float(*)[3]>(rgb_buffer_image.data());

      // Use unified processing pipeline
      if (process_raw_to_rgb(raw_buffer, rgb_buffer, current_params)) {
        timing_info.total_time = get_elapsed_time();
        return LIBRAW_SUCCESS;

      } else {
        std::cout << "❌ Unified pipeline failed, NO FALLBACK (testing mode)"
                  << std::endl;
        return LIBRAW_UNSPECIFIED_ERROR; // フォールバック無効
      }
    }

    // FALLBACK DISABLED FOR TESTING
    std::cout << "❌ LibRaw dcraw_process fallback DISABLED for testing"
              << std::endl;
    timing_info.total_time = get_elapsed_time();
    return LIBRAW_UNSPECIFIED_ERROR;

    // Fall back to standard LibRaw CPU processing (DISABLED)
    // int result = processor.dcraw_process();
    // return result;
  }

#ifdef __arm64__
  // Store processing results
  AlignedFloatVec rgb_buffer_image;
  ImageBufferFloat rgb_buffer;
#endif

  ProcessedImageData get_processed_image() {
    ProcessedImageData result;

#ifdef __arm64__
    // Check if we have Metal-processed data
    if (rgb_buffer.is_valid()) {
      result.width = rgb_buffer.width;
      result.height = rgb_buffer.height;
      result.channels = rgb_buffer.channels;

      switch (current_params.output_bps) {
      case 8:
        result.bits_per_sample = 8;
        break;
      case 16:
        result.bits_per_sample = 16;
        break;
      default:
        result.bits_per_sample = 32;
        break;
      }
      result.data = reinterpret_cast<float *>(rgb_buffer.image);

      result.error_code = LIBRAW_SUCCESS;
      result.timing_info = timing_info; // 計測情報を含める

      // カラーマトリックス情報を設定
      result.color_matrix = last_color_matrix;

      return result;
    }
#endif

    // Standard LibRaw processing
    int error_code;
    libraw_processed_image_t *processed_image =
        processor.dcraw_make_mem_image(&error_code);

    if (!processed_image) {
      result.error_code = error_code;
      return result;
    }

    result.width = processed_image->width;
    result.height = processed_image->height;
    result.channels = processed_image->colors;
    result.bits_per_sample = processed_image->bits;

    // データをコピー
    // size_t data_size = result.width * result.height * result.channels *
    // (result.bits_per_sample / 8); result.data.resize(data_size);
    // memcpy(result.data.data(), processed_image->data, data_size);

    result.error_code = LIBRAW_SUCCESS;
    result.timing_info = timing_info; // 計測情報を含める

    LibRaw::dcraw_clear_mem(processed_image);
    return result;
  }

#ifdef __arm64__
  void set_processing_params(const ProcessingParams &params) {
    current_params = params;

    // Map all parameters to LibRaw

    // Basic processing parameters
    processor.imgdata.params.use_camera_wb = params.use_camera_wb ? 1 : 0;
    processor.imgdata.params.half_size = params.half_size ? 1 : 0;
    processor.imgdata.params.four_color_rgb = params.four_color_rgb ? 1 : 0;
    processor.imgdata.params.output_bps = params.output_bps;
    processor.imgdata.params.user_flip = params.user_flip;

    // Demosaicing parameters
    processor.imgdata.params.user_qual = params.demosaic_algorithm;
    processor.imgdata.params.dcb_iterations = params.dcb_iterations;
    processor.imgdata.params.dcb_enhance_fl = params.dcb_enhance ? 1 : 0;

    // Noise reduction parameters
    processor.imgdata.params.fbdd_noiserd = params.fbdd_noise_reduction;
    processor.imgdata.params.threshold = params.noise_thr;
    processor.imgdata.params.med_passes = params.median_filter_passes;

    // White balance parameters
    processor.imgdata.params.use_auto_wb = params.use_auto_wb ? 1 : 0;
    processor.imgdata.params.user_mul[0] = params.user_wb[0];
    processor.imgdata.params.user_mul[1] = params.user_wb[1];
    processor.imgdata.params.user_mul[2] = params.user_wb[2];
    processor.imgdata.params.user_mul[3] = params.user_wb[3];

    // Color space and output parameters
    processor.imgdata.params.output_color = params.output_color_space;

    // Brightness and exposure parameters
    processor.imgdata.params.bright = params.bright;
    processor.imgdata.params.no_auto_bright = params.no_auto_bright ? 1 : 0;
    processor.imgdata.params.auto_bright_thr = params.auto_bright_thr;
    processor.imgdata.params.adjust_maximum_thr = params.adjust_maximum_thr;

    // Highlight processing
    processor.imgdata.params.highlight = params.highlight_mode;

    // Exposure correction parameters
    processor.imgdata.params.exp_shift = params.exp_shift;
    processor.imgdata.params.exp_preser = params.exp_preserve_highlights;

    // Gamma correction parameters
    processor.imgdata.params.gamm[0] = 1.0 / params.gamma_power;
    processor.imgdata.params.gamm[1] = params.gamma_slope;
    // Set no_auto_scale
    if (params.no_auto_scale) {
      processor.imgdata.params.no_auto_scale = 1;
    }

    // Color correction parameters
    processor.imgdata.params.aber[0] = params.chromatic_aberration_red;
    processor.imgdata.params.aber[1] = 1.0; // Green channel (no correction)
    processor.imgdata.params.aber[2] = params.chromatic_aberration_blue;
    processor.imgdata.params.aber[3] = 1.0; // Green channel (no correction)

    // User adjustments
    if (params.user_black >= 0) {
      processor.imgdata.params.user_black = params.user_black;
    }
    if (params.user_sat >= 0) {
      processor.imgdata.params.user_sat = params.user_sat;
    }

    // File-based corrections
    if (!params.bad_pixels_path.empty()) {
      processor.imgdata.params.bad_pixels =
          const_cast<char *>(params.bad_pixels_path.c_str());
    }
  }

  std::string get_device_info() const {
    if (accelerator) {
      return accelerator->get_device_info();
    }
    return "Metal not available";
  }

  std::vector<uint16_t> get_raw_image() {
    if (!processor.imgdata.rawdata.raw_image) {
      throw std::runtime_error("RAW data not available - call unpack() first");
    }

    size_t raw_width = processor.imgdata.sizes.raw_width;
    size_t raw_height = processor.imgdata.sizes.raw_height;
    size_t pixel_count = raw_width * raw_height;

    std::vector<uint16_t> raw_data(pixel_count);
    std::copy(processor.imgdata.rawdata.raw_image,
              processor.imgdata.rawdata.raw_image + pixel_count,
              raw_data.begin());

    return raw_data;
  }

  py::array_t<uint16_t> get_raw_image_as_numpy() {
    if (!processor.imgdata.rawdata.raw_image) {
      throw std::runtime_error("RAW data not available - call unpack() first");
    }

    size_t raw_width = processor.imgdata.sizes.raw_width;
    size_t raw_height = processor.imgdata.sizes.raw_height;
    size_t total_pixels = raw_width * raw_height;

    // Create numpy array with copied data (safer than direct memory reference)
    auto result = py::array_t<uint16_t>({raw_height, raw_width});
    auto buf = result.request();
    uint16_t *ptr = static_cast<uint16_t *>(buf.ptr);

    // Copy data from LibRaw buffer
    std::copy(processor.imgdata.rawdata.raw_image,
              processor.imgdata.rawdata.raw_image + total_pixels, ptr);

    return result;
  }
#endif
};

// LibRawWrapper実装
LibRawWrapper::LibRawWrapper() : pimpl(std::make_unique<Impl>()) {}

LibRawWrapper::~LibRawWrapper() {}

int LibRawWrapper::load_file(const std::string &filename) {
  return pimpl->load_file(filename);
}

int LibRawWrapper::unpack() { return pimpl->unpack(); }

int LibRawWrapper::process() { return pimpl->process(); }

ProcessedImageData LibRawWrapper::get_processed_image() {
  return pimpl->get_processed_image();
}

void LibRawWrapper::set_processing_params(const ProcessingParams &params) {
  pimpl->set_processing_params(params);
}

void LibRawWrapper::set_gpu_acceleration(bool enable) {
  pimpl->accelerator->set_use_gpu_acceleration(enable);
}

std::string LibRawWrapper::get_device_info() const {
  return pimpl->get_device_info();
}

// New methods for high-level API support
int LibRawWrapper::load_buffer(const std::vector<uint8_t> &buffer) {
  // For prototype: just return success, actual implementation would use LibRaw
  // buffer loading
  return pimpl->load_file(""); // Placeholder
}

std::vector<uint16_t> LibRawWrapper::get_raw_image() {
  return pimpl->get_raw_image();
}

py::array_t<uint16_t> LibRawWrapper::get_raw_image_as_numpy() {
  auto vec = pimpl->get_raw_image();
  return py::array_t<uint16_t>(vec.size(), vec.data());
}

ImageInfo LibRawWrapper::get_image_info() {
  ImageInfo info;
  // Extract info from LibRaw
  libraw_image_sizes_t &sizes = pimpl->processor.imgdata.sizes;
  libraw_iparams_t &iparams = pimpl->processor.imgdata.idata;

  info.width = sizes.width;
  info.height = sizes.height;
  info.raw_width = sizes.raw_width;
  info.raw_height = sizes.raw_height;
  info.colors = iparams.colors;
  info.camera_make = std::string(iparams.make);
  info.camera_model = std::string(iparams.model);
  info.is_xtrans = (iparams.filters == 9);

  if (iparams.cdesc[0] != '\0') {
    info.color_desc = std::string(iparams.cdesc);
  } else {
    // Default fallback if libraw doesn't present it
    info.color_desc = "RGBG";
  }

  // Copy white balance multipliers if available
  if (pimpl->processor.imgdata.color.cam_mul[0] > 0) {
    for (int i = 0; i < 4; i++) {
      info.cam_mul[i] = pimpl->processor.imgdata.color.cam_mul[i];
      info.pre_mul[i] = pimpl->processor.imgdata.color.pre_mul[i];
    }
  }

  return info;
}

std::string LibRawWrapper::get_camera_make() const {
  return std::string(pimpl->processor.imgdata.idata.make);
}

std::string LibRawWrapper::get_camera_model() const {
  return std::string(pimpl->processor.imgdata.idata.model);
}

ProcessedImageData LibRawWrapper::process_with_dict(
    const std::map<std::string, float> &float_params,
    const std::map<std::string, int> &int_params,
    const std::map<std::string, bool> &bool_params,
    const std::map<std::string, std::string> &string_params) {

  // Convert parameters to ProcessingParams
  ProcessingParams params;

  // Extract parameters from maps
  for (const auto &p : bool_params) {
    if (p.first == "use_camera_wb")
      params.use_camera_wb = p.second;
    else if (p.first == "half_size")
      params.half_size = p.second;
    else if (p.first == "four_color_rgb")
      params.four_color_rgb = p.second;
    else if (p.first == "use_auto_wb")
      params.use_auto_wb = p.second;
    else if (p.first == "no_auto_bright")
      params.no_auto_bright = p.second;
    else if (p.first == "dcb_enhance")
      params.dcb_enhance = p.second;
    else if (p.first == "no_auto_scale")
      params.no_auto_scale = p.second;
    else if (p.first == "use_gpu_acceleration")
      params.use_gpu_acceleration = p.second;
    else if (p.first == "preprocess")
      params.preprocess = p.second;
    else if (p.first == "defringe")
      params.defringe = p.second;
    else if (p.first == "defringe_green")
      params.defringe_green = p.second;
    else if (p.first == "lateral_ca_correction")
      params.lateral_ca_correction = p.second;
    else if (p.first == "axial_ca_correction")
      params.axial_ca_correction = p.second;
  }

  for (const auto &p : int_params) {

    if (p.first == "demosaic_algorithm")
      params.demosaic_algorithm = p.second;
    else if (p.first == "output_bps")
      params.output_bps = p.second;
    else if (p.first == "user_flip")
      params.user_flip = p.second;
    else if (p.first == "dcb_iterations")
      params.dcb_iterations = p.second;
    else if (p.first == "fbdd_noise_reduction")
      params.fbdd_noise_reduction = p.second;
    else if (p.first == "median_filter_passes")
      params.median_filter_passes = p.second;
    else if (p.first == "output_color_space")
      params.output_color_space = p.second;
    else if (p.first == "highlight_mode")
      params.highlight_mode = p.second;
    else if (p.first == "user_black")
      params.user_black = p.second;
    else if (p.first == "user_sat")
      params.user_sat = p.second;
    else if (p.first == "lateral_ca_cell_size")
      params.lateral_ca_cell_size = p.second;
    else if (p.first == "lateral_ca_max_iterations")
      params.lateral_ca_max_iterations = p.second;
    else if (p.first == "lateral_ca_pyramid_levels")
      params.lateral_ca_pyramid_levels = p.second;
    else if (p.first == "axial_ca_radius")
      params.axial_ca_radius = p.second;
  }

  for (const auto &p : float_params) {
    if (p.first == "noise_thr")
      params.noise_thr = p.second;
    else if (p.first == "bright")
      params.bright = p.second;
    else if (p.first == "auto_bright_thr")
      params.auto_bright_thr = p.second;
    else if (p.first == "adjust_maximum_thr")
      params.adjust_maximum_thr = p.second;
    else if (p.first == "exp_shift")
      params.exp_shift = p.second;
    else if (p.first == "exp_preserve_highlights")
      params.exp_preserve_highlights = p.second;
    else if (p.first == "gamma_power")
      params.gamma_power = p.second;
    else if (p.first == "gamma_slope")
      params.gamma_slope = p.second;
    else if (p.first == "chromatic_aberration_red")
      params.chromatic_aberration_red = p.second;
    else if (p.first == "chromatic_aberration_blue")
      params.chromatic_aberration_blue = p.second;
    else if (p.first == "defringe_radius")
      params.defringe_radius = p.second;
    else if (p.first == "defringe_strength")
      params.defringe_strength = p.second;
    else if (p.first == "defringe_green_strength")
      params.defringe_green_strength = p.second;
    else if (p.first == "lateral_ca_max_shift")
      params.lateral_ca_max_shift = p.second;
    else if (p.first == "lateral_ca_min_confidence")
      params.lateral_ca_min_confidence = p.second;
    else if (p.first == "axial_ca_epsilon")
      params.axial_ca_epsilon = p.second;
    else if (p.first == "axial_ca_strength")
      params.axial_ca_strength = p.second;
  }

  for (const auto &p : string_params) {
    if (p.first == "bad_pixels_path")
      params.bad_pixels_path = p.second;
  }

  // Apply parameters and process
  pimpl->set_processing_params(params);

  // CRITICAL: Unpack the RAW data before processing (only if not already
  // unpacked)
  if (!(pimpl->processor.imgdata.progress_flags & LIBRAW_PROGRESS_LOAD_RAW)) {
    int unpack_result = pimpl->unpack();
    if (unpack_result != LIBRAW_SUCCESS) {

      ProcessedImageData error_result;
      error_result.error_code = unpack_result;
      return error_result;
    }
  }

  // Process the image
  int process_result = pimpl->process();

  if (process_result != LIBRAW_SUCCESS) {
    ProcessedImageData error_result;
    error_result.error_code = process_result;
    return error_result;
  }

  return pimpl->get_processed_image();
}

void LibRawWrapper::close() {
  // Reset LibRaw processor
  pimpl->processor.recycle();
}

#ifdef __arm64__

// ---------------------------------------------------------------
// Standalone numpy-based image processing methods
// ---------------------------------------------------------------

// Helper: compute default threshold = maximum / data_maximum
static float compute_default_threshold(LibRaw &processor) {
  float maximum = (float)processor.imgdata.color.maximum;
  float data_maximum = (float)processor.imgdata.color.data_maximum;
  if (data_maximum > 0.f && maximum > 0.f) {
    return maximum / data_maximum;
  }
  return 1.f; // fallback: no adjustment
}

float LibRawWrapper::get_threshold() const {
  return compute_default_threshold(pimpl->processor);
}

py::dict LibRawWrapper::get_output_geometry_dict(bool half_size) const {
  const auto &imgdata = pimpl->processor.imgdata;
  const auto &sizes = imgdata.sizes;

  int width = static_cast<int>(sizes.width);
  int height = static_cast<int>(sizes.height);

  const bool can_shrink = half_size && imgdata.idata.filters &&
                          !imgdata.rawdata.color4_image &&
                          !imgdata.rawdata.color3_image;
  const int shrink = can_shrink ? 1 : 0;
  if (can_shrink) {
    width = (width + 1) >> 1;
    height = (height + 1) >> 1;
  }

  const bool is_fuji_honeycomb =
      imgdata.rawdata.ioparams.fuji_width > 0 && imgdata.idata.filters >= 1000;
  const bool use_fuji_rotate = imgdata.params.use_fuji_rotate != 0;
  int rotated_width = width;
  int rotated_height = height;

  if (is_fuji_honeycomb && use_fuji_rotate) {
    const int fuji_width =
        (imgdata.rawdata.ioparams.fuji_width - 1 + shrink) >> shrink;
    if (fuji_width > 0) {
      const double step = std::sqrt(0.5);
      const int wide = static_cast<int>(fuji_width / step);
      const int high = static_cast<int>((height - fuji_width) / step);
      if (wide > 1 && high > 1) {
        rotated_width = wide;
        rotated_height = high;
      }
    }
  }

  py::dict result;
  result["width"] = rotated_width;
  result["height"] = rotated_height;
  result["is_fuji_honeycomb"] = is_fuji_honeycomb;
  result["use_fuji_rotate"] = use_fuji_rotate;
  result["is_fuji_rotated_output"] = is_fuji_honeycomb && use_fuji_rotate;
  return result;
}

py::array_t<float>
LibRawWrapper::recover_highlights_numpy(py::array_t<float> image,
                                        float threshold) {
  py::buffer_info buf = image.request();
  if (buf.ndim != 3 || buf.shape[2] != 3) {
    throw std::invalid_argument(
        "recover_highlights_numpy: image must be shape (H, W, 3) float32");
  }
  if (threshold < 0.f) {
    threshold = compute_default_threshold(pimpl->processor);
  }

  const size_t height = buf.shape[0];
  const size_t width = buf.shape[1];
  const size_t channels = 3;
  const size_t num_pixels = height * width;

  // Allocate output array
  py::array_t<float> output({height, width, channels});
  py::buffer_info out_buf = output.request();

  // Copy input to output
  std::memcpy(out_buf.ptr, buf.ptr, num_pixels * channels * sizeof(float));

  // Wrap output buffer as ImageBufferFloat (in-place on the copy)
  ImageBufferFloat rgb_buffer;
  rgb_buffer.image = reinterpret_cast<float(*)[3]>(out_buf.ptr);
  rgb_buffer.width = width;
  rgb_buffer.height = height;
  rgb_buffer.channels = channels;

  pimpl->recover_highlights(rgb_buffer, threshold);

  return output;
}

py::array_t<float> LibRawWrapper::tone_mapping_numpy(py::array_t<float> image,
                                                     float after_scale) {
  py::buffer_info buf = image.request();
  if (buf.ndim != 3 || buf.shape[2] != 3) {
    throw std::invalid_argument(
        "tone_mapping_numpy: image must be shape (H, W, 3) float32");
  }

  const size_t height = buf.shape[0];
  const size_t width = buf.shape[1];
  const size_t channels = 3;
  const size_t num_pixels = height * width;

  // Allocate output array
  py::array_t<float> output({height, width, channels});
  py::buffer_info out_buf = output.request();

  // Copy input to output
  std::memcpy(out_buf.ptr, buf.ptr, num_pixels * channels * sizeof(float));

  // Wrap as ImageBufferFloat
  ImageBufferFloat rgb_buffer;
  rgb_buffer.image = reinterpret_cast<float(*)[3]>(out_buf.ptr);
  rgb_buffer.width = width;
  rgb_buffer.height = height;
  rgb_buffer.channels = channels;

  if (!pimpl->accelerator) {
    throw std::runtime_error("tone_mapping_numpy: accelerator not initialized");
  }
  pimpl->accelerator->tone_mapping(rgb_buffer, rgb_buffer, after_scale);

  return output;
}

py::array_t<float>
LibRawWrapper::enhance_micro_contrast_numpy(py::array_t<float> image,
                                            float threshold, float strength,
                                            float target_contrast) {
  py::buffer_info buf = image.request();
  if (buf.ndim != 3 || buf.shape[2] != 3) {
    throw std::invalid_argument(
        "enhance_micro_contrast_numpy: image must be shape (H, W, 3) float32");
  }
  if (threshold < 0.f) {
    threshold = compute_default_threshold(pimpl->processor);
  }

  const size_t height = buf.shape[0];
  const size_t width = buf.shape[1];
  const size_t channels = 3;
  const size_t num_pixels = height * width;

  // Allocate output array
  py::array_t<float> output({height, width, channels});
  py::buffer_info out_buf = output.request();

  // Copy input to output
  std::memcpy(out_buf.ptr, buf.ptr, num_pixels * channels * sizeof(float));

  // Wrap as ImageBufferFloat
  ImageBufferFloat rgb_buffer;
  rgb_buffer.image = reinterpret_cast<float(*)[3]>(out_buf.ptr);
  rgb_buffer.width = width;
  rgb_buffer.height = height;
  rgb_buffer.channels = channels;

  if (!pimpl->accelerator) {
    throw std::runtime_error(
        "enhance_micro_contrast_numpy: accelerator not initialized");
  }
  pimpl->accelerator->enhance_micro_contrast(rgb_buffer, rgb_buffer, threshold,
                                             strength, target_contrast);

  return output;
}

py::array_t<float>
LibRawWrapper::defringe_numpy(py::array_t<float> image,
                              float radius,
                              float strength,
                              bool defringe_green) {
  py::buffer_info buf = image.request();
  if (buf.ndim != 3 || buf.shape[2] != 3) {
    throw std::invalid_argument(
        "defringe_numpy: image must be shape (H, W, 3) float32");
  }

  const size_t height   = buf.shape[0];
  const size_t width    = buf.shape[1];
  const size_t channels = 3;
  const size_t N        = height * width;

  // Allocate output and copy input
  py::array_t<float> output({height, width, channels});
  py::buffer_info out_buf = output.request();
  std::memcpy(out_buf.ptr, buf.ptr, N * channels * sizeof(float));

  // Wrap input and output as ImageBufferFloat
  ImageBufferFloat rgb_in, rgb_out;
  rgb_in.image    = reinterpret_cast<float(*)[3]>(buf.ptr);
  rgb_in.width    = width;
  rgb_in.height   = height;
  rgb_in.channels = channels;

  rgb_out.image    = reinterpret_cast<float(*)[3]>(out_buf.ptr);
  rgb_out.width    = width;
  rgb_out.height   = height;
  rgb_out.channels = channels;

  if (!pimpl->accelerator) {
    throw std::runtime_error("defringe_numpy: accelerator not initialized");
  }
  pimpl->accelerator->defringe(rgb_in, rgb_out,
                                radius, 0.1f, 0.15f, strength, defringe_green);
  return output;
}

#endif // __arm64__

// rawpy完全互換性のための処理パラメータ変換関数
ProcessingParams create_params_from_rawpy_args(
    // Basic parameters
    bool use_camera_wb, bool half_size, bool four_color_rgb, int output_bps,
    int user_flip,

    // Demosaicing parameters
    int demosaic_algorithm, int dcb_iterations, bool dcb_enhance,

    // Noise reduction parameters
    int fbdd_noise_reduction, float noise_thr, int median_filter_passes,

    // White balance parameters
    bool use_auto_wb, const std::array<float, 4> &user_wb,

    // Color and output parameters
    int output_color,

    // Brightness and exposure parameters
    float bright, bool no_auto_bright, float auto_bright_thr,
    float adjust_maximum_thr,

    // Highlight processing
    int highlight_mode,

    // Exposure correction parameters
    float exp_shift, float exp_preserve_highlights,

    // Gamma and scaling
    const std::pair<float, float> &gamma, bool no_auto_scale,

    // Color correction parameters
    float chromatic_aberration_red, float chromatic_aberration_blue,

    // User adjustments
    int user_black, int user_sat,

    // File-based corrections
    const std::string &bad_pixels_path,

    // LibRaw Enhanced extensions
    bool use_gpu_acceleration, bool preprocess) {
  ProcessingParams params;

  // Map all parameters to ProcessingParams structure

  // Basic processing parameters
  params.use_camera_wb = use_camera_wb;
  params.half_size = half_size;
  params.four_color_rgb = four_color_rgb;
  params.output_bps = output_bps;
  params.user_flip = user_flip;

  // Demosaicing parameters
  params.demosaic_algorithm = demosaic_algorithm;
  params.dcb_iterations = dcb_iterations;
  params.dcb_enhance = dcb_enhance;

  // Noise reduction parameters
  params.fbdd_noise_reduction = fbdd_noise_reduction;
  params.noise_thr = noise_thr;
  params.median_filter_passes = median_filter_passes;

  // White balance parameters
  params.use_auto_wb = use_auto_wb;
  params.user_wb[0] = user_wb[0];
  params.user_wb[1] = user_wb[1];
  params.user_wb[2] = user_wb[2];
  params.user_wb[3] = user_wb[3];

  // Color space and output parameters
  params.output_color_space = output_color;

  // Brightness and exposure parameters
  params.bright = bright;
  params.no_auto_bright = no_auto_bright;
  params.auto_bright_thr = auto_bright_thr;
  params.adjust_maximum_thr = adjust_maximum_thr;

  // Highlight processing
  params.highlight_mode = highlight_mode;

  // Exposure correction parameters
  params.exp_shift = exp_shift;
  params.exp_preserve_highlights = exp_preserve_highlights;

  // Gamma correction parameters
  params.gamma_power = gamma.first;
  params.gamma_slope = gamma.second;
  params.no_auto_scale = no_auto_scale;

  // Color correction parameters
  params.chromatic_aberration_red = chromatic_aberration_red;
  params.chromatic_aberration_blue = chromatic_aberration_blue;

  // User adjustments
  params.user_black = user_black;
  params.user_sat = user_sat;

  // File-based corrections
  params.bad_pixels_path = bad_pixels_path;

  // Metal-specific settings
  params.use_gpu_acceleration = use_gpu_acceleration;
  params.preprocess = preprocess;

  return params;
}

// Platform detection functions implementation

bool is_apple_silicon() {
#ifdef __arm64__
  return true;
#else
  return false;
#endif
}

bool is_available() {
  // Backward compatibility helper used by Python package initialization.
#ifdef __arm64__
  return true;
#else
  return false;
#endif
}

std::vector<std::string> get_device_list() {
  std::vector<std::string> device_list;

#ifdef __arm64__
#ifdef __OBJC__
  @autoreleasepool {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
      std::string device_name = std::string([device.name UTF8String]);
      device_list.push_back(device_name);
    }
  }
#endif
#endif

  return device_list;
}

} // namespace libraw_enhanced
