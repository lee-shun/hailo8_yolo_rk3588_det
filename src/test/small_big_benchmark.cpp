#include "yolo_detector.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct BenchResult {
  double total_ms;
  double fps;
  double ms_per_frame;
};

BenchResult benchmark(YoloDetector &det, const uint8_t *data, int loop) {
  // Warmup
  for (int i = 0; i < 10; ++i) {
    std::memcpy(det.input_ptr(), data, det.input_bytes());
    auto dets = det.infer();
    (void)dets;
  }
  // Benchmark
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < loop; ++i) {
    std::memcpy(det.input_ptr(), data, det.input_bytes());
    auto dets = det.infer();
    (void)dets;
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return {ms, loop / (ms / 1000.0), ms / loop};
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <mode> <hef_path> <image_path> [loop_count]\n"
              << "  mode:\n"
              << "    big   - single frame inference (batch=1), resize to "
                 "model input size\n"
              << "    small - tiled batch inference (batch=6), 3x2 grid "
                 "640x540->640x640\n"
              << "  e.g.:\n"
              << "    " << argv[0]
              << " big   yolov8n_1920x1088.hef img.jpg 100\n"
              << "    " << argv[0]
              << " small yolov8n_640x640.hef   img.jpg 100\n";
    return 1;
  }

  const std::string mode = argv[1];
  const std::string hef_path = argv[2];
  const std::string img_path = argv[3];
  const int LOOP = (argc >= 5) ? std::stoi(argv[4]) : 100;

  cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Failed to load image: " << img_path << "\n";
    return 1;
  }
  cv::Mat rgb;
  cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

  // ============================================================
  // 1) Big Mode: 单帧 batch=1，直接 resize 到模型输入尺寸
  // ============================================================
  if (mode == "big") {
    YoloDetector det;
    if (!det.load(hef_path, 0.25f, 0.45f, 1)) {
      std::cerr << "Failed to load model\n";
      return 1;
    }

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(det.input_width(), det.input_height()));
    size_t bytes = resized.total() * resized.elemSize();
    std::vector<uint8_t> buf(bytes);
    std::memcpy(buf.data(), resized.data, bytes);

    auto r = benchmark(det, buf.data(), LOOP);
    std::cout << "\n[Big Mode: " << det.input_width() << "x"
              << det.input_height() << ", batch=1]\n";
    std::cout << "  Total time : " << r.total_ms << " ms\n";
    std::cout << "  FPS        : " << r.fps << "\n";
    std::cout << "  ms/frame   : " << r.ms_per_frame << "\n";
  }
  // ============================================================
  // 2) Small Mode: batch=6，3x2 网格切分，底部 padding 到 640x640
  // ============================================================
  else if (mode == "small") {
    const uint16_t BATCH = 6;
    YoloDetector det;
    if (!det.load(hef_path, 0.25f, 0.45f, BATCH)) {
      std::cerr << "Failed to load model\n";
      return 1;
    }

    // 统一先 resize 到 1920x1080，保证两种模式对比的是同一张图的内容
    cv::Mat rgb_std;
    cv::resize(rgb, rgb_std, cv::Size(1920, 1080));

    const int PW = det.input_width();  // 640
    const int PH = det.input_height(); // 640
    const int CROP_H = 540;            // 1080 / 2
    const int PAD_H = PH - CROP_H;     // 640 - 540 = 100
    const int COLS = 3;
    const int ROWS = 2;

    size_t patch_bytes = PW * PH * 3;
    std::vector<uint8_t> buf(patch_bytes * BATCH);

    for (int r = 0; r < ROWS; ++r) {
      for (int c = 0; c < COLS; ++c) {
        int x = c * PW;
        int y = r * CROP_H;
        cv::Rect roi(x, y, PW, CROP_H);
        cv::Mat patch = rgb_std(roi).clone();

        // 底部 padding 100 像素黑边 -> 640x640
        cv::Mat padded;
        cv::copyMakeBorder(patch, padded, 0, PAD_H, 0, 0, cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));

        int idx = r * COLS + c;
        std::memcpy(buf.data() + idx * patch_bytes, padded.data, patch_bytes);
      }
    }

    auto r = benchmark(det, buf.data(), LOOP);
    std::cout << "\n[Small Mode: " << PW << "x" << PH << ", batch=" << BATCH
              << "]\n";
    std::cout << "  Total time : " << r.total_ms << " ms\n";
    std::cout << "  FPS        : " << r.fps << "\n";
    std::cout << "  ms/batch   : " << r.ms_per_frame << "\n";
    std::cout << "  (1 batch = " << BATCH
              << " tiles = 1 full 1920x1080 frame)\n";
  } else {
    std::cerr << "Unknown mode '" << mode << "'. Use 'big' or 'small'.\n";
    return 1;
  }

  return 0;
}
