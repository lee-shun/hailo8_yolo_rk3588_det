#include "yolo_detector.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <hef_path> <image_path>\n";
    return 1;
  }

  const std::string hef_path = argv[1];
  const std::string img_path = argv[2];
  const int LOOP_CNT = 1000;

  // 1. 初始化 Hailo 检测器
  YoloDetector detector;
  if (!detector.load(hef_path, 0.25f, 0.45f)) {
    std::cerr << "Failed to load HEF model\n";
    return 1;
  }

  const int w = detector.input_width();
  const int h = detector.input_height();
  const size_t buf_bytes = detector.input_bytes();
  std::cout << "[INFO] Ready. Input: " << w << "x" << h
            << ", buf_bytes=" << buf_bytes << "\n";

  // 2. OpenCV 读取本地图片
  cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Failed to load image: " << img_path << "\n";
    return 1;
  }

  // 3. 预处理：Resize 到模型输入尺寸 + BGR->RGB
  cv::Mat resized, rgb;
  cv::resize(img, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  const size_t img_bytes =
      static_cast<size_t>(rgb.rows * rgb.cols * rgb.elemSize());
  const size_t copy_bytes = std::min(img_bytes, buf_bytes);

  // 4. 获取 detector 输入缓冲区指针，并将图片数据拷贝进去
  uint8_t *input_buf = detector.input_ptr();
  std::memcpy(input_buf, rgb.data, copy_bytes);

  // 5. Warm-up（排除首次推理的初始化开销）
  std::cout << "[INFO] Warming up 10 iterations...\n";
  try {
    for (int i = 0; i < 10; ++i) {
      auto dets = detector.infer();
      (void)dets;
    }
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Warm-up failed: " << e.what() << "\n";
    return 1;
  }

  // 6. 正式循环 1000 次，测量帧率
  std::cout << "[INFO] Benchmarking " << LOOP_CNT << " iterations...\n";
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < LOOP_CNT; ++i) {
    // 若需模拟真实场景（每帧重新喂图），取消下面注释：
    std::memcpy(input_buf, rgb.data, copy_bytes);

    try {
      auto dets = detector.infer();
      (void)dets;
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Inference failed at iteration " << i << ": "
                << e.what() << "\n";
      return 1;
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  double fps = LOOP_CNT / (elapsed_ms / 1000.0);
  double avg_ms = elapsed_ms / LOOP_CNT;

  std::cout << "========================================\n";
  std::cout << "  Total frames : " << LOOP_CNT << "\n";
  std::cout << "  Total time   : " << elapsed_ms << " ms\n";
  std::cout << "  Average FPS  : " << fps << "\n";
  std::cout << "  Avg latency  : " << avg_ms << " ms/frame\n";
  std::cout << "========================================\n";

  return 0;
}
