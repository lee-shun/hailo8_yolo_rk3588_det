/**
 * Test Case 2: 1280x1280 -> RGA 格式转换 -> Hailo 单张直接推理 (Batch=1)
 *
 * HailoRT 4.23 API (基于 infer_model.hpp)
 *
 * 编译:
 *   g++ -std=c++17 test_full_image.cpp -o test_full_image \
 *       -I/usr/include/hailo -I/usr/include/rga \
 *       -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
 *       -lrga -lhailort -lpthread
 */
#include "hailo/hailort.hpp"
#include "im2d.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/mman.h>
#include <vector>

using namespace hailort;

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size) {
  auto addr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (addr == MAP_FAILED)
    throw std::bad_alloc();
  return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t *>(addr),
                                  [size](uint8_t *p) { munmap(p, size); });
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model_1280.hef> <image_1280.jpg>"
              << std::endl;
    return 1;
  }

  if (imcheckHeader() != IM_STATUS_SUCCESS) {
    std::cerr << "RGA header version mismatch!" << std::endl;
    return 1;
  }
  std::cout << "RGA Version: " << querystring(RGA_VERSION) << std::endl;

  const std::string hef_path = argv[1];
  const std::string img_path = argv[2];

  // ========== 1. 读取原图 ==========
  cv::Mat src = cv::imread(img_path);
  if (src.empty()) {
    std::cerr << "Failed to load image: " << img_path << std::endl;
    return 1;
  }
  if (src.cols != 1280 || src.rows != 1280) {
    cv::resize(src, src, cv::Size(1280, 1280));
  }

  // ========== 2. RGA BGR -> RGB ==========
  auto t_rga_start = std::chrono::high_resolution_clock::now();

  cv::Mat dst_rgb(1280, 1280, CV_8UC3);
  rga_buffer_t src_buf =
      wrapbuffer_virtualaddr(src.data, 1280, 1280, RK_FORMAT_BGR_888);
  rga_buffer_t dst_buf =
      wrapbuffer_virtualaddr(dst_rgb.data, 1280, 1280, RK_FORMAT_RGB_888);

  IM_STATUS ret =
      imcvtcolor(src_buf, dst_buf, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
  if (ret != IM_STATUS_SUCCESS) {
    std::cerr << "imcvtcolor failed: " << imStrError(ret) << std::endl;
    return 1;
  }

  auto t_rga_end = std::chrono::high_resolution_clock::now();
  auto rga_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    t_rga_end - t_rga_start)
                    .count();

  // ========== 3. Hailo 单张推理 ==========
  auto t_hailo_start = std::chrono::high_resolution_clock::now();

  auto vdevice = VDevice::create().expect("Failed to create VDevice");
  auto infer_model = vdevice->create_infer_model(hef_path).expect(
      "Failed to create infer model");

  infer_model->set_batch_size(1);

  auto configured_infer_model =
      infer_model->configure().expect("Failed to configure");

  auto bindings = configured_infer_model.create_bindings().expect(
      "Failed to create bindings");

  const auto &input_streams = infer_model->inputs();
  const auto &output_streams = infer_model->outputs();

  // 设置输入
  if (!input_streams.empty()) {
    size_t input_size = input_streams[0].get_frame_size();
    auto input_binding =
        bindings.input(input_streams[0].name()).expect("input binding failed");
    input_binding.set_buffer(MemoryView(dst_rgb.data, input_size));
  }

  // 设置输出（页对齐）
  std::vector<std::shared_ptr<uint8_t>> output_buffers;
  for (const auto &s : output_streams) {
    size_t output_size = s.get_frame_size();
    auto buf = page_aligned_alloc(output_size);
    output_buffers.push_back(buf);
    auto output_binding =
        bindings.output(s.name()).expect("output binding failed");
    output_binding.set_buffer(MemoryView(buf.get(), output_size));
  }

  hailo_status status = configured_infer_model.activate();
  if (status != HAILO_SUCCESS) {
    std::cerr << "Failed to activate: " << status << std::endl;
    return 1;
  }

  status =
      configured_infer_model.run(bindings, std::chrono::milliseconds(1000));
  if (status != HAILO_SUCCESS) {
    std::cerr << "Inference failed: " << status << std::endl;
    return 1;
  }

  auto t_hailo_end = std::chrono::high_resolution_clock::now();
  auto hailo_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      t_hailo_end - t_hailo_start)
                      .count();

  // ========== 4. 后处理 ==========
  std::cout << "\n========== Test Case 2: Full Image Direct =========="
            << std::endl;
  std::cout << "RGA convert:      " << rga_us << " us (" << rga_us / 1000.0
            << " ms)" << std::endl;
  std::cout << "Hailo inference:  " << hailo_us << " us (" << hailo_us / 1000.0
            << " ms)" << std::endl;
  std::cout << "End-to-end:       " << (rga_us + hailo_us) << " us ("
            << (rga_us + hailo_us) / 1000.0 << " ms)" << std::endl;

  return 0;
}
