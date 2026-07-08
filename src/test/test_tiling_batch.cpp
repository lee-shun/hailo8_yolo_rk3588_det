/**
 * Test Case 1: 1280x1280 -> RGA 分块 4x640x640 -> Hailo Batch=4 推理
 *
 * HailoRT 4.23 API (基于 infer_model.hpp)
 *
 * 编译:
 *   g++ -std=c++17 test_tiling_batch.cpp -o test_tiling_batch \
 *       -I/usr/include/hailo -I/usr/include/rga \
 *       -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
 *       -lrga -lhailort -lpthread
 */
#include "hailo/hailort.hpp"
#include "im2d.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/mman.h>
#include <vector>

using namespace hailort;

struct ROI {
  int x, y, w, h;
};
const ROI ROIS[4] = {{0, 0, 640, 640},
                     {640, 0, 640, 640},
                     {0, 640, 640, 640},
                     {640, 640, 640, 640}};

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
    std::cerr << "Usage: " << argv[0] << " <model_640.hef> <image_1280.jpg>"
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

  // ========== 2. RGA 分块 + BGR->RGB ==========
  auto t_rga_start = std::chrono::high_resolution_clock::now();

  std::vector<cv::Mat> tiles_bgr(4);
  std::vector<cv::Mat> tiles_rgb(4);
  for (int i = 0; i < 4; i++) {
    tiles_bgr[i] = cv::Mat(640, 640, CV_8UC3);
    tiles_rgb[i] = cv::Mat(640, 640, CV_8UC3);
  }

  for (int i = 0; i < 4; i++) {
    // crop: 1280x1280 BGR -> 640x640 BGR
    rga_buffer_t src_buf =
        wrapbuffer_virtualaddr(src.data, 1280, 1280, RK_FORMAT_BGR_888);
    rga_buffer_t dst_bgr_buf =
        wrapbuffer_virtualaddr(tiles_bgr[i].data, 640, 640, RK_FORMAT_BGR_888);
    im_rect crop_rect = {ROIS[i].x, ROIS[i].y, ROIS[i].w, ROIS[i].h};

    IM_STATUS ret = imcrop(src_buf, dst_bgr_buf, crop_rect);
    if (ret != IM_STATUS_SUCCESS) {
      std::cerr << "imcrop failed on tile " << i << ": " << imStrError(ret)
                << std::endl;
      return 1;
    }

    // cvtcolor: 640x640 BGR -> 640x640 RGB
    rga_buffer_t src_bgr =
        wrapbuffer_virtualaddr(tiles_bgr[i].data, 640, 640, RK_FORMAT_BGR_888);
    rga_buffer_t dst_rgb =
        wrapbuffer_virtualaddr(tiles_rgb[i].data, 640, 640, RK_FORMAT_RGB_888);

    ret = imcvtcolor(src_bgr, dst_rgb, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    if (ret != IM_STATUS_SUCCESS) {
      std::cerr << "imcvtcolor failed on tile " << i << ": " << imStrError(ret)
                << std::endl;
      return 1;
    }
  }

  auto t_rga_end = std::chrono::high_resolution_clock::now();
  auto rga_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    t_rga_end - t_rga_start)
                    .count();

  // ========== 3. Hailo Batch=4 推理 ==========
  auto t_hailo_start = std::chrono::high_resolution_clock::now();

  auto vdevice = VDevice::create().expect("Failed to create VDevice");
  auto infer_model = vdevice->create_infer_model(hef_path).expect(
      "Failed to create infer model");

  // 设置 batch_size = 4
  infer_model->set_batch_size(4);
  // 可选: infer_model->set_power_mode(HAILO_POWER_MODE_ULTRA_PERFORMANCE);

  auto configured_infer_model =
      infer_model->configure().expect("Failed to configure");

  // 创建 bindings
  auto bindings = configured_infer_model.create_bindings().expect(
      "Failed to create bindings");

  // 获取输入/输出信息（用于计算 buffer 大小）
  const auto &input_streams = infer_model->inputs();
  const auto &output_streams = infer_model->outputs();

  std::cout << "Model inputs: " << input_streams.size()
            << ", outputs: " << output_streams.size() << std::endl;
  for (const auto &s : input_streams) {
    std::cout << "  Input: " << s.name()
              << ", frame_size=" << s.get_frame_size() << std::endl;
  }
  for (const auto &s : output_streams) {
    std::cout << "  Output: " << s.name()
              << ", frame_size=" << s.get_frame_size() << std::endl;
  }

  // 组装 batch input: 4 个 tile 连续排列
  const size_t TILE_SIZE = 640 * 640 * 3;
  std::vector<uint8_t> batch_input(TILE_SIZE * 4);
  for (int i = 0; i < 4; i++) {
    std::memcpy(batch_input.data() + i * TILE_SIZE, tiles_rgb[i].data,
                TILE_SIZE);
  }

  // 设置输入 buffer（单输入模型简化版）
  if (!input_streams.empty()) {
    size_t input_size = input_streams[0].get_frame_size() * 4; // batch=4
    auto input_binding =
        bindings.input(input_streams[0].name()).expect("input binding failed");
    input_binding.set_buffer(MemoryView(batch_input.data(), input_size));
  }

  // 设置输出 buffer（页对齐）
  std::vector<std::shared_ptr<uint8_t>> output_buffers;
  for (const auto &s : output_streams) {
    size_t output_size = s.get_frame_size() * 4; // batch=4
    auto buf = page_aligned_alloc(output_size);
    output_buffers.push_back(buf);
    auto output_binding =
        bindings.output(s.name()).expect("output binding failed");
    output_binding.set_buffer(MemoryView(buf.get(), output_size));
  }

  // 激活
  hailo_status status = configured_infer_model.activate();
  if (status != HAILO_SUCCESS) {
    std::cerr << "Failed to activate: " << status << std::endl;
    return 1;
  }

  // 推理
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

  // ========== 4. 后处理（框架） ==========
  // output_buffers 包含 4 个 tile 的结果
  // 需按 tile 索引解析，并将 local_x/y 映射回原图坐标

  std::cout << "\n========== Test Case 1: Tiling + Batch=4 =========="
            << std::endl;
  std::cout << "RGA crop + convert: " << rga_us << " us (" << rga_us / 1000.0
            << " ms)" << std::endl;
  std::cout << "Hailo inference:    " << hailo_us << " us ("
            << hailo_us / 1000.0 << " ms)" << std::endl;
  std::cout << "End-to-end:         " << (rga_us + hailo_us) << " us ("
            << (rga_us + hailo_us) / 1000.0 << " ms)" << std::endl;

  return 0;
}
