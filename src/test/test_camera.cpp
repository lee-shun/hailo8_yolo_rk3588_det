#include "camera.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>
#include <utility_tool/step_timer.h>

// NV12 -> BGR，正确处理 MPP 的 hor_stride/ver_stride 对齐
// 使用 OpenCV Mat 的 ptr(row) 访问，自动处理 Mat 内部 step 对齐
static cv::Mat nv12_to_bgr(uint8_t *ptr, int w, int h, int hor_stride,
                           int ver_stride) {
  // 只有当 hor_stride == w 且 ver_stride == h 时，才是真正的紧凑 NV12
  if (hor_stride == w && ver_stride == h) {
    cv::Mat nv12(h * 3 / 2, w, CV_8UC1, ptr);
    cv::Mat bgr;
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
  }

  // 有 padding，拷贝到连续 buffer
  cv::Mat nv12(h * 3 / 2, w, CV_8UC1);
  int y_rows = h;
  int uv_rows = h / 2;

  // 拷贝 Y 平面：只取前 h 行有效数据
  for (int i = 0; i < y_rows; ++i)
    memcpy(nv12.ptr(i), ptr + (size_t)i * hor_stride, w);

  // 拷贝 UV 平面：从 hor_stride * ver_stride 偏移处开始（MPP 对齐后的物理布局）
  uint8_t *uv_src = ptr + (size_t)hor_stride * ver_stride;
  for (int i = 0; i < uv_rows; ++i)
    memcpy(nv12.ptr(h + i), uv_src + (size_t)i * hor_stride, w);

  cv::Mat bgr;
  cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

// YUYV -> BGR
static cv::Mat yuyv_to_bgr(uint8_t *ptr, int w, int h) {
  cv::Mat yuyv(h, w, CV_8UC2, ptr);
  cv::Mat bgr;
  cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);
  return bgr;
}

int main(int argc, char **argv) {
  std::string dev = "/dev/video0";
  int width = 1920, height = 1080, fps = 30;
  bool useMjpeg = false;

  int opt;
  while ((opt = getopt(argc, argv, "m")) != -1) {
    switch (opt) {
    case 'm':
      useMjpeg = true;
      break;
    default:
      std::cerr << "Usage: " << argv[0]
                << " [-m] [device] [width] [height] [fps]\n";
      return 1;
    }
  }

  if (argc - optind >= 1)
    dev = argv[optind];
  if (argc - optind >= 2)
    width = std::stoi(argv[optind + 1]);
  if (argc - optind >= 3)
    height = std::stoi(argv[optind + 2]);
  if (argc - optind >= 4)
    fps = std::stoi(argv[optind + 3]);

  CameraFormat fmt = useMjpeg ? CameraFormat::MJPEG : CameraFormat::YUYV;
  std::cout << "=== Camera Test ===\n";
  std::cout << "Device: " << dev << "\n";
  std::cout << "Resolution: " << width << "x" << height << " @ " << fps
            << "fps\n";
  std::cout << "Format: " << (fmt == CameraFormat::MJPEG ? "MJPEG" : "YUYV")
            << "\n\n";

  Camera cam;
  if (!cam.init(dev, width, height, fps, fmt)) {
    std::cerr << "Camera init failed\n";
    return -1;
  }

  if (!cam.start()) {
    std::cerr << "Camera start failed\n";
    return -1;
  }

  const int total_frames = 300;
  int ok_frames = 0;
  auto t_start = std::chrono::steady_clock::now();

  TIMER_TEST_BEGIN(mjpg_cam_test);

  for (int i = 0; i < total_frames; i++) {
    TIMER_STEP_START(mjpg_cam_test, grab);
    if (!cam.grab()) {
      std::cerr << "  grab() failed, skip\n";
      continue;
    }
    TIMER_STEP_END(mjpg_cam_test, grab);

    uint8_t *ptr = cam.src_ptr();
    int w = cam.src_w();
    int h = cam.src_h();
    int hor_stride = cam.src_stride();
    int ver_stride = cam.src_ver_stride();

    if (!ptr) {
      std::cerr << "  src_ptr() is null\n";
      cam.release();
      continue;
    }

    cv::Mat bgr;
    if (fmt == CameraFormat::MJPEG) {
      bgr = nv12_to_bgr(ptr, w, h, hor_stride, ver_stride);
    } else {
      bgr = yuyv_to_bgr(ptr, w, h);
    }

    if (i % 30 == 0) {
      std::cout << "Frame " << i << " " << w << "x" << h
                << " hor_stride=" << hor_stride << " ver_stride=" << ver_stride
                << "\n";
      cv::imwrite(std::to_string(i) + "_Camera.png", bgr);
    }

    cam.release();
    ok_frames++;
  }

  TIMER_TEST_END(mjpg_cam_test);

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  double actual_fps = ok_frames / elapsed;

  std::cout << "\n=== Summary ===\n";
  std::cout << "Total requested: " << total_frames << "\n";
  std::cout << "Successful: " << ok_frames << "\n";
  std::cout << "Elapsed: " << elapsed << "s\n";
  std::cout << "Actual FPS: " << actual_fps << "\n";

  cv::destroyAllWindows();
  cam.stop();
  return 0;
}
