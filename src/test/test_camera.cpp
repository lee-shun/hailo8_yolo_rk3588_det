#include "camera.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility_tool/step_timer.h>

// 保存原始 NV12 数据为文件，可用 ffplay 播放验证：
// ffplay -f rawvideo -pixel_format nv12 -video_size 1920x1080 frame_0.yuv
static void save_yuv(const std::string &filename, uint8_t *ptr, int w, int h,
                     int stride) {
  // NV12: Y 平面 stride*h 字节，UV 平面 stride*h/2 字节
  int y_size = stride * h;
  int uv_size = stride * h / 2;

  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open " << filename << "\n";
    return;
  }
  ofs.write((char *)ptr, y_size + uv_size);
  std::cout << "Saved " << filename << " (" << (y_size + uv_size)
            << " bytes)\n";
}

// 简单验证 NV12 数据合理性（Y 平面值应在 0-255 之间，通常不会全 0 或全 255）
static bool sanity_check_nv12(uint8_t *ptr, int w, int h, int stride) {
  if (!ptr)
    return false;
  int y_size = stride * h;
  int zeros = 0, valid = 0;
  for (int i = 0; i < y_size; i += 100) { // 采样检查
    if (ptr[i] == 0)
      zeros++;
    else
      valid++;
  }
  std::cout << "  Sanity check: sampled " << (zeros + valid)
            << " pixels, zeros=" << zeros << "\n";
  return valid > 0; // 至少有一些非零值
}

int main(int argc, char **argv) {
  std::string dev = (argc > 1) ? argv[1] : "/dev/video0";
  int width = (argc > 2) ? std::stoi(argv[2]) : 1920;
  int height = (argc > 3) ? std::stoi(argv[3]) : 1080;
  int fps = (argc > 4) ? std::stoi(argv[4]) : 30;
  CameraFormat fmt = CameraFormat::MJPEG;

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

  const int total_frames = 300; // 抓 30 帧
  int ok_frames = 0;
  auto t_start = std::chrono::steady_clock::now();


  TIMER_TEST_BEGIN(mjpg_cam_test);

  for (int i = 0; i < total_frames; i++) {
    std::cout << "\n--- Frame " << i << " ---\n";

    TIMER_STEP_START(mjpg_cam_test, grab);
    if (!cam.grab()) {
      std::cerr << "  grab() failed, skip\n";
      continue;
    }
    TIMER_STEP_END(mjpg_cam_test, grab);

    uint8_t *ptr = cam.src_ptr();
    int w = cam.src_w();
    int h = cam.src_h();
    int stride = cam.src_stride();
    int rga_fmt = cam.src_fmt();

    if (!ptr) {
      std::cerr << "  src_ptr() is null\n";
      cam.release();
      continue;
    }

    std::cout << "  Output: " << w << "x" << h << " stride=" << stride
              << " rga_fmt=" << rga_fmt << "\n";

    // 打印前 16 字节（Y 平面开头）
    std::cout << "  Y-plane first 16 bytes: ";
    for (int j = 0; j < 16 && j < stride; j++) {
      std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)ptr[j]
                << " ";
    }
    std::cout << std::dec << "\n";

    // 数据合理性检查
    if (fmt == CameraFormat::MJPEG) {
      if (!sanity_check_nv12(ptr, w, h, stride)) {
        std::cerr << "  Warning: NV12 sanity check failed\n";
      }
    }

    // 保存第 0 帧和第 15 帧为文件，方便外部验证
    if (i == 0 || i == 15) {
      std::string fname = "frame_" + std::to_string(i) + ".yuv";
      save_yuv(fname, ptr, w, h, stride);
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

  cam.stop();
  return 0;
}
