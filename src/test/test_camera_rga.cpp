// test_rga_camera.cpp
// 编译:
//   g++ -O2 -std=c++17 test_rga_camera.cpp camera.cpp rga_engine.cpp \
//       -lrga -lmpp -lv4l2 -lpthread -o test_camera_rga

#include "camera.h"
#include "rga_engine.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "utility_tool/step_timer.h"

#define TEST_LOG(msg) std::cout << "[Test] " << msg << "\n"

// ============================================================================
// 7-Zone 几何配置（与 fusion_roi_predictor.h 保持一致）
// ============================================================================
struct ZoneConfig {
  int x, y;
  int w, h;
  const char *name;
};

static const ZoneConfig ZONES[7] = {
    {0, 0, 640, 640, "zone0"},      // 左上
    {640, 0, 640, 640, "zone1"},    // 中上
    {1280, 0, 640, 640, "zone2"},   // 右上
    {640, 220, 640, 640, "zone3"},  // 中心（关键区域）
    {0, 440, 640, 640, "zone4"},    // 左下
    {640, 440, 640, 640, "zone5"},  // 中下
    {1280, 440, 640, 640, "zone6"}, // 右下
};

static bool save_ppm(const char *path, const uint8_t *data, int w, int h) {
  std::ofstream f(path, std::ios::binary);
  if (!f)
    return false;
  f << "P6\n" << w << " " << h << "\n255\n";
  f.write(reinterpret_cast<const char *>(data), static_cast<size_t>(w) * h * 3);
  return f.good();
}

static bool save_pgm(const char *path, const uint8_t *data, int w, int h) {
  std::ofstream f(path, std::ios::binary);
  if (!f)
    return false;
  f << "P5\n" << w << " " << h << "\n255\n";
  f.write(reinterpret_cast<const char *>(data), static_cast<size_t>(w) * h);
  return f.good();
}

static bool is_all_zero(const uint8_t *ptr, size_t sz) {
  if (!ptr)
    return true;
  const size_t step = std::max<size_t>(1, sz / 1024);
  for (size_t i = 0; i < sz; i += step)
    if (ptr[i] != 0)
      return false;
  return true;
}

int main(int argc, char **argv) {
  std::string dev = (argc > 1) ? argv[1] : "/dev/video0";
  std::string fmt_str = (argc > 2) ? argv[2] : "yuyv";
  CameraFormat fmt =
      (fmt_str == "mjpeg") ? CameraFormat::MJPEG : CameraFormat::YUYV;

  Camera cam;
  RgaEngine rga;

  TEST_LOG("dev=" << dev << " fmt=" << fmt_str);

  // 1. 初始化 Camera
  if (!cam.init(dev, 1920, 1080, 60, fmt)) {
    std::cerr << "[Test] camera init failed\n";
    return EXIT_FAILURE;
  }
  TEST_LOG("camera init OK: " << cam.src_w() << "x" << cam.src_h());

  // 2. 初始化 RGA：7 个 ROI 640x640，灰度 320x180
  if (!rga.init(7, 640, 640, 320, 180)) {
    std::cerr << "[Test] rga init failed\n";
    return EXIT_FAILURE;
  }

  if (!cam.start()) {
    std::cerr << "[Test] camera start failed\n";
    return EXIT_FAILURE;
  }

  const int loop = 1000;
  int ok_frames = 0;
  int empty_frames = 0;

  for (int i = 0; i < 10; ++i)
    cam.grab();

  TIMER_TEST_BEGIN(cam_rga_test);
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < loop; ++i) {

    TIMER_STEP_START(cam_rga_test, grab);
    bool grab_ret = cam.grab();
    TIMER_STEP_END(cam_rga_test, grab);
    if (!grab_ret) {
      TEST_LOG("grab failed at frame " << i << " (empty or error)");
      ++empty_frames;
      continue;
    }

    uint8_t *src = cam.src_ptr();
    if (!src) {
      TEST_LOG("src_ptr is null, skip frame " << i);
      cam.release();
      continue;
    }

    int src_w = cam.src_w();
    int src_h = cam.src_h();
    int src_fmt = cam.src_fmt();

    // Camera YUYV 模式下 src_stride() 为 0，测试层兜底（不动 camera）
    int src_stride = cam.src_stride();
    if (src_stride <= 0) {
      if (fmt == CameraFormat::YUYV)
        src_stride = src_w * 2; // YUYV: 2 bytes/pixel
      else
        src_stride = src_w; // MJPEG->YUV420SP fallback
    }

    TEST_LOG("frame " << i << " src=" << static_cast<void *>(src)
                      << " stride=" << src_stride << " fmt=" << src_fmt);

    bool frame_ok = true;

    // 3. 全图 BGR 转换（零拷贝）
    TIMER_STEP_START(cam_rga_test, full_bgr);
    if (!rga.convert_full_bgr(src, src_stride, src_w, src_h, src_fmt)) {
      TEST_LOG("convert_full_bgr failed");
      frame_ok = false;
    } else {
      uint8_t *bgr = rga.full_bgr_ptr();
      // if (ok_frames == 0) { // 首帧成功保存
      //   save_ppm("/tmp/test_full_bgr.ppm", bgr, 1920, 1080);
      //   TEST_LOG("saved /tmp/test_full_bgr.ppm");
      // }
    }
    TIMER_STEP_END(cam_rga_test, full_bgr);

    // 4. 灰度 320x180（零拷贝）
    TIMER_STEP_START(cam_rga_test, gray);
    if (!rga.convert_gray(src, src_stride, src_w, src_h, src_fmt)) {
      TEST_LOG("convert_gray failed");
      frame_ok = false;
    } else {
      uint8_t *gray = rga.gray_ptr();
      // if (ok_frames == 0) {
      //   save_pgm("/tmp/test_gray.pgm", gray, 320, 180);
      //   TEST_LOG("saved /tmp/test_gray.pgm");
      // }
    }
    TIMER_STEP_END(cam_rga_test, gray);

    // 5. 裁剪 7 个 ROI（零拷贝）
    TIMER_STEP_START(cam_rga_test, crop);
    for (int z = 0; z < 7; ++z) {
      const auto &zone = ZONES[z];
      if (!rga.crop_roi(src, src_stride, src_w, src_h, src_fmt, zone.x, zone.y,
                        z)) {
        TEST_LOG("crop_roi " << zone.name << " failed");
        frame_ok = false;
        continue;
      }

      //   // 首帧成功时保存所有 7 个 ROI
      //   if (ok_frames == 0) {
      //     uint8_t *roi = rga.roi_ptr(z);
      //     char path[128];
      //     snprintf(path, sizeof(path), "/tmp/test_roi_%s.ppm", zone.name);
      //     save_ppm(path, roi, zone.w, zone.h);
      //     TEST_LOG("saved " << path);
      //   }
    }
    TIMER_STEP_END(cam_rga_test, crop);

    if (frame_ok)
      ++ok_frames;

    // RGA 已同步完成，安全释放 V4L2 buffer
    TIMER_STEP_START(cam_rga_test, release);
    cam.release();
    TIMER_STEP_END(cam_rga_test, release);

    TIMER_TEST_PEEK(cam_rga_test);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double fps = ok_frames / (ms / 1000.0);

  TIMER_TEST_END(cam_rga_test);

  TEST_LOG("finished: total=" << loop << " ok=" << ok_frames
                              << " empty=" << empty_frames << " elapsed=" << ms
                              << "ms fps=" << fps);

  cam.stop();
  TEST_LOG("camera stopped, test done");
  return (ok_frames > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
