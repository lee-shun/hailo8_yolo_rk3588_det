#include "cam_dma_mmap.h"
#include "rga_cropper.h"
#include "yolo_detector.h"

#include <getopt.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <csignal>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <string>
#include <vector>

#include "utility_tool/step_timer.h"

static volatile bool g_running = true;
static void sig_handler(int) { g_running = false; }

struct StageStats {
  double grab_ms = 0.0;
  double rga_ms = 0.0;
  double copy_ms = 0.0;
  double infer_ms = 0.0;
  double total_ms = 0.0;
};

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  -d, --dev <path>       V4L2 device (default: /dev/video0)\n"
      << "  -W, --width <n>        Capture width (default: 1920)\n"
      << "  -H, --height <n>       Capture height (default: 1080)\n"
      << "  -f, --fps <n>          FPS (default: 30)\n"
      << "  -m, --model <path>     HEF model path (required)\n"
      << "  -c, --conf <f>         Confidence threshold (default: 0.25)\n"
      << "  -i, --iou <f>          IoU threshold (default: 0.45)\n"
      << "  -b, --batch <n>        Batch size (default: 6)\n"
      << "  -l, --loop <n>         Loop count, 0=unlimited (default: 100)\n"
      << "  -w, --warmup <n>       Warmup frames (default: 10)\n"
      << "  -t, --tile-w <n>       Tile width (default: 640)\n"
      << "  -u, --tile-h <n>       Tile height (default: 640)\n"
      << "  -x, --overlap-w <n>    Horizontal overlap (default: auto)\n"
      << "  -y, --overlap-h <n>    Vertical overlap (default: auto)\n"
      << "  -o, --output <dir>     Output dir for saved tiles (default: ./output)\n"
      << "  --save                 Save tiles once after warmup\n"
      << "  --save-det <n>         Save detection images every N frames (0=off)\n"
      << "  -h, --help             Show this help\n";
}

// ==================== Cross-Region NMS ====================

static float compute_iou(const Detection &a, const Detection &b) {
  float x1 = std::max(a.x, b.x);
  float y1 = std::max(a.y, b.y);
  float x2 = std::min(a.x + a.w, b.x + b.w);
  float y2 = std::min(a.y + a.h, b.y + b.h);
  if (x2 <= x1 || y2 <= y1) return 0.0f;
  float inter = (x2 - x1) * (y2 - y1);
  float area_a = a.w * a.h;
  float area_b = b.w * b.h;
  return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<Detection> cross_region_nms(
    const std::vector<std::vector<Detection>> &all_tile_dets,
    int tile_cols, int /*tile_rows*/, int tile_w, int tile_h,
    int overlap_w, int overlap_h, float iou_thresh) {
  int step_x = tile_w - overlap_w;
  int step_y = tile_h - overlap_h;
  std::vector<Detection> global;
  for (int i = 0; i < (int)all_tile_dets.size(); ++i) {
    int row = i / tile_cols;
    int col = i % tile_cols;
    float offset_x = (float)(col * step_x);
    float offset_y = (float)(row * step_y);
    for (const auto &det : all_tile_dets[i]) {
      Detection g;
      g.x = det.x + offset_x;
      g.y = det.y + offset_y;
      g.w = det.w;
      g.h = det.h;
      g.score = det.score;
      g.cls = det.cls;
      global.push_back(g);
    }
  }
  std::sort(global.begin(), global.end(),
            [](const Detection &a, const Detection &b) { return a.score > b.score; });
  std::vector<Detection> kept;
  for (const auto &det : global) {
    bool suppress = false;
    for (const auto &k : kept) {
      if (compute_iou(det, k) > iou_thresh) {
        suppress = true;
        break;
      }
    }
    if (!suppress) kept.push_back(det);
  }
  return kept;
}

// ===============================================================

int main(int argc, char **argv) {
  std::signal(SIGINT, sig_handler);
  std::signal(SIGTERM, sig_handler);

  // ---------- 默认参数 ----------
  std::string dev = "/dev/video0";
  int width = 1920, height = 1080, fps = 30;
  std::string model_path;
  float conf = 0.25f, iou = 0.45f;
  int batch = 6;
  int loop = 100;
  int warmup = 10;
  int tile_w = 640, tile_h = 640, tile_cols = 3, tile_rows = 2;
  int overlap_w = -1, overlap_h = -1; // -1 = auto
  std::string output_dir = "./output";
  bool save_tiles = false;
  int save_det_interval = 0;

  static struct option long_opts[] = {
      {"dev", required_argument, 0, 'd'},
      {"width", required_argument, 0, 'W'},
      {"height", required_argument, 0, 'H'},
      {"fps", required_argument, 0, 'f'},
      {"model", required_argument, 0, 'm'},
      {"conf", required_argument, 0, 'c'},
      {"iou", required_argument, 0, 'i'},
      {"batch", required_argument, 0, 'b'},
      {"loop", required_argument, 0, 'l'},
      {"warmup", required_argument, 0, 'w'},
      {"tile-w", required_argument, 0, 't'},
      {"tile-h", required_argument, 0, 'u'},
      {"cols", required_argument, 0, 'C'},
      {"rows", required_argument, 0, 'R'},
      {"overlap-w", required_argument, 0, 'x'},
      {"overlap-h", required_argument, 0, 'y'},
      {"output", required_argument, 0, 'o'},
      {"save", no_argument, 0, 's'},
      {"save-det", required_argument, 0, 'S'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int c, opt_idx = 0;
  while ((c = getopt_long(argc, argv,
                          "d:W:H:f:m:c:i:b:l:w:t:u:C:R:x:y:o:sS:h",
                          long_opts, &opt_idx)) != -1) {
    switch (c) {
      case 'd': dev = optarg; break;
      case 'W': width = atoi(optarg); break;
      case 'H': height = atoi(optarg); break;
      case 'f': fps = atoi(optarg); break;
      case 'm': model_path = optarg; break;
      case 'c': conf = std::stof(optarg); break;
      case 'i': iou = std::stof(optarg); break;
      case 'b': batch = atoi(optarg); break;
      case 'l': loop = atoi(optarg); break;
      case 'w': warmup = atoi(optarg); break;
      case 't': tile_w = atoi(optarg); break;
      case 'u': tile_h = atoi(optarg); break;
      case 'C': tile_cols = atoi(optarg); break;
      case 'R': tile_rows = atoi(optarg); break;
      case 'x': overlap_w = atoi(optarg); break;
      case 'y': overlap_h = atoi(optarg); break;
      case 'o': output_dir = optarg; break;
      case 's': save_tiles = true; break;
      case 'S': save_det_interval = atoi(optarg); break;
      case 'h': print_usage(argv[0]); return 0;
      default: print_usage(argv[0]); return 1;
    }
  }

  if (model_path.empty()) {
    std::cerr << "[Test] ERROR: --model is required\n";
    print_usage(argv[0]);
    return 1;
  }

  // ---------- 自动计算 overlap（使 tile 网格刚好覆盖源图）----------
  if (overlap_w < 0) {
    overlap_w = (tile_cols > 1) ? (tile_cols * tile_w - width) / (tile_cols - 1) : 0;
    if (overlap_w < 0) overlap_w = 0;
  }
  if (overlap_h < 0) {
    overlap_h = (tile_rows > 1) ? (tile_rows * tile_h - height) / (tile_rows - 1) : 0;
    if (overlap_h < 0) overlap_h = 0;
  }

  int required_w = tile_cols * tile_w - (tile_cols - 1) * overlap_w;
  int required_h = tile_rows * tile_h - (tile_rows - 1) * overlap_h;
  if (required_w > width || required_h > height) {
    std::cerr << "[Test] ERROR: Tile grid exceeds source image!\n"
              << "  Required: " << required_w << "x" << required_h << "\n"
              << "  Source:   " << width << "x" << height << "\n";
    return 1;
  }

  if (tile_cols * tile_rows != batch) {
    std::cerr << "[Test] ERROR: tile_cols * tile_rows (" << tile_cols * tile_rows
              << ") must equal batch_size (" << batch << ")\n";
    return 1;
  }

  mkdir(output_dir.c_str(), 0755);

  std::cout << "[Test] ==================================================\n"
            << "[Test] Device    : " << dev << "\n"
            << "[Test] Capture   : " << width << "x" << height << "@" << fps << "\n"
            << "[Test] Model     : " << model_path << "\n"
            << "[Test] Batch     : " << batch << "\n"
            << "[Test] Tiles     : " << tile_w << "x" << tile_h
            << "  grid=" << tile_cols << "x" << tile_rows
            << "  overlap=" << overlap_w << "x" << overlap_h << "\n"
            << "[Test] Loop      : " << (loop == 0 ? "unlimited" : std::to_string(loop)) << "\n"
            << "[Test] ==================================================\n";

  // ============================================================
  // 1) 初始化 Camera（DMA + YUYV）
  // ============================================================
  CamDmaMmap cam(CamDmaMmap::Mode::DMA);
  if (!cam.init(dev, width, height, fps, CamDmaMmap::Format::YUYV)) {
    std::cerr << "[Test] ERROR: Camera init failed\n";
    return 1;
  }
  if (!cam.start()) {
    std::cerr << "[Test] ERROR: Camera start failed\n";
    return 1;
  }

  // ============================================================
  // 2) 初始化 RgaCropper（色转 YUYV->RGB + Crop + 拼接）
  // ============================================================
  RgaCropper cropper(width, height, tile_w, tile_h, tile_cols, tile_rows, overlap_w, overlap_h);
  if (!cropper.init()) {
    std::cerr << "[Test] ERROR: RGA cropper init failed\n";
    return 1;
  }

  // mmap RGA 输出 buffer（只 mmap 一次，循环复用）
  void *dst_ptr = mmap(nullptr, cropper.dst_size(), PROT_READ, MAP_SHARED, cropper.dst_fd(), 0);
  if (dst_ptr == MAP_FAILED) {
    std::cerr << "[Test] ERROR: mmap dst_fd failed: " << strerror(errno) << "\n";
    return 1;
  }

  // ============================================================
  // 3) 初始化 Hailo YOLO（batch=6，输入 640x640 RGB）
  // ============================================================
  YoloDetector det;
  if (!det.load(model_path, conf, iou, (uint16_t)batch)) {
    std::cerr << "[Test] ERROR: Hailo model load failed\n";
    return 1;
  }
  if (det.input_width() != tile_w || det.input_height() != tile_h) {
    std::cerr << "[Test] ERROR: Model input size " << det.input_width() << "x"
              << det.input_height() << " does not match tile size " << tile_w << "x"
              << tile_h << "\n";
    return 1;
  }

  det.set_input_buffer((uint8_t *)dst_ptr);

  if (cropper.dst_size() != det.input_bytes()) {
    std::cerr << "[Test] ERROR: RGA dst_size (" << cropper.dst_size()
              << ") != Hailo input_bytes (" << det.input_bytes() << ")\n";
    return 1;
  }

  // ============================================================
  // 4) 主循环：抓图 -> RGA -> Hailo -> 解析（zero-copy）
  // ============================================================
  std::vector<StageStats> stats;
  int frame_count = 0;
  bool saved = false;
  auto t_start = std::chrono::steady_clock::now();

  TIMER_TEST_BEGIN(pipeline);

  cv::Mat save_bgr;
  while (g_running) {
    if (loop > 0 && frame_count >= loop + warmup) break;

    StageStats s;
    auto t0 = std::chrono::steady_clock::now();

    // ---- 4.1 抓图 ----
    TIMER_STEP_START(pipeline, grab);
    if (!cam.grab()) {
      continue;
    }
    TIMER_STEP_END(pipeline, grab);

    auto t1 = std::chrono::steady_clock::now();
    s.grab_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int src_fd = cam.src_fd();
    int src_fmt = cam.src_fmt();
    int src_w = cam.src_w();
    int src_h = cam.src_h();

    // ---- 4.2 RGA 色转 + Crop + 拼接（异步提交，job API）----
    TIMER_STEP_START(pipeline, rga_submit);
    bool rga_ok = cropper.process_async(src_fd, src_fmt, src_w, src_h, false);
    TIMER_STEP_END(pipeline, rga_submit);
    auto t2 = std::chrono::steady_clock::now();

    // RGA job 已提交，归还 V4L2 buffer
    bool save_this_frame = (save_det_interval > 0 && frame_count >= warmup &&
                             (frame_count - warmup + 1) % save_det_interval == 0);
    if (save_this_frame) {
      uint8_t *yuyv_mmap = nullptr;
      size_t yuyv_mmap_sz = 0;
      uint8_t *yuyv_raw = cam.src_ptr();
      if (yuyv_raw) {
        cv::Mat yuyv(cam.src_h(), cam.src_w(), CV_8UC2, yuyv_raw, cam.src_stride());
        cv::cvtColor(yuyv, save_bgr, cv::COLOR_YUV2BGR_YUYV);
      } else if (cam.src_fd() >= 0) {
        yuyv_mmap_sz = (size_t)cam.src_stride() * cam.src_h();
        yuyv_mmap = (uint8_t *)mmap(nullptr, yuyv_mmap_sz, PROT_READ, MAP_SHARED,
                                    cam.src_fd(), 0);
        if (yuyv_mmap != MAP_FAILED) {
          cv::Mat yuyv(cam.src_h(), cam.src_w(), CV_8UC2, yuyv_mmap, cam.src_stride());
          cv::cvtColor(yuyv, save_bgr, cv::COLOR_YUV2BGR_YUYV);
        }
      }
      if (yuyv_mmap && yuyv_mmap != MAP_FAILED) munmap(yuyv_mmap, yuyv_mmap_sz);
    }

    cam.release();

    if (!rga_ok) {
      std::cerr << "[Test] ERROR: RGA async submit failed\n";
      continue;
    }

    // 等待 RGA job 完成
    auto t2b = std::chrono::steady_clock::now();
    TIMER_STEP_START(pipeline, rga_wait);
    if (!cropper.wait_fence()) {
      std::cerr << "[Test] ERROR: RGA wait failed\n";
      continue;
    }
    TIMER_STEP_END(pipeline, rga_wait);
    auto t3 = std::chrono::steady_clock::now();

    // rga_ms = submit + wait 的总 RGA 耗时
    s.rga_ms = std::chrono::duration<double, std::milli>(t3 - t1).count();
    s.copy_ms = std::chrono::duration<double, std::milli>(t3 - t2b).count();

    // ---- 4.4 Hailo 推理 + 解析全部 batch ----
    TIMER_STEP_START(pipeline, infer);
    auto all_dets = det.infer_all();
    TIMER_STEP_END(pipeline, infer);
    auto t4 = std::chrono::steady_clock::now();
    s.infer_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // ---- 4.5 Cross-Region NMS ----
    TIMER_STEP_START(pipeline, cross_nms);
    auto merged_dets = cross_region_nms(all_dets, tile_cols, tile_rows,
                                        tile_w, tile_h, overlap_w, overlap_h, iou);
    TIMER_STEP_END(pipeline, cross_nms);

    TIMER_TEST_PEEK(pipeline);

    // ---- 4.6 Draw & Save (only frames with detections) ----
    if (save_this_frame && !save_bgr.empty() && !merged_dets.empty()) {
      for (const auto &d : merged_dets) {
        cv::Scalar blue(255, 0, 0);
        cv::Scalar white(255, 255, 255);
        cv::rectangle(save_bgr, cv::Point((int)d.x, (int)d.y),
                      cv::Point((int)(d.x + d.w), (int)(d.y + d.h)), blue, 2);
        std::string label = cv::format("%.2f", d.score);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Rect bg((int)d.x, (int)d.y - ts.height - 4, ts.width + 8, ts.height + 4);
        cv::rectangle(save_bgr, bg, blue, -1);
        cv::putText(save_bgr, label, cv::Point((int)d.x + 4, (int)d.y - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv::LINE_AA);
      }
      std::string timing = cv::format(
          "grab:%.1f rga:%.1f copy:%.1f infer:%.1f total:%.1f ms",
          s.grab_ms, s.rga_ms, s.copy_ms, s.infer_ms, s.total_ms);
      int baseline = 0;
      cv::Size ts = cv::getTextSize(timing, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
      cv::rectangle(save_bgr, cv::Point(0, 0), cv::Point(ts.width + 8, ts.height + 8),
                    cv::Scalar(0, 0, 0), -1);
      cv::putText(save_bgr, timing, cv::Point(4, ts.height + 4),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      std::string save_path = output_dir + "/det_" +
          std::to_string(frame_count - warmup) + ".jpg";
      cv::imwrite(save_path, save_bgr);
      std::cout << "[Test] >>> Saved detection image: " << save_path << " ("
                << merged_dets.size() << " dets)\n";
    }

    s.total_ms = std::chrono::duration<double, std::milli>(t4 - t0).count();
    frame_count++;

    // ---- 4.5 打印单帧信息 ----
    if (frame_count > warmup) {
      stats.push_back(s);

      // 统计每帧检测数量
      std::string det_counts;
      for (size_t b = 0; b < all_dets.size(); ++b) {
        det_counts += "b" + std::to_string(b) + ":" + std::to_string(all_dets[b].size()) + " ";
      }
      det_counts += "merged:" + std::to_string(merged_dets.size());

      std::cout << "[Test] Frame " << std::setw(4) << (frame_count - warmup)
                << " | grab=" << std::fixed << std::setprecision(2) << s.grab_ms
                << " rga=" << s.rga_ms
                << " copy=" << s.copy_ms
                << " infer=" << s.infer_ms
                << " total=" << s.total_ms
                << " | FPS=" << std::setprecision(1) << (1000.0 / s.total_ms)
                << " | dets=[" << det_counts << "]\n";
    } else {
      std::cout << "[Test] Warmup " << frame_count << "/" << warmup << "\n";
    }

    // ---- 4.6 保存 tile 图像（仅一次，可选）----
    if (save_tiles && frame_count == warmup && !saved) {
      std::string prefix = output_dir + "/tile";
      if (RgaCropper::save_tiles(cropper.dst_fd(), cropper.dst_size(),
                                 tile_w, tile_h, cropper.tile_count(), prefix)) {
        std::cout << "[Test] >>> Saved " << cropper.tile_count()
                  << " tiles to " << output_dir << "\n";
      }
      saved = true;
    }
  }

  // ============================================================
  // 5) 汇总统计
  // ============================================================
  auto t_end = std::chrono::steady_clock::now();
  double total_elapsed_s =
      std::chrono::duration<double>(t_end - t_start).count();

  if (!stats.empty()) {
    double avg_grab = 0, avg_rga = 0, avg_copy = 0, avg_infer = 0, avg_total = 0;
    double max_total = 0, min_total = 1e9;
    for (const auto &s : stats) {
      avg_grab += s.grab_ms;
      avg_rga += s.rga_ms;
      avg_copy += s.copy_ms;
      avg_infer += s.infer_ms;
      avg_total += s.total_ms;
      max_total = std::max(max_total, s.total_ms);
      min_total = std::min(min_total, s.total_ms);
    }
    avg_grab /= stats.size();
    avg_rga /= stats.size();
    avg_copy /= stats.size();
    avg_infer /= stats.size();
    avg_total /= stats.size();

    std::cout << "\n[Test] ==================== BENCHMARK SUMMARY ====================\n"
              << "[Test] Valid frames    : " << stats.size() << "\n"
              << "[Test] Avg grab        : " << avg_grab << " ms\n"
              << "[Test] Avg RGA         : " << avg_rga << " ms\n"
              << "[Test] Avg copy        : " << avg_copy << " ms\n"
              << "[Test] Avg Hailo infer : " << avg_infer << " ms\n"
              << "[Test] Avg total       : " << avg_total << " ms\n"
              << "[Test] Min total       : " << min_total << " ms\n"
              << "[Test] Max total       : " << max_total << " ms\n"
              << "[Test] Overall FPS     : " << (stats.size() / total_elapsed_s) << "\n"
              << "[Test] Theoretical FPS  : " << (1000.0 / avg_total) << "\n"
              << "[Test] =========================================================\n";
  }

  TIMER_TEST_END(pipeline);

  munmap(dst_ptr, cropper.dst_size());
  return 0;
}
