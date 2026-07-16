#include "cam_dma_mmap.h"
#include "rga_cropper.h"
#include "yolo_detector.h"

#include <chrono>
#include <csignal>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <vector>

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
      << "  -o, --output <dir>     Output dir for saved tiles (default: "
         "./output)\n"
      << "  --save                 Save tiles once after warmup\n"
      << "  -h, --help             Show this help\n";
}

int main(int argc, char **argv) {
  std::signal(SIGINT, sig_handler);
  std::signal(SIGTERM, sig_handler);

  std::string dev = "/dev/video0";
  int width = 1920, height = 1080, fps = 30;
  std::string model_path;
  float conf = 0.25f, iou = 0.45f;
  int batch = 6;
  int loop = 100;
  int warmup = 10;
  int tile_w = 640, tile_h = 640, tile_cols = 3, tile_rows = 2;
  int overlap_w = -1, overlap_h = -1;
  std::string output_dir = "./output";
  bool save_tiles = false;

  static struct option long_opts[] = {{"dev", required_argument, 0, 'd'},
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
                                      {"help", no_argument, 0, 'h'},
                                      {0, 0, 0, 0}};

  int c, opt_idx = 0;
  while ((c = getopt_long(argc, argv, "d:W:H:f:m:c:i:b:l:w:t:u:C:R:x:y:o:sh",
                          long_opts, &opt_idx)) != -1) {
    switch (c) {
    case 'd':
      dev = optarg;
      break;
    case 'W':
      width = atoi(optarg);
      break;
    case 'H':
      height = atoi(optarg);
      break;
    case 'f':
      fps = atoi(optarg);
      break;
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      conf = std::stof(optarg);
      break;
    case 'i':
      iou = std::stof(optarg);
      break;
    case 'b':
      batch = atoi(optarg);
      break;
    case 'l':
      loop = atoi(optarg);
      break;
    case 'w':
      warmup = atoi(optarg);
      break;
    case 't':
      tile_w = atoi(optarg);
      break;
    case 'u':
      tile_h = atoi(optarg);
      break;
    case 'C':
      tile_cols = atoi(optarg);
      break;
    case 'R':
      tile_rows = atoi(optarg);
      break;
    case 'x':
      overlap_w = atoi(optarg);
      break;
    case 'y':
      overlap_h = atoi(optarg);
      break;
    case 'o':
      output_dir = optarg;
      break;
    case 's':
      save_tiles = true;
      break;
    case 'h':
      print_usage(argv[0]);
      return 0;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  if (model_path.empty()) {
    std::cerr << "[Test] ERROR: --model is required\n";
    print_usage(argv[0]);
    return 1;
  }

  if (overlap_w < 0) {
    overlap_w =
        (tile_cols > 1) ? (tile_cols * tile_w - width) / (tile_cols - 1) : 0;
    if (overlap_w < 0)
      overlap_w = 0;
  }
  if (overlap_h < 0) {
    overlap_h =
        (tile_rows > 1) ? (tile_rows * tile_h - height) / (tile_rows - 1) : 0;
    if (overlap_h < 0)
      overlap_h = 0;
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
    std::cerr << "[Test] ERROR: tile_cols * tile_rows ("
              << tile_cols * tile_rows << ") must equal batch_size (" << batch
              << ")\n";
    return 1;
  }

  mkdir(output_dir.c_str(), 0755);

  std::cout << "[Test] ==================================================\n"
            << "[Test] Device    : " << dev << "\n"
            << "[Test] Capture   : " << width << "x" << height << "@" << fps
            << "\n"
            << "[Test] Model     : " << model_path << "\n"
            << "[Test] Batch     : " << batch << "\n"
            << "[Test] Tiles     : " << tile_w << "x" << tile_h
            << "  grid=" << tile_cols << "x" << tile_rows
            << "  overlap=" << overlap_w << "x" << overlap_h << "\n"
            << "[Test] Loop      : "
            << (loop == 0 ? "unlimited" : std::to_string(loop)) << "\n"
            << "[Test] ==================================================\n";

  // 1) Camera
  CamDmaMmap cam(CamDmaMmap::Mode::DMA);
  if (!cam.init(dev, width, height, fps, CamDmaMmap::Format::YUYV)) {
    std::cerr << "[Test] ERROR: Camera init failed\n";
    return 1;
  }
  if (!cam.start()) {
    std::cerr << "[Test] ERROR: Camera start failed\n";
    return 1;
  }

  // 2) RGA：每个 tile 独立 dmabuf
  RgaCropper cropper(width, height, tile_w, tile_h, tile_cols, tile_rows,
                     overlap_w, overlap_h);
  if (!cropper.init()) {
    std::cerr << "[Test] ERROR: RGA cropper init failed\n";
    return 1;
  }

  // 3) Hailo：真零拷贝，传入 tile fds
  std::vector<int> tile_fds;
  for (int i = 0; i < cropper.tile_count(); ++i) {
    tile_fds.push_back(cropper.tile_fd(i));
  }

  YoloDetector det;
  if (!det.load(model_path, conf, iou, (uint16_t)batch, tile_fds)) {
    std::cerr << "[Test] ERROR: Hailo model load failed\n";
    return 1;
  }
  if (det.input_width() != tile_w || det.input_height() != tile_h) {
    std::cerr << "[Test] ERROR: Model input size mismatch\n";
    return 1;
  }

  if (cropper.tile_size() != det.input_bytes() / det.batch_size()) {
    std::cerr << "[Test] ERROR: RGA tile_size (" << cropper.tile_size()
              << ") != Hailo single frame ("
              << det.input_bytes() / det.batch_size() << ")\n";
    return 1;
  }

  // 4) 主循环：抓图 -> RGA -> Hailo（零拷贝）
  std::vector<StageStats> stats;
  int frame_count = 0;
  bool saved = false;
  auto t_start = std::chrono::steady_clock::now();

  while (g_running) {
    if (loop > 0 && frame_count >= loop + warmup)
      break;

    StageStats s;
    auto t0 = std::chrono::steady_clock::now();

    if (!cam.grab()) {
      continue;
    }
    auto t1 = std::chrono::steady_clock::now();
    s.grab_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int src_fd = cam.src_fd();
    int src_fmt = cam.src_fmt();
    int src_w = cam.src_w();
    int src_h = cam.src_h();

    bool rga_ok = cropper.process(src_fd, src_fmt, src_w, src_h);
    auto t2 = std::chrono::steady_clock::now();
    s.rga_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    cam.release(); // 立即归还 V4L2 buffer

    if (!rga_ok) {
      std::cerr << "[Test] ERROR: RGA process failed\n";
      continue;
    }

    // 零拷贝：无 memcpy，RGA 输出已直接绑定到 Hailo
    auto t3 = t2;
    s.copy_ms = 0.0;

    auto all_dets = det.infer_all();
    auto t4 = std::chrono::steady_clock::now();
    s.infer_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    s.total_ms = std::chrono::duration<double, std::milli>(t4 - t0).count();
    frame_count++;

    if (frame_count > warmup) {
      stats.push_back(s);

      std::string det_counts;
      for (size_t b = 0; b < all_dets.size(); ++b) {
        det_counts += "b" + std::to_string(b) + ":" +
                      std::to_string(all_dets[b].size()) + " ";
      }

      std::cout << "[Test] Frame " << std::setw(4) << (frame_count - warmup)
                << " | grab=" << std::fixed << std::setprecision(2) << s.grab_ms
                << " rga=" << s.rga_ms << " copy=" << s.copy_ms
                << " infer=" << s.infer_ms << " total=" << s.total_ms
                << " | FPS=" << std::setprecision(1) << (1000.0 / s.total_ms)
                << " | dets=[" << det_counts << "]\n";
    } else {
      std::cout << "[Test] Warmup " << frame_count << "/" << warmup << "\n";
    }

    if (save_tiles && frame_count == warmup && !saved) {
      if (cropper.save_tiles(output_dir + "/tile")) {
        std::cout << "[Test] >>> Saved " << cropper.tile_count() << " tiles to "
                  << output_dir << "\n";
      }
      saved = true;
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double total_elapsed_s =
      std::chrono::duration<double>(t_end - t_start).count();

  if (!stats.empty()) {
    double avg_grab = 0, avg_rga = 0, avg_copy = 0, avg_infer = 0,
           avg_total = 0;
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

    std::cout
        << "\n[Test] ==================== BENCHMARK SUMMARY "
           "====================\n"
        << "[Test] Valid frames    : " << stats.size() << "\n"
        << "[Test] Avg grab        : " << avg_grab << " ms\n"
        << "[Test] Avg RGA         : " << avg_rga << " ms\n"
        << "[Test] Avg copy        : " << avg_copy << " ms\n"
        << "[Test] Avg Hailo infer : " << avg_infer << " ms\n"
        << "[Test] Avg total       : " << avg_total << " ms\n"
        << "[Test] Min total       : " << min_total << " ms\n"
        << "[Test] Max total       : " << max_total << " ms\n"
        << "[Test] Overall FPS     : " << (stats.size() / total_elapsed_s)
        << "\n"
        << "[Test] Theoretical FPS  : " << (1000.0 / avg_total) << "\n"
        << "[Test] =========================================================\n";
  }

  return 0;
}
