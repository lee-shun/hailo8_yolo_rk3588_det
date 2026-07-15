#include "cam_dma_mmap.h"
#include "rga_cropper.h"

#include <getopt.h>
#include <sys/stat.h>
#include <csignal>
#include <chrono>
#include <iostream>
#include <string>

static volatile bool g_running = true;

static void sig_handler(int) {
    g_running = false;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -d, --dev <path>      V4L2 device (default: /dev/video0)\n"
              << "  -W, --width <n>       Capture width (default: 1920)\n"
              << "  -H, --height <n>      Capture height (default: 1080)\n"
              << "  -f, --fps <n>         FPS (default: 30)\n"
              << "  -o, --output <dir>    Output directory (default: ./output)\n"
              << "  -n, --warmup <n>      Warmup frames before save (default: 10)\n"
              << "  -t, --tile-w <n>      Tile width (default: 640)\n"
              << "  -u, --tile-h <n>      Tile height (default: 540)\n"
              << "  -c, --cols <n>        Horizontal tiles (default: 3)\n"
              << "  -r, --rows <n>        Vertical tiles (default: 2)\n"
              << "  -F, --format <fmt>    YUYV or MJPEG (default: YUYV)\n"
              << "  --help                Show this help\n";
}

int main(int argc, char** argv) {
    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    std::string dev = "/dev/video0";
    int width = 1920, height = 1080, fps = 30;
    std::string output_dir = "./output";
    int warmup = 10;
    int tile_w = 640, tile_h = 540, tile_cols = 3, tile_rows = 2;
    std::string fmt_str = "YUYV";

    static struct option long_opts[] = {
        {"dev", required_argument, 0, 'd'},
        {"width", required_argument, 0, 'W'},
        {"height", required_argument, 0, 'H'},
        {"fps", required_argument, 0, 'f'},
        {"output", required_argument, 0, 'o'},
        {"warmup", required_argument, 0, 'n'},
        {"tile-w", required_argument, 0, 't'},
        {"tile-h", required_argument, 0, 'u'},
        {"cols", required_argument, 0, 'c'},
        {"rows", required_argument, 0, 'r'},
        {"format", required_argument, 0, 'F'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };

    int c, opt_idx = 0;
    while ((c = getopt_long(argc, argv, "d:W:H:f:o:n:t:u:c:r:F:", long_opts, &opt_idx)) != -1) {
        switch (c) {
            case 'd': dev = optarg; break;
            case 'W': width = atoi(optarg); break;
            case 'H': height = atoi(optarg); break;
            case 'f': fps = atoi(optarg); break;
            case 'o': output_dir = optarg; break;
            case 'n': warmup = atoi(optarg); break;
            case 't': tile_w = atoi(optarg); break;
            case 'u': tile_h = atoi(optarg); break;
            case 'c': tile_cols = atoi(optarg); break;
            case 'r': tile_rows = atoi(optarg); break;
            case 'F': fmt_str = optarg; break;
            case 0: print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    // 验证 tile 网格
    if (tile_cols * tile_w != width || tile_rows * tile_h != height) {
        std::cerr << "[Test] ERROR: Tile grid mismatch: "
                  << tile_cols << "x" << tile_w << "=" << (tile_cols*tile_w)
                  << " (expected " << width << "), "
                  << tile_rows << "x" << tile_h << "=" << (tile_rows*tile_h)
                  << " (expected " << height << ")\n";
        return 1;
    }

    mkdir(output_dir.c_str(), 0755);

    CamDmaMmap::Format cam_fmt = (fmt_str == "MJPEG" || fmt_str == "mjpeg")
                                 ? CamDmaMmap::Format::MJPEG
                                 : CamDmaMmap::Format::YUYV;

    std::cout << "[Test] ==================================================\n"
              << "[Test] Device : " << dev << "\n"
              << "[Test] Size   : " << width << "x" << height << "@" << fps << "\n"
              << "[Test] Mode   : DMA + " << (cam_fmt == CamDmaMmap::Format::MJPEG ? "MJPEG" : "YUYV") << "\n"
              << "[Test] Tiles  : " << tile_w << "x" << tile_h
              << "  grid=" << tile_cols << "x" << tile_rows << "\n"
              << "[Test] Output : " << output_dir << "\n"
              << "[Test] ==================================================\n";

    // 1. 初始化 Camera（DMA 模式）
    CamDmaMmap cam(CamDmaMmap::Mode::DMA);
    if (!cam.init(dev, width, height, fps, cam_fmt)) {
        std::cerr << "[Test] ERROR: Camera init failed\n";
        return 1;
    }
    if (!cam.start()) {
        std::cerr << "[Test] ERROR: Camera start failed\n";
        return 1;
    }

    // 2. 初始化 RGA Cropper（复用 dst buffer）
    RgaCropper cropper(width, height, tile_w, tile_h, tile_cols, tile_rows);
    if (!cropper.init()) {
        std::cerr << "[Test] ERROR: RGA cropper init failed\n";
        return 1;
    }

    int frame_count = 0;
    bool saved = false;
    auto t_start = std::chrono::steady_clock::now();
    double total_rga_ms = 0.0;

    while (g_running) {
        if (!cam.grab()) {
            continue;
        }

        auto t_rga0 = std::chrono::steady_clock::now();

        int src_fd = cam.src_fd();
        int src_fmt = cam.src_fmt();
        int src_w = cam.src_w();
        int src_h = cam.src_h();

        if (src_fd < 0) {
            std::cerr << "[Test] ERROR: src_fd invalid\n";
            cam.release();
            continue;
        }

        // RGA 硬件处理：色转 + 6 图 crop + 拼接
        bool ok = cropper.process(src_fd, src_fmt, src_w, src_h);

        auto t_rga1 = std::chrono::steady_clock::now();
        double rga_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_rga1 - t_rga0).count() / 1000.0;
        total_rga_ms += rga_ms;

        // 释放 V4L2 buffer（RGA 已完成，可以归还）
        cam.release();

        if (!ok) {
            std::cerr << "[Test] ERROR: RGA process failed\n";
            continue;
        }

        frame_count++;

        // 计算 FPS
        auto t_now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start).count() / 1000.0;
        double fps_actual = (elapsed_s > 0) ? (frame_count / elapsed_s) : 0;
        double avg_rga_ms = total_rga_ms / frame_count;

        std::cout << "[Test] Frame " << frame_count
                  << " | RGA=" << rga_ms << " ms"
                  << " (avg=" << avg_rga_ms << " ms)"
                  << " | FPS=" << fps_actual << "\n";

        // 达到 warmup 帧后，保存一次 6 张图
        if (frame_count == warmup && !saved) {
            auto t_save0 = std::chrono::steady_clock::now();
            std::string prefix = output_dir + "/tile";
            bool save_ok = RgaCropper::save_tiles(
                cropper.dst_fd(), cropper.dst_size(),
                tile_w, tile_h, cropper.tile_count(), prefix);
            auto t_save1 = std::chrono::steady_clock::now();
            double save_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_save1 - t_save0).count() / 1000.0;

            if (save_ok) {
                std::cout << "[Test] >>> Saved " << cropper.tile_count()
                          << " BMP tiles to " << output_dir
                          << " (save took " << save_ms << " ms)\n";
                saved = true;
            } else {
                std::cerr << "[Test] ERROR: Save tiles failed\n";
            }
        }

        // 保存后再跑 30 帧统计稳定 FPS，然后退出
        if (saved && frame_count >= warmup + 30) {
            break;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0;
    double avg_fps = (total_s > 0) ? (frame_count / total_s) : 0;
    double avg_rga_ms = (frame_count > 0) ? (total_rga_ms / frame_count) : 0;

    std::cout << "[Test] ==================================================\n"
              << "[Test] Total frames : " << frame_count << "\n"
              << "[Test] Total time   : " << total_s << " s\n"
              << "[Test] Average FPS  : " << avg_fps << "\n"
              << "[Test] Average RGA  : " << avg_rga_ms << " ms\n"
              << "[Test] Output files : " << output_dir << "/tile_{0.."
              << (cropper.tile_count()-1) << "}.bmp\n"
              << "[Test] ==================================================\n";

    return 0;
}
