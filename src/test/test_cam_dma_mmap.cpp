#include "cam_dma_mmap.h"
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

// ---------------------------------------------------------------------------
// 全局运行标志，用于 Ctrl+C 优雅退出
// ---------------------------------------------------------------------------
static volatile bool g_running = true;

static void signal_handler(int sig)
{
    std::cerr << "\n[Signal] Caught " << sig << ", stopping...\n";
    g_running = false;
}

// ---------------------------------------------------------------------------
// 命令行帮助
// ---------------------------------------------------------------------------
static void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -d <dev>      V4L2 device path (default: /dev/video0)\n"
              << "  -m <mode>     Access mode: dma | mmap (default: dma)\n"
              << "  -f <fmt>      Pixel format: mjpeg | yuyv (default: mjpeg)\n"
              << "  -W <width>    Image width (default: 1920)\n"
              << "  -H <height>   Image height (default: 1080)\n"
              << "  -r <fps>      Frame rate (default: 60)\n"
              << "  -n <count>    Number of frames to grab, 0=unlimited (default: 0)\n"
              << "  -h            Show this help\n"
              << "\nExamples:\n"
              << "  # DMA + MJPEG 1920x1080@60\n"
              << "  " << prog << "\n"
              << "  # MMAP + YUYV 1280x720@30, grab 100 frames\n"
              << "  " << prog << " -m mmap -f yuyv -W 1280 -H 720 -r 30 -n 100\n"
              << "  # DMA + YUYV 640x480@60\n"
              << "  " << prog << " -m dma -f yuyv -W 640 -H 480\n";
}

// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // 默认值
    std::string dev_path = "/dev/video0";
    std::string mode_str = "dma";
    std::string fmt_str  = "mjpeg";
    int width  = 1920;
    int height = 1080;
    int fps    = 60;
    int max_frames = 0;  // 0 = 无限

    int opt;
    while ((opt = getopt(argc, argv, "d:m:f:W:H:r:n:h")) != -1) {
        switch (opt) {
            case 'd': dev_path = optarg; break;
            case 'm': mode_str = optarg; break;
            case 'f': fmt_str  = optarg; break;
            case 'W': width  = std::atoi(optarg); break;
            case 'H': height = std::atoi(optarg); break;
            case 'r': fps    = std::atoi(optarg); break;
            case 'n': max_frames = std::atoi(optarg); break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    // 解析模式
    CamDmaMmap::Mode mode = CamDmaMmap::Mode::DMA;
    if (mode_str == "mmap" || mode_str == "MMAP") {
        mode = CamDmaMmap::Mode::MMAP;
    } else if (mode_str != "dma" && mode_str != "DMA") {
        std::cerr << "ERROR: Unknown mode '" << mode_str << "', use 'dma' or 'mmap'\n";
        return 1;
    }

    // 解析格式
    CamDmaMmap::Format fmt = CamDmaMmap::Format::MJPEG;
    if (fmt_str == "yuyv" || fmt_str == "YUYV") {
        fmt = CamDmaMmap::Format::YUYV;
    } else if (fmt_str != "mjpeg" && fmt_str != "MJPEG") {
        std::cerr << "ERROR: Unknown format '" << fmt_str << "', use 'mjpeg' or 'yuyv'\n";
        return 1;
    }

    // 注册信号
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "========================================\n"
              << "  CamDmaMmap Test\n"
              << "========================================\n"
              << "  Device : " << dev_path << "\n"
              << "  Mode   : " << (mode == CamDmaMmap::Mode::DMA ? "DMA" : "MMAP") << "\n"
              << "  Format : " << (fmt == CamDmaMmap::Format::MJPEG ? "MJPEG" : "YUYV") << "\n"
              << "  Size   : " << width << "x" << height << "\n"
              << "  FPS    : " << fps << "\n"
              << "  Frames : " << (max_frames > 0 ? std::to_string(max_frames) : "unlimited") << "\n"
              << "========================================\n";

    // 创建相机对象
    CamDmaMmap cam(mode);
    if (!cam.init(dev_path, width, height, fps, fmt)) {
        std::cerr << "ERROR: Camera init failed\n";
        return 1;
    }

    if (!cam.start()) {
        std::cerr << "ERROR: Camera start failed\n";
        return 1;
    }

    // 统计变量
    int frame_count = 0;
    int ok_count = 0;
    int err_count = 0;
    auto t_start = std::chrono::steady_clock::now();
    auto t_last  = t_start;

    while (g_running) {
        if (max_frames > 0 && frame_count >= max_frames)
            break;

        bool ok = cam.grab();
        frame_count++;

        if (!ok) {
            err_count++;
            // grab 内部已经做了 release/QBUF，无需额外处理
            continue;
        }

        ok_count++;

        // 根据模式读取数据（验证接口可用性）
        if (mode == CamDmaMmap::Mode::DMA) {
            int fd = cam.src_fd();
            if (fd < 0) {
                std::cerr << "ERROR: src_fd() returned " << fd << "\n";
            } else {
                // DMA 模式下 fd 有效，可在此送入 RGA/MPP Encoder
                (void)fd;  // 避免未使用警告
            }
        } else {
            uint8_t* ptr = cam.src_ptr();
            if (!ptr) {
                std::cerr << "ERROR: src_ptr() returned null\n";
            } else {
                // MMAP 模式下 ptr 有效，可在此做 CPU 处理
                (void)ptr;
            }
        }

        cam.release();

        // 每秒打印一次实时帧率
        auto t_now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t_now - t_last).count();
        if (elapsed_sec >= 1.0) {
            double instant_fps = (frame_count - (ok_count + err_count - ok_count)) / elapsed_sec; // 简化
            // 实际计算：这一秒内成功抓取的帧数
            static int last_ok = 0;
            int delta_ok = ok_count - last_ok;
            last_ok = ok_count;

            std::cout << "[Stats] " << std::fixed << std::setprecision(1)
                      << "time=" << std::chrono::duration<double>(t_now - t_start).count()
                      << "s fps=" << (delta_ok / elapsed_sec)
                      << " ok=" << ok_count
                      << " err=" << err_count
                      << " total=" << frame_count << "\n";
            t_last = t_now;
        }
    }

    // 停止并打印最终报告
    cam.stop();

    auto t_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================\n"
              << "  Final Report\n"
              << "========================================\n"
              << "  Total time : " << std::fixed << std::setprecision(2) << total_sec << " s\n"
              << "  Total grab : " << frame_count << "\n"
              << "  OK         : " << ok_count << "\n"
              << "  Error      : " << err_count << "\n"
              << "  Avg FPS    : " << (total_sec > 0 ? (ok_count / total_sec) : 0.0) << "\n"
              << "  Class stats: total=" << cam.total_frames()
              << " ok=" << cam.ok_frames()
              << " err=" << cam.err_frames() << "\n"
              << "========================================\n";

    return 0;
}
