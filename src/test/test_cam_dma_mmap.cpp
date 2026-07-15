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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

static volatile bool g_running = true;

static void signal_handler(int sig)
{
    std::cerr << "\n[Signal] Caught " << sig << ", stopping...\n";
    g_running = false;
}

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
              << "  # DMA + MJPEG 1920x1080@60, save 500th frame as .nv12\n"
              << "  " << prog << " -n 1000\n"
              << "  # MMAP + YUYV 1280x720@30, save 50th frame as .yuyv\n"
              << "  " << prog << " -m mmap -f yuyv -W 1280 -H 720 -r 30 -n 100\n"
              << "\nView saved frame:\n"
              << "  ffplay -f rawvideo -pixel_format nv12    -video_size 1920x1080 dma_jpeg_500_1000_frame.nv12\n"
              << "  ffplay -f rawvideo -pixel_format yuyv422 -video_size 1280x720  mmap_yuv_50_100_frame.yuyv\n";
}

int main(int argc, char** argv)
{
    std::string dev_path = "/dev/video0";
    std::string mode_str = "dma";
    std::string fmt_str  = "mjpeg";
    int width  = 1920;
    int height = 1080;
    int fps    = 60;
    int max_frames = 0;

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

    CamDmaMmap::Mode mode = CamDmaMmap::Mode::DMA;
    if (mode_str == "mmap" || mode_str == "MMAP") {
        mode = CamDmaMmap::Mode::MMAP;
    } else if (mode_str != "dma" && mode_str != "DMA") {
        std::cerr << "ERROR: Unknown mode '" << mode_str << "'\n";
        return 1;
    }

    CamDmaMmap::Format fmt = CamDmaMmap::Format::MJPEG;
    if (fmt_str == "yuyv" || fmt_str == "YUYV") {
        fmt = CamDmaMmap::Format::YUYV;
    } else if (fmt_str != "mjpeg" && fmt_str != "MJPEG") {
        std::cerr << "ERROR: Unknown format '" << fmt_str << "'\n";
        return 1;
    }

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
              << "  Frames : " << (max_frames > 0 ? std::to_string(max_frames) : "unlimited") << "\n";
    if (max_frames > 0) {
        std::cout << "  Save   : Will save frame #" << (max_frames / 2)
                  << " (n/2 of " << max_frames << ")\n";
    }
    std::cout << "========================================\n";

    CamDmaMmap cam(mode);
    if (!cam.init(dev_path, width, height, fps, fmt)) {
        std::cerr << "ERROR: Camera init failed\n";
        return 1;
    }

    if (!cam.start()) {
        std::cerr << "ERROR: Camera start failed\n";
        return 1;
    }

    int frame_count = 0;
    int ok_count = 0;
    int err_count = 0;
    bool saved_frame = false;
    auto t_start = std::chrono::steady_clock::now();
    auto t_last  = t_start;

    auto do_save_frame = [&](int frame_num) {
        std::string mode_name = (mode == CamDmaMmap::Mode::DMA) ? "dma" : "mmap";
        std::string fmt_name  = (fmt == CamDmaMmap::Format::MJPEG) ? "jpeg" : "yuv";
        std::string ext       = (fmt == CamDmaMmap::Format::MJPEG) ? "nv12" : "yuyv";
        std::string filename = mode_name + "_" + fmt_name + "_" 
                             + std::to_string(frame_num) + "_" 
                             + std::to_string(max_frames) + "_frame." + ext;

        int w = cam.src_w();
        int h = cam.src_h();
        int stride = cam.src_stride();
        int ver_stride = cam.src_ver_stride();

        uint8_t* ptr = nullptr;
        bool need_munmap = false;
        size_t mmap_size = 0;

        // -----------------------------------------------------------------
        // 核心修复：MJPEG 统一使用 src_ptr()，不再重新 mmap fd
        // MPP 的 ION buffer 在 mpp_buffer_get() 时已 mmap，且 MPP 内部已处理 cache sync
        // 重新 mmap 同一个 fd 会导致二次映射 cache 不一致，出现花屏/错位
        // -----------------------------------------------------------------
        if (fmt == CamDmaMmap::Format::MJPEG) {
            ptr = cam.src_ptr();
            if (!ptr) {
                std::cerr << "\n[Save] ERROR: src_ptr() returned null for MJPEG\n";
                return;
            }
            // DMA/MMAP 模式下，src_ptr() 均返回 MPP 内部管理的 CPU 映射地址
            std::cout << "MJPEG save using src_ptr=" << static_cast<void*>(ptr)
                    << " stride=" << stride << " ver_stride=" << ver_stride<<std::endl;
        } else {
            // YUYV: MMAP 直接取 ptr，DMA 需要 mmap fd
            if (mode == CamDmaMmap::Mode::MMAP) {
                ptr = cam.src_ptr();
                if (!ptr) {
                    std::cerr << "\n[Save] ERROR: src_ptr() returned null for YUYV MMAP\n";
                    return;
                }
            } else {
                // YUYV + DMA: V4L2 dma-buf 没有 CPU 映射，必须 mmap
                int fd = cam.src_fd();
                if (fd < 0) {
                    std::cerr << "\n[Save] ERROR: src_fd() invalid for YUYV DMA\n";
                    return;
                }
                mmap_size = (size_t)stride * h; // YUYV: stride * height
                ptr = (uint8_t*)mmap(nullptr, mmap_size, PROT_READ, MAP_SHARED, fd, 0);
                if (ptr == MAP_FAILED) {
                    std::cerr << "\n[Save] ERROR: mmap failed: " << strerror(errno) << "\n";
                    return;
                }
                need_munmap = true;
            }
        }

        FILE* fp = fopen(filename.c_str(), "wb");
        if (!fp) {
            std::cerr << "\n[Save] ERROR: Failed to open " << filename << "\n";
            if (need_munmap) munmap(ptr, mmap_size);
            return;
        }

        if (fmt == CamDmaMmap::Format::MJPEG) {
            // NV12: 逐行拷贝有效区域，去除 stride padding
            // Y 平面
            for (int row = 0; row < h; ++row) {
                fwrite(ptr + (size_t)row * stride, 1, w, fp);
            }
            // UV 平面：从 stride * ver_stride 偏移开始（MPP 物理布局）
            uint8_t* uv_ptr = ptr + (size_t)stride * ver_stride;
            for (int row = 0; row < h / 2; ++row) {
                fwrite(uv_ptr + (size_t)row * stride, 1, w, fp);
            }
        } else {
            // YUYV: h 行，每行 w*2 字节
            for (int row = 0; row < h; ++row) {
                fwrite(ptr + (size_t)row * stride, 1, w * 2, fp);
            }
        }

        fclose(fp);
        if (need_munmap) munmap(ptr, mmap_size);

        std::cout << "\n========================================\n"
                  << "  FRAME SAVED\n"
                  << "========================================\n"
                  << "  File:       " << filename << "\n"
                  << "  Resolution: " << w << "x" << h << "\n"
                  << "  Stride:     " << stride << "x" << ver_stride << "\n";
        if (fmt == CamDmaMmap::Format::MJPEG) {
            std::cout << "  Format:     NV12 (YUV420SP, no padding)\n"
                      << "  View cmd:   ffplay -f rawvideo -pixel_format nv12 -video_size "
                      << w << "x" << h << " " << filename << "\n";
        } else {
            std::cout << "  Format:     YUYV422 (packed, no padding)\n"
                      << "  View cmd:   ffplay -f rawvideo -pixel_format yuyv422 -video_size "
                      << w << "x" << h << " " << filename << "\n";
        }
        std::cout << "========================================\n";
    };

    while (g_running) {
        if (max_frames > 0 && frame_count >= max_frames)
            break;

        bool ok = cam.grab();
        frame_count++;

        if (!ok) {
            err_count++;
            continue;
        }

        ok_count++;

        if (max_frames > 0 && !saved_frame && ok_count == max_frames / 2) {
            do_save_frame(ok_count);
            saved_frame = true;
        }

        // 接口验证
        if (mode == CamDmaMmap::Mode::DMA) {
            (void)cam.src_fd();
        } else {
            (void)cam.src_ptr();
        }

        cam.release();

        auto t_now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t_now - t_last).count();
        if (elapsed_sec >= 1.0) {
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
              << " err=" << cam.err_frames() << "\n";
    if (max_frames > 0 && !saved_frame) {
        std::cout << "  Save       : FAILED (not enough OK frames)\n";
    } else if (saved_frame) {
        std::cout << "  Save       : OK (frame #" << max_frames / 2 << ")\n";
    }
    std::cout << "========================================\n";

    return 0;
}
