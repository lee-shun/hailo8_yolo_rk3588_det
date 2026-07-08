#include "camera.h"
#include "rga_engine.h"
#include "fusion_roi_predictor.h"
#include "yolo_detector.h"
#include "display_thread.h"

#include <chrono>
#include <iostream>
#include <optional>

int main(int argc, char** argv) {
    std::string cam = "/dev/video0";
    std::string hef = "../yolov8n_640x640.hef";
    CameraFormat fmt = CameraFormat::YUYV;

    if (argc >= 2) cam = argv[1];
    if (argc >= 3) hef = argv[2];
    if (argc >= 4) {
        std::string f = argv[3];
        if (f == "mjpeg" || f == "MJPEG") fmt = CameraFormat::MJPEG;
    }

    // ==================== 1. 初始化各模块 ====================
    Camera camera;
    if (!camera.init(cam, 1920, 1080, 60, fmt)) {
        std::cerr << "Camera init failed\n";
        return 1;
    }

    RgaEngine rga;
    if (!rga.init(7, 640, 640, 320, 180)) {
        std::cerr << "RGA init failed\n";
        return 1;
    }

    YoloDetector detector;
    if (!detector.load(hef, 0.25f, 0.45f)) {
        std::cerr << "Detector load failed\n";
        return 1;
    }

    // 直接使用 FusionROIPredictor，无需 RoiTracker 包装
    FusionROIPredictor tracker;
    DisplayThread display;
    if (!display.start("ZeroCopy Pipeline")) {
        std::cerr << "Display thread init failed\n";
        return 1;
    }

    camera.start();
    std::cout << "[Run] Format: " << (fmt == CameraFormat::YUYV ? "YUYV" : "MJPEG")
              << " | ESC to quit, 'r' to reset\n";

    std::optional<Detection> last_det;
    int frame_count = 0;
    uint8_t* hailo_input = detector.input_ptr();

    // ==================== 2. 实时主循环 ====================
    while (display.running()) {
        auto t0 = std::chrono::steady_clock::now();

        // Step 1: 取帧
        if (!camera.grab()) continue;

        // Step 2: RGA 硬件处理
        rga.convert_gray(camera.src_ptr(), camera.src_stride(), camera.src_w(), camera.src_h(), camera.src_fmt());
        rga.convert_full_bgr(camera.src_ptr(), camera.src_stride(), camera.src_w(), camera.src_h(), camera.src_fmt());

        // Step 3: ROI 预测（直接使用 FusionROIPredictor）
        cv::Mat gray(180, 320, CV_8UC1, rga.gray_ptr(), rga.gray_stride());
        auto track_res = tracker.predict(last_det, gray);
        const auto& rois = track_res.rois;

        // Step 4: 串行 ROI 检测，命中即 break
        std::vector<Detection> dets;
        bool hit = false;
        int tried = 0;

        for (size_t i = 0; i < rois.size() && i < 7; ++i) {
            if (!rga.crop_to_ptr(camera.src_ptr(), camera.src_stride(), camera.src_w(), camera.src_h(), camera.src_fmt(),
                                 rois[i].x, rois[i].y, hailo_input, 640, 640))
                continue;

            dets = detector.infer();
            tried++;

            for (auto& d : dets) {
                d.x += rois[i].x;
                d.y += rois[i].y;
            }

            if (!dets.empty()) {
                hit = true;
                last_det = dets[0];
                break;
            }
        }

        if (!hit) {
            last_det.reset();
            dets.clear();
        }

        // Step 5: 推送到显示线程
        cv::Mat display_bgr(1080, 1920, CV_8UC3, rga.full_bgr_ptr(), 1920 * 3);
        display.push(display_bgr, track_res.pbas_res, rois, dets, hit, frame_count);

        // Step 6: 释放帧
        camera.release();

        frame_count++;
        if (frame_count % 30 == 0) {
            auto ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "[Stats] frame " << frame_count << "  " << ms
                      << " ms  rois=" << tried << "  hit=" << (hit ? "Y" : "N") << "\n";
        }
    }

    display.stop();
    camera.stop();
    std::cout << "[Exit] total: " << frame_count << "\n";
    return 0;
}
