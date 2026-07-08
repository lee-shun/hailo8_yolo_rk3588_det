/**
 * Test Case 1: 1280x1280 图 -> RGA 分块 4x640x640 -> Hailo Batch=4 推理
 *
 * 编译依赖:
 *   - OpenCV (读图、fallback 处理)
 *   - librga (Rockchip RGA 硬件加速)
 *   - HailoRT 4.23 (libhailort)
 *
 * 编译示例:
 *   g++ -std=c++17 test_tiling_batch.cpp -o test_tiling_batch \
 *       -I/usr/include/hailo -I/usr/include/rga \
 *       -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
 *       -lrga -lhailort -lpthread
 */
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "hailo/hailort.hpp"
#include "im2d_api.h"

using namespace hailort;

// 4 个 640x640 ROI 在原图 1280x1280 中的位置
struct ROI { int x, y, w, h; };
const ROI ROIS[4] = {
    {0,   0,   640, 640},   // 左上
    {640, 0,   640, 640},   // 右上
    {0,   640, 640, 640},   // 左下
    {640, 640, 640, 640}    // 右下
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_640.hef> <image.jpg>" << std::endl;
        return 1;
    }

    const std::string hef_path = argv[1];
    const std::string img_path = argv[2];

    // ========== 1. 读取原图并确保 1280x1280 ==========
    cv::Mat src = cv::imread(img_path);
    if (src.empty()) {
        std::cerr << "Failed to load image: " << img_path << std::endl;
        return 1;
    }
    if (src.cols != 1280 || src.rows != 1280) {
        cv::resize(src, src, cv::Size(1280, 1280));
    }

    // ========== 2. RGA 硬件分块 (4x640x640) ==========
    auto t_rga_start = std::chrono::high_resolution_clock::now();

    std::vector<cv::Mat> tiles_bgr(4);
    std::vector<cv::Mat> tiles_rgb(4);
    for (int i = 0; i < 4; i++) {
        tiles_bgr[i] = cv::Mat(640, 640, CV_8UC3);
        tiles_rgb[i] = cv::Mat(640, 640, CV_8UC3);
    }

    // RGA 源图 buffer
    rga_buffer_t src_buf = wrapbuffer_virtualaddr(
        src.data, src.cols, src.rows, RK_FORMAT_BGR_888);

    for (int i = 0; i < 4; i++) {
        // Step 1: RGA 裁剪 — 从 1280x1280 中裁出 640x640 ROI
        rga_buffer_t dst_bgr_buf = wrapbuffer_virtualaddr(
            tiles_bgr[i].data, 640, 640, RK_FORMAT_BGR_888);
        im_rect crop_rect = {ROIS[i].x, ROIS[i].y, ROIS[i].w, ROIS[i].h};

        IM_STATUS ret = imcrop(src_buf, dst_bgr_buf, crop_rect);
        if (ret != IM_STATUS_SUCCESS) {
            std::cerr << "RGA imcrop failed on tile " << i
                      << ", fallback to OpenCV" << std::endl;
            cv::Mat roi = src(cv::Rect(ROIS[i].x, ROIS[i].y, ROIS[i].w, ROIS[i].h));
            roi.copyTo(tiles_bgr[i]);
        }

        // Step 2: RGA 格式转换 — BGR -> RGB
        rga_buffer_t src_bgr_buf = wrapbuffer_virtualaddr(
            tiles_bgr[i].data, 640, 640, RK_FORMAT_BGR_888);
        rga_buffer_t dst_rgb_buf = wrapbuffer_virtualaddr(
            tiles_rgb[i].data, 640, 640, RK_FORMAT_RGB_888);

        ret = imcvtcolor(src_bgr_buf, dst_rgb_buf,
                         RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
        if (ret != IM_STATUS_SUCCESS) {
            cv::cvtColor(tiles_bgr[i], tiles_rgb[i], cv::COLOR_BGR2RGB);
        }
    }

    auto t_rga_end = std::chrono::high_resolution_clock::now();
    auto rga_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_rga_end - t_rga_start).count();

    // ========== 3. Hailo Batch=4 推理 ==========
    auto t_hailo_start = std::chrono::high_resolution_clock::now();

    // 3.1 加载 HEF
    auto hef = Hef::create(hef_path).expect("Failed to load HEF");

    // 3.2 创建 VDevice
    auto vdevice = VDevice::create().expect("Failed to create VDevice");

    // 3.3 配置参数: batch_size = 4
    auto configure_params = vdevice->create_configure_params(hef).value();
    for (auto& [name, params] : configure_params) {
        params.batch_size = 4;
        // 可选: params.power_mode = HAILO_POWER_MODE_ULTRA_PERFORMANCE;
    }

    // 3.4 配置网络组
    auto network_groups = vdevice->configure(hef, configure_params).value();
    auto network_group = network_groups.at(0);

    // 3.5 创建 input/output vstreams
    // 注: HailoRT 4.23 的 VStream API 请以实际头文件为准
    auto vstream_params = network_group->create_vstream_params();
    auto input_vstreams = hailort::VStreams::create_input_vstreams(
        *network_group, vstream_params).value();
    auto output_vstreams = hailort::VStreams::create_output_vstreams(
        *network_group, vstream_params).value();

    // 3.6 组装 batch input: 4 个 tile 连续排列
    // 假设模型输入: 640x640, RGB, uint8
    const size_t TILE_SIZE = 640 * 640 * 3;
    std::vector<uint8_t> batch_input(TILE_SIZE * 4);
    for (int i = 0; i < 4; i++) {
        std::memcpy(batch_input.data() + i * TILE_SIZE,
                    tiles_rgb[i].data, TILE_SIZE);
    }

    // 3.7 写入 input (batch=4 一次性写入)
    for (auto& input_vstream : input_vstreams) {
        input_vstream.write(MemoryView(batch_input.data(), batch_input.size()));
    }

    // 3.8 读取 output
    // output_frame_size 已包含 batch=4 维度
    const size_t output_frame_size = output_vstreams[0].get_frame_size();
    std::vector<uint8_t> output_buffer(output_frame_size);
    for (auto& output_vstream : output_vstreams) {
        output_vstream.read(MemoryView(output_buffer.data(), output_buffer.size()));
    }

    auto t_hailo_end = std::chrono::high_resolution_clock::now();
    auto hailo_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_hailo_end - t_hailo_start).count();

    // ========== 4. 后处理 (示例框架) ==========
    // output_buffer 包含 4 个 tile 的推理结果。
    // 解析时需将检测框坐标映射回原图:
    //   global_x = ROIS[tile_idx].x + local_x
    //   global_y = ROIS[tile_idx].y + local_y
    // 最后做跨 tile NMS 去重。

    std::cout << "\n========== Test Case 1: Tiling + Batch=4 ==========" << std::endl;
    std::cout << "RGA crop + convert time: " << rga_us << " us ("
              << rga_us / 1000.0 << " ms)" << std::endl;
    std::cout << "Hailo inference time:      " << hailo_us << " us ("
              << hailo_us / 1000.0 << " ms)" << std::endl;
    std::cout << "End-to-end latency:        " << (rga_us + hailo_us)
              << " us (" << (rga_us + hailo_us) / 1000.0 << " ms)" << std::endl;
    std::cout << "Note: Hailo run() returns after ALL 4 tiles complete" << std::endl;

    return 0;
}
