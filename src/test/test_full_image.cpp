/**
 * Test Case 2: 1280x1280 图 -> RGA 预处理 -> Hailo 单张直接推理 (Batch=1)
 *
 * 假设模型输入尺寸为 1280x1280。
 * 如果模型输入是 640，请修改 RGA 步骤为 resize 到 640x640。
 *
 * 编译依赖同上
 */
#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "hailo/hailort.hpp"
#include "im2d_api.h"

using namespace hailort;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_1280.hef> <image.jpg>" << std::endl;
        return 1;
    }

    const std::string hef_path = argv[1];
    const std::string img_path = argv[2];

    // ========== 1. 读取原图 ==========
    cv::Mat src = cv::imread(img_path);
    if (src.empty()) {
        std::cerr << "Failed to load image: " << img_path << std::endl;
        return 1;
    }
    if (src.cols != 1280 || src.rows != 1280) {
        cv::resize(src, src, cv::Size(1280, 1280));
    }

    // ========== 2. RGA 预处理 (Format Convert: BGR -> RGB) ==========
    auto t_rga_start = std::chrono::high_resolution_clock::now();

    cv::Mat dst_rgb(1280, 1280, CV_8UC3);

    rga_buffer_t src_buf = wrapbuffer_virtualaddr(
        src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
    rga_buffer_t dst_buf = wrapbuffer_virtualaddr(
        dst_rgb.data, 1280, 1280, RK_FORMAT_RGB_888);

    // 如果模型输入是 1280x1280，只需格式转换
    // 如果模型输入是 640x640，请改用 imresize() + imcvtcolor()
    IM_STATUS ret = imcvtcolor(src_buf, dst_buf,
                               RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    if (ret != IM_STATUS_SUCCESS) {
        std::cerr << "RGA imcvtcolor failed, fallback to OpenCV" << std::endl;
        cv::cvtColor(src, dst_rgb, cv::COLOR_BGR2RGB);
    }

    auto t_rga_end = std::chrono::high_resolution_clock::now();
    auto rga_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_rga_end - t_rga_start).count();

    // ========== 3. Hailo 单张推理 (Batch=1) ==========
    auto t_hailo_start = std::chrono::high_resolution_clock::now();

    auto hef = Hef::create(hef_path).expect("Failed to load HEF");
    auto vdevice = VDevice::create().expect("Failed to create VDevice");

    // batch_size = 1 (默认)
    auto configure_params = vdevice->create_configure_params(hef).value();
    for (auto& [name, params] : configure_params) {
        params.batch_size = 1;
    }

    auto network_groups = vdevice->configure(hef, configure_params).value();
    auto network_group = network_groups.at(0);

    auto vstream_params = network_group->create_vstream_params();
    auto input_vstreams = hailort::VStreams::create_input_vstreams(
        *network_group, vstream_params).value();
    auto output_vstreams = hailort::VStreams::create_output_vstreams(
        *network_group, vstream_params).value();

    // 写入单张图
    const size_t INPUT_SIZE = 1280 * 1280 * 3;
    for (auto& input_vstream : input_vstreams) {
        input_vstream.write(MemoryView(dst_rgb.data, INPUT_SIZE));
    }

    // 读取结果
    const size_t output_frame_size = output_vstreams[0].get_frame_size();
    std::vector<uint8_t> output_buffer(output_frame_size);
    for (auto& output_vstream : output_vstreams) {
        output_vstream.read(MemoryView(output_buffer.data(), output_buffer.size()));
    }

    auto t_hailo_end = std::chrono::high_resolution_clock::now();
    auto hailo_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_hailo_end - t_hailo_start).count();

    // ========== 4. 后处理 ==========
    // 直接解析 output_buffer，检测框坐标即原图坐标，无需映射。

    std::cout << "\n========== Test Case 2: Full Image Direct ==========" << std::endl;
    std::cout << "RGA convert time:          " << rga_us << " us ("
              << rga_us / 1000.0 << " ms)" << std::endl;
    std::cout << "Hailo inference time:      " << hailo_us << " us ("
              << hailo_us / 1000.0 << " ms)" << std::endl;
    std::cout << "End-to-end latency:        " << (rga_us + hailo_us)
              << " us (" << (rga_us + hailo_us) / 1000.0 << " ms)" << std::endl;
    std::cout << "Note: Single frame, no batch waiting" << std::endl;

    return 0;
}
