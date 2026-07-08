#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <hailo/hailort.h>
#include <hailo/hailort_common.hpp>
#include <hailo/infer_model.hpp>
#include <hailo/vdevice.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fusion_roi_predictor.h"

// Hailo-8 同步推理器
// 构造函数预分配页对齐输入/输出缓冲区，终身复用
// 暴露 input_ptr() 供 RGA 直接 DMA 写入，实现零拷贝
class YoloDetector {
public:
    YoloDetector();
    ~YoloDetector();

    bool load(const std::string& hef_path, float conf = 0.25f, float iou = 0.45f);

    uint8_t* input_ptr() const;
    size_t input_bytes() const;

    std::vector<Detection> infer();
    std::vector<Detection> infer(const uint8_t* bgr_buf);

    int input_width() const;
    int input_height() const;

private:
    std::shared_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::InferModel> infer_model_;
    std::shared_ptr<hailort::ConfiguredInferModel> configured_model_;
    std::unique_ptr<hailort::ConfiguredInferModel::Bindings> bindings_;

    std::string input_name_;
    size_t input_frame_size_ = 0;
    int nn_w_ = 0, nn_h_ = 0;
    float conf_thresh_ = 0.25f;

    std::shared_ptr<uint8_t> input_buf_;
    std::vector<std::shared_ptr<uint8_t>> output_bufs_;
    std::vector<uint8_t*> output_ptrs_;

    static std::shared_ptr<uint8_t> aligned_alloc(size_t size);
    std::vector<Detection> parse_output(const float* raw, int w, int h) const;
};

#endif
