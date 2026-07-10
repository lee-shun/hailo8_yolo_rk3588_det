#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <cstdint>
#include <hailo/hailort.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fusion_roi_predictor.h"

class YoloDetector {
public:
  YoloDetector();
  ~YoloDetector();

  bool load(const std::string &hef_path, float conf, float iou);
  uint8_t *input_ptr() const;
  size_t input_bytes() const;
  int input_width() const;
  int input_height() const;

  std::vector<Detection> infer();
  std::vector<Detection> infer(const uint8_t *bgr_buf);

private:
  std::shared_ptr<uint8_t> aligned_alloc(size_t size);
  std::vector<Detection> parse_output(const float *raw, int w, int h) const;

  std::unique_ptr<hailort::VDevice> vdevice_;
  std::shared_ptr<hailort::InferModel> infer_model_;
  std::optional<hailort::ConfiguredInferModel> configured_model_;
  hailort::ConfiguredInferModel::Bindings bindings_;

  std::string input_name_;
  std::shared_ptr<uint8_t> input_buf_;
  size_t input_frame_size_ = 0;
  int nn_w_ = 0, nn_h_ = 0;
  float conf_thresh_ = 0.25f;

  std::vector<std::shared_ptr<uint8_t>> output_bufs_;
  std::vector<uint8_t *> output_ptrs_;
};

#endif
