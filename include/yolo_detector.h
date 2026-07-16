#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <hailo/hailort.hpp>
#include <memory>
#include <string>
#include <vector>

struct Detection {
  float x, y, w, h, conf;
  int cls;
};

class YoloDetector {
public:
  YoloDetector();
  ~YoloDetector();

  // 新增 input_fds 参数，默认空表示走传统 CPU buffer 模式（兼容旧代码）
  bool load(const std::string &hef_path, float conf, float iou,
            uint16_t batch_size, const std::vector<int> &input_fds = {});

  uint8_t *input_ptr() const;
  size_t input_bytes() const;
  int input_width() const;
  int input_height() const;
  uint16_t batch_size() const;

  std::vector<Detection> infer();
  std::vector<std::vector<Detection>> infer_all();
  std::vector<Detection> infer(const uint8_t *bgr_buf);

private:
  std::shared_ptr<uint8_t> aligned_alloc(size_t size);
  std::vector<Detection> parse_output(const float *raw, int w, int h) const;

  float conf_thresh_ = 0.25f;
  uint16_t batch_size_ = 1;
  int nn_w_ = 0, nn_h_ = 0;
  size_t single_frame_size_ = 0;
  std::string input_name_;

  std::unique_ptr<hailort::VDevice> vdevice_;
  std::shared_ptr<hailort::InferModel> infer_model_;
  std::shared_ptr<hailort::ConfiguredInferModel> configured_model_;
  std::vector<hailort::ConfiguredInferModel::Bindings> bindings_vec_;

  std::shared_ptr<uint8_t> input_buf_; // 零拷贝模式下为 nullptr
  std::vector<std::shared_ptr<uint8_t>> output_bufs_;
  std::vector<uint8_t *> output_ptrs_;
  std::vector<int> output_fds_;  // 记录 dmabuf fd，析构时 munmap/close
};

#endif
