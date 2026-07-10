#include "yolo_detector.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

YoloDetector::YoloDetector() = default;

YoloDetector::~YoloDetector() {
  if (configured_model_) {
    auto status = configured_model_->deactivate();
    (void)status; // 清理阶段忽略错误
  }
}

std::shared_ptr<uint8_t> YoloDetector::aligned_alloc(size_t size) {
  void *p = nullptr;
  if (posix_memalign(&p, 4096, size) != 0)
    throw std::bad_alloc();
  return std::shared_ptr<uint8_t>((uint8_t *)p, free);
}

bool YoloDetector::load(const std::string &hef_path, float conf, float iou) {
  conf_thresh_ = conf;

  hailo_vdevice_params_t params{};
  hailo_init_vdevice_params(&params);
  params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_NONE;

  auto vdev = hailort::VDevice::create(params);
  if (!vdev) {
    std::cerr << "[ERROR] VDevice::create failed: " << vdev.status() << "\n";
    return false;
  }
  vdevice_ = vdev.release();

  auto model = vdevice_->create_infer_model(hef_path);
  if (!model) {
    std::cerr << "[ERROR] create_infer_model failed: " << model.status()
              << "\n";
    return false;
  }
  infer_model_ = model.release();
  infer_model_->set_batch_size(1);

  try {
    auto out_names = infer_model_->get_output_names();
    for (const auto &n : out_names) {
      auto out = infer_model_->output(n);
      if (out) {
        out->set_nms_score_threshold(conf);
        out->set_nms_iou_threshold(iou);
      }
    }
  } catch (...) {
    std::cerr
        << "[WARN] set_nms thresholds not supported by this HEF, ignored\n";
  }

  const auto &inputs = infer_model_->inputs();
  if (inputs.empty()) {
    std::cerr << "[ERROR] Model has no inputs\n";
    return false;
  }
  nn_w_ = inputs[0].shape().width;
  nn_h_ = inputs[0].shape().height;

  auto in_stream = infer_model_->input();
  if (!in_stream) {
    std::cerr << "[ERROR] Failed to get input stream info\n";
    return false;
  }
  input_frame_size_ = in_stream->get_frame_size();

  std::cout << "[INFO] Model input: " << nn_w_ << "x" << nn_h_
            << ", frame_size=" << input_frame_size_ << "\n";

  auto cfg = infer_model_->configure();
  if (!cfg) {
    std::cerr << "[ERROR] infer_model_->configure() failed: " << cfg.status()
              << "\n";
    return false;
  }
  configured_model_.emplace(cfg.release());

  // 关键修复：显式激活推理管道（解决 HAILO_STREAM_NOT_ACTIVATED）
  auto act_status = configured_model_->activate();
  if (act_status != HAILO_SUCCESS) {
    std::cerr << "[ERROR] configured_model_->activate() failed: " << act_status
              << "\n";
    return false;
  }

  input_name_ = infer_model_->get_input_names()[0];
  std::cout << "[INFO] Input name: " << input_name_ << "\n";

  input_buf_ = aligned_alloc(input_frame_size_);

  auto b = configured_model_->create_bindings();
  if (!b) {
    std::cerr << "[ERROR] create_bindings() failed: " << b.status() << "\n";
    return false;
  }
  bindings_ = b.release();

  auto bind_in = bindings_.input(input_name_);
  if (!bind_in) {
    std::cerr << "[ERROR] bindings_.input() failed: " << bind_in.status()
              << "\n";
    return false;
  }
  auto in_status = bind_in->set_buffer(
      hailort::MemoryView(input_buf_.get(), input_frame_size_));
  if (in_status != HAILO_SUCCESS) {
    std::cerr << "[ERROR] set_buffer(input) failed: " << in_status << "\n";
    return false;
  }

  auto out_names = infer_model_->get_output_names();
  std::cout << "[INFO] Outputs: " << out_names.size() << "\n";
  for (auto &n : out_names) {
    auto out_info = infer_model_->output(n);
    if (!out_info)
      continue;
    size_t sz = out_info->get_frame_size();
    std::cout << "  - " << n << " size=" << sz << "\n";
    auto buf = aligned_alloc(sz);
    output_bufs_.push_back(buf);
    output_ptrs_.push_back(buf.get());

    auto bind_out = bindings_.output(n);
    if (!bind_out) {
      std::cerr << "[ERROR] bindings_.output(" << n << ") failed\n";
      return false;
    }
    auto out_status = bind_out->set_buffer(hailort::MemoryView(buf.get(), sz));
    if (out_status != HAILO_SUCCESS) {
      std::cerr << "[ERROR] set_buffer(output " << n
                << ") failed: " << out_status << "\n";
      return false;
    }
  }

  std::cout << "[INFO] Load success, ready for inference\n";
  return true;
}

uint8_t *YoloDetector::input_ptr() const { return input_buf_.get(); }

size_t YoloDetector::input_bytes() const { return input_frame_size_; }

int YoloDetector::input_width() const { return nn_w_; }
int YoloDetector::input_height() const { return nn_h_; }

std::vector<Detection> YoloDetector::infer() {
  auto status =
      configured_model_->run(bindings_, std::chrono::milliseconds(5000));
  if (status != HAILO_SUCCESS) {
    std::cerr << "[ERROR] run() failed, status=" << status << "\n";
    throw std::runtime_error("Hailo inference failed");
  }
  return parse_output((float *)output_ptrs_[0], nn_w_, nn_h_);
}

std::vector<Detection> YoloDetector::infer(const uint8_t *bgr_buf) {
  std::memcpy(input_buf_.get(), bgr_buf, input_frame_size_);
  return infer();
}

std::vector<Detection> YoloDetector::parse_output(const float *raw, int w,
                                                  int h) const {
  std::vector<Detection> dets;
  size_t idx = 0;
  size_t num_classes = 1;

  for (size_t cls = 0; cls < num_classes; ++cls) {
    size_t n = (size_t)raw[idx++];
    if (n > 1000) {
      std::cerr << "[WARN] Invalid detection count: " << n << ", skip\n";
      break;
    }
    for (size_t i = 0; i < n; ++i) {
      float ymin = raw[idx], xmin = raw[idx + 1];
      float ymax = raw[idx + 2], xmax = raw[idx + 3];
      float conf = raw[idx + 4];

      int x1 = (int)(xmin * w), y1 = (int)(ymin * h);
      int x2 = (int)(xmax * w), y2 = (int)(ymax * h);
      x1 = std::max(0, std::min(x1, w - 1));
      y1 = std::max(0, std::min(y1, h - 1));
      x2 = std::max(0, std::min(x2, w - 1));
      y2 = std::max(0, std::min(y2, h - 1));

      if (conf > conf_thresh_ && x2 > x1 && y2 > y1) {
        Detection d{(float)x1,        (float)y1, (float)(x2 - x1),
                    (float)(y2 - y1), conf,      (int)cls};
        dets.push_back(d);
      }
      idx += 5;
    }
  }
  return dets;
}
