#include "yolo_detector.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

YoloDetector::YoloDetector() = default;

YoloDetector::~YoloDetector() {
  if (configured_model_) {
    auto status = configured_model_->deactivate();
    (void)status;
  }
}

std::shared_ptr<uint8_t> YoloDetector::aligned_alloc(size_t size) {
  void *p = nullptr;
  if (posix_memalign(&p, 4096, size) != 0)
    throw std::bad_alloc();
  return std::shared_ptr<uint8_t>((uint8_t *)p, free);
}

bool YoloDetector::load(const std::string &hef_path, float conf, float iou,
                        uint16_t batch_size) {
  conf_thresh_ = conf;
  batch_size_ = batch_size;

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
  infer_model_->set_batch_size(batch_size_);

  try {
    for (const auto &n : infer_model_->get_output_names()) {
      auto out = infer_model_->output(n);
      if (out) {
        out->set_nms_score_threshold(conf);
        out->set_nms_iou_threshold(iou);
      }
    }
  } catch (...) {
    std::cerr << "[WARN] set_nms thresholds not supported, ignored\n";
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
    std::cerr << "[ERROR] Failed to get input stream\n";
    return false;
  }
  single_frame_size_ = in_stream->get_frame_size();

  std::cout << "[INFO] Model input: " << nn_w_ << "x" << nn_h_
            << ", single_frame=" << single_frame_size_
            << ", batch=" << batch_size_ << "\n";

  auto cfg = infer_model_->configure();
  if (!cfg) {
    std::cerr << "[ERROR] configure() failed: " << cfg.status() << "\n";
    return false;
  }
  configured_model_ =
      std::make_shared<hailort::ConfiguredInferModel>(cfg.release());

  auto act = configured_model_->activate();
  if (act != HAILO_SUCCESS) {
    std::cerr << "[ERROR] activate() failed: " << act << "\n";
    return false;
  }

  input_name_ = infer_model_->get_input_names()[0];
  std::cout << "[INFO] Input name: " << input_name_ << "\n";

  // 分配连续输入 buffer（所有 batch 帧）
  input_buf_ = aligned_alloc(single_frame_size_ * batch_size_);

  auto out_names = infer_model_->get_output_names();
  std::cout << "[INFO] Outputs: " << out_names.size() << "\n";

  // 创建 batch_size 个 Bindings，每个绑定独立的输入/输出偏移
  for (uint16_t b = 0; b < batch_size_; ++b) {
    auto b_expected = configured_model_->create_bindings();
    if (!b_expected) {
      std::cerr << "[ERROR] create_bindings failed: " << b_expected.status()
                << "\n";
      return false;
    }
    hailort::ConfiguredInferModel::Bindings binding = b_expected.release();

    // 输入：绑定到大 buffer 的对应偏移（单帧大小）
    auto bind_in = binding.input(input_name_);
    if (!bind_in) {
      std::cerr << "[ERROR] binding.input() failed\n";
      return false;
    }
    auto in_st = bind_in->set_buffer(hailort::MemoryView(
        input_buf_.get() + b * single_frame_size_, single_frame_size_));
    if (in_st != HAILO_SUCCESS) {
      std::cerr << "[ERROR] set_buffer(input) failed: " << in_st << "\n";
      return false;
    }

    // 输出：每个 binding 独立的输出 buffer
    for (auto &n : out_names) {
      auto out_info = infer_model_->output(n);
      if (!out_info)
        continue;
      size_t out_single = out_info->get_frame_size();

      auto buf = aligned_alloc(out_single);
      output_bufs_.push_back(buf);
      output_ptrs_.push_back(buf.get());

      auto bind_out = binding.output(n);
      if (!bind_out) {
        std::cerr << "[ERROR] binding.output() failed\n";
        return false;
      }
      auto out_st =
          bind_out->set_buffer(hailort::MemoryView(buf.get(), out_single));
      if (out_st != HAILO_SUCCESS) {
        std::cerr << "[ERROR] set_buffer(output) failed: " << out_st << "\n";
        return false;
      }
    }

    bindings_vec_.push_back(std::move(binding));
  }

  std::cout << "[INFO] Load success, " << bindings_vec_.size()
            << " bindings created\n";
  return true;
}

uint8_t *YoloDetector::input_ptr() const { return input_buf_.get(); }

size_t YoloDetector::input_bytes() const {
  return single_frame_size_ * batch_size_;
}

int YoloDetector::input_width() const { return nn_w_; }
int YoloDetector::input_height() const { return nn_h_; }
uint16_t YoloDetector::batch_size() const { return batch_size_; }

std::vector<Detection> YoloDetector::infer() {
  if (batch_size_ == 1) {
    // batch=1：同步 run
    auto status = configured_model_->run(bindings_vec_[0],
                                         std::chrono::milliseconds(5000));
    if (status != HAILO_SUCCESS) {
      std::cerr << "[ERROR] run() failed: " << status << "\n";
      throw std::runtime_error("Hailo inference failed");
    }
  } else {
    // batch>1：通过 run_async(vector<Bindings>) 一次提交所有帧
    auto job_expected = configured_model_->run_async(bindings_vec_);
    if (!job_expected) {
      std::cerr << "[ERROR] run_async failed: " << job_expected.status()
                << "\n";
      throw std::runtime_error("Hailo run_async failed");
    }
    auto job = job_expected.release();
    auto wait_status = job.wait(std::chrono::milliseconds(10000));
    if (wait_status != HAILO_SUCCESS) {
      std::cerr << "[ERROR] wait failed: " << wait_status << "\n";
      throw std::runtime_error("Hailo wait failed");
    }
  }
  // benchmark 只解析首帧输出，不关心结果正确性
  return parse_output((float *)output_ptrs_[0], nn_w_, nn_h_);
}

std::vector<std::vector<Detection>> YoloDetector::infer_all() {
  // ---- 执行推理（与 infer() 的 run 逻辑完全一致）----
  if (batch_size_ == 1) {
    auto status = configured_model_->run(bindings_vec_[0],
                                         std::chrono::milliseconds(5000));
    if (status != HAILO_SUCCESS) {
      std::cerr << "[ERROR] run() failed: " << status << "\n";
      throw std::runtime_error("Hailo inference failed");
    }
  } else {
    auto job_expected = configured_model_->run_async(bindings_vec_);
    if (!job_expected) {
      std::cerr << "[ERROR] run_async failed: " << job_expected.status() << "\n";
      throw std::runtime_error("Hailo run_async failed");
    }
    auto job = job_expected.release();
    auto wait_status = job.wait(std::chrono::milliseconds(10000));
    if (wait_status != HAILO_SUCCESS) {
      std::cerr << "[ERROR] wait failed: " << wait_status << "\n";
      throw std::runtime_error("Hailo wait failed");
    }
  }

  // ---- 解析每一帧输出（假设单输出节点）----
  std::vector<std::vector<Detection>> all_dets;
  for (uint16_t b = 0; b < batch_size_; ++b) {
    if (b < output_ptrs_.size()) {
      all_dets.push_back(parse_output((float *)output_ptrs_[b], nn_w_, nn_h_));
    } else {
      all_dets.push_back({});
    }
  }
  return all_dets;
}


std::vector<Detection> YoloDetector::infer(const uint8_t *bgr_buf) {
  std::memcpy(input_buf_.get(), bgr_buf, single_frame_size_ * batch_size_);
  return infer();
}

std::vector<Detection> YoloDetector::parse_output(const float *raw, int w,
                                                  int h) const {
  std::vector<Detection> dets;
  size_t idx = 0;
  for (size_t cls = 0; cls < 1; ++cls) {
    size_t n = (size_t)raw[idx++];
    if (n > 10000) {
      std::cerr << "[WARN] Bad detection count\n";
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
        dets.push_back({(float)x1, (float)y1, (float)(x2 - x1),
                        (float)(y2 - y1), conf, (int)cls});
      }
      idx += 5;
    }
  }
  return dets;
}
