#include "yolo_detector.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
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
                        uint16_t batch_size,
                        const std::vector<int> &input_fds) {
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

  // ===== 性能优化：最高功率模式 + 关闭测量开销 =====
  infer_model_->set_power_mode(HAILO_POWER_MODE_ULTRA_PERFORMANCE);
  infer_model_->set_hw_latency_measurement_flags(HAILO_LATENCY_NONE);
  // infer_model_->set_hw_latency_measurement_flags(HAILO_LATENCY_MEASURE);
  // ====================================================

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

  auto out_names = infer_model_->get_output_names();
  std::cout << "[INFO] Outputs: " << out_names.size() << "\n";

  // 判断是否真零拷贝模式
  bool zero_copy = (!input_fds.empty()) &&
                   (static_cast<int>(input_fds.size()) == batch_size_);
  std::cout << "[INFO] YOLO: " << (zero_copy ? "Using Zero Copy!" : "CPU mode")
            << std::endl;

  if (!zero_copy) {
    // 传统模式：分配 CPU 输入 buffer
    input_buf_ = aligned_alloc(single_frame_size_ * batch_size_);
  }

  // 创建 batch_size 个 Bindings
  for (uint16_t b = 0; b < batch_size_; ++b) {
    auto b_expected = configured_model_->create_bindings();
    if (!b_expected) {
      std::cerr << "[ERROR] create_bindings failed: " << b_expected.status()
                << "\n";
      return false;
    }
    hailort::ConfiguredInferModel::Bindings binding = b_expected.release();

    // ---- 输入绑定 ----
    auto bind_in = binding.input(input_name_);
    if (!bind_in) {
      std::cerr << "[ERROR] binding.input() failed\n";
      return false;
    }

    if (zero_copy) {
      // 真零拷贝：直接绑定 dmabuf fd
      hailo_dma_buffer_t dma_buf{input_fds[b], single_frame_size_};
      auto in_st = bind_in->set_dma_buffer(dma_buf);
      if (in_st != HAILO_SUCCESS) {
        std::cerr << "[ERROR] set_dma_buffer(input, fd=" << input_fds[b]
                  << ") failed: " << in_st << "\n";
        return false;
      }
      // 提前注册 DMA mapping，避免运行时开销，并提升 async 性能
      auto map_st = vdevice_->dma_map_dmabuf(input_fds[b], single_frame_size_,
                                             HAILO_DMA_BUFFER_DIRECTION_H2D);
      if (map_st != HAILO_SUCCESS) {
        std::cerr << "[WARN] dma_map_dmabuf(input fd=" << input_fds[b]
                  << ") failed: " << map_st << "\n";
      }
    } else {
      // 传统模式：绑定 CPU 指针
      auto in_st = bind_in->set_buffer(hailort::MemoryView(
          input_buf_.get() + b * single_frame_size_, single_frame_size_));
      if (in_st != HAILO_SUCCESS) {
        std::cerr << "[ERROR] set_buffer(input) failed: " << in_st << "\n";
        return false;
      }
    }

    // ---- 输出绑定（不变，仍用 CPU buffer）----
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
            << " bindings created" << (zero_copy ? " (ZERO-COPY mode)" : "")
            << "\n";

  // 在 load() 中打印输出格式
  for (const auto &n : infer_model_->get_output_names()) {
    auto out = infer_model_->output(n);
    std::cout << "Output: " << n << " format_order=" << out->format().order
              << " is_nms=" << out->is_nms() << "\n";
  }
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
    auto status = configured_model_->run(bindings_vec_[0],
                                         std::chrono::milliseconds(5000));
    if (status != HAILO_SUCCESS) {
      std::cerr << "[ERROR] run() failed: " << status << "\n";
      throw std::runtime_error("Hailo inference failed");
    }
  } else {
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
  return parse_output((float *)output_ptrs_[0], nn_w_, nn_h_);
}

std::vector<std::vector<Detection>> YoloDetector::infer_all() {
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

  // // 读取硬件延迟（与 hailortcli 的 HW Latency 同口径）
  // auto hw_lat = configured_model_->get_hw_latency_measurement();
  // if (hw_lat) {
  //     double ms = std::chrono::duration<double, std::milli>(hw_lat->avg_hw_latency).count();
  //     std::cout << "[INFO] HW Latency: " << std::fixed << std::setprecision(6) << ms << " ms\n";
  // }

  // 解析每一帧输出
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
