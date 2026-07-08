#include "yolo_detector.h"

YoloDetector::YoloDetector() = default;
YoloDetector::~YoloDetector() = default;

std::shared_ptr<uint8_t> YoloDetector::aligned_alloc(size_t size) {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, size) != 0) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>((uint8_t*)p, free);
}

bool YoloDetector::load(const std::string& hef_path, float conf, float iou) {
    conf_thresh_ = conf;

    hailo_vdevice_params_t params{};
    hailo_init_vdevice_params(&params);
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_NONE;

    auto vdev = hailort::VDevice::create(params);
    if (!vdev) return false;
    vdevice_ = vdev.release();

    auto model = vdevice_->create_infer_model(hef_path);
    if (!model) return false;
    infer_model_ = model.release();
    infer_model_->set_batch_size(1);

    try {
        infer_model_->output()->set_nms_score_threshold(conf);
        infer_model_->output()->set_nms_iou_threshold(iou);
    } catch (...) {}

    nn_w_ = infer_model_->inputs()[0].shape().width;
    nn_h_ = infer_model_->inputs()[0].shape().height;
    input_frame_size_ = infer_model_->input()->get_frame_size();

    auto cfg = infer_model_->configure();
    if (!cfg) return false;
    configured_model_ = std::make_shared<hailort::ConfiguredInferModel>(cfg.release());

    input_name_ = infer_model_->get_input_names()[0];

    input_buf_ = aligned_alloc(input_frame_size_);
    auto b = configured_model_->create_bindings();
    if (!b) return false;
    bindings_ = std::make_unique<hailort::ConfiguredInferModel::Bindings>(b.release());

    bindings_->input(input_name_)->set_buffer(
        hailort::MemoryView(input_buf_.get(), input_frame_size_));

    for (auto& n : infer_model_->get_output_names()) {
        size_t sz = infer_model_->output(n)->get_frame_size();
        auto buf = aligned_alloc(sz);
        output_bufs_.push_back(buf);
        output_ptrs_.push_back(buf.get());
        bindings_->output(n)->set_buffer(hailort::MemoryView(buf.get(), sz));
    }
    return true;
}

uint8_t* YoloDetector::input_ptr() const {
    return input_buf_.get();
}

size_t YoloDetector::input_bytes() const {
    return input_frame_size_;
}

std::vector<Detection> YoloDetector::infer() {
    auto status = configured_model_->run(*bindings_, std::chrono::milliseconds(1000));
    if (status != HAILO_SUCCESS)
        throw std::runtime_error("Hailo inference failed");
    return parse_output((float*)output_ptrs_[0], nn_w_, nn_h_);
}

std::vector<Detection> YoloDetector::infer(const uint8_t* bgr_buf) {
    std::memcpy(input_buf_.get(), bgr_buf, input_frame_size_);
    return infer();
}

int YoloDetector::input_width() const { return nn_w_; }
int YoloDetector::input_height() const { return nn_h_; }

std::vector<Detection> YoloDetector::parse_output(const float* raw, int w, int h) const {
    std::vector<Detection> dets;
    size_t idx = 0;
    size_t num_classes = 1;

    for (size_t cls = 0; cls < num_classes; ++cls) {
        size_t n = (size_t)raw[idx++];
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
                Detection d{(float)x1, (float)y1, (float)(x2 - x1), (float)(y2 - y1), conf, (int)cls};
                dets.push_back(d);
            }
            idx += 5;
        }
    }
    return dets;
}
