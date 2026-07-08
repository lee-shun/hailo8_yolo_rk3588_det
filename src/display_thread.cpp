#include "display_thread.h"
#include <chrono>
#include <iostream>

DisplayThread::DisplayThread() : running_(false), queue_(5) {}
DisplayThread::~DisplayThread() { stop(); }

bool DisplayThread::start(const std::string& win_name) {
    if (!visualizer_.init()) return false;
    win_name_ = win_name;
    running_ = true;
    thread_ = std::thread(&DisplayThread::loop, this);
    return true;
}

void DisplayThread::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    cv::destroyWindow(win_name_);
}

bool DisplayThread::running() const { return running_.load(); }

bool DisplayThread::push(cv::Mat frame_bgr, const PBASResult& pbas,
                         const std::vector<ROI>& rois, const std::vector<Detection>& dets,
                         bool hit, int frame_count) {
    DisplayPacket pkt;
    pkt.frame_bgr = frame_bgr.clone();
    pkt.pbas = pbas;
    pkt.pbas.fg_mask = pbas.fg_mask.clone();
    pkt.rois = rois;
    pkt.dets = dets;
    pkt.hit = hit;
    pkt.frame_count = frame_count;
    return queue_.try_enqueue(std::move(pkt));
}

void DisplayThread::loop() {
    while (running_.load()) {
        DisplayPacket pkt;
        if (queue_.try_dequeue(pkt)) {
            cv::Mat vis = visualizer_.draw(pkt.frame_bgr, pkt.pbas, pkt.rois, pkt.dets, pkt.hit);

            cv::Mat display;
            double scale = std::min(1.0, 1080.0 / vis.rows);
            if (scale < 1.0)
                cv::resize(vis, display, cv::Size(), scale, scale);
            else
                display = vis;

            cv::imshow(win_name_, display);

            char key = (char)cv::waitKey(1);
            if (key == 27) running_ = false;
            if (key == 'r' || key == 'R') visualizer_.reset_trail();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}
