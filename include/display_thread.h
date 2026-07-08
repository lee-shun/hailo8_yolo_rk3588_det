#ifndef DISPLAY_THREAD_H
#define DISPLAY_THREAD_H

// 标准库优先
#include <atomic>
#include <string>
#include <thread>
#include <vector>

// OpenCV 其次
#include <opencv2/opencv.hpp>

#include "fusion_roi_predictor.h"
#include "rga_visualizer.h"
#include "utility_tool/readerwriterqueue.h"

struct DisplayPacket {
  cv::Mat frame_bgr;
  PBASResult pbas;
  std::vector<ROI> rois;
  std::vector<Detection> dets;
  bool hit = false;
  int frame_count = 0;
};

class DisplayThread {
public:
  DisplayThread();
  ~DisplayThread();

  bool start(const std::string &win_name);
  void stop();
  bool running() const;

  bool push(cv::Mat frame_bgr, const PBASResult &pbas,
            const std::vector<ROI> &rois, const std::vector<Detection> &dets,
            bool hit, int frame_count);

private:
  std::atomic<bool> running_;
  std::thread thread_;
  std::string win_name_;
  moodycamel::ReaderWriterQueue<DisplayPacket> queue_;
  RgaVisualizer visualizer_;

  void loop();
};

#endif
