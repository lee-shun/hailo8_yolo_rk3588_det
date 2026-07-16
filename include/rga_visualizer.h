#ifndef RGA_VISUALIZER_H
#define RGA_VISUALIZER_H

// 1. 先包含标准库（修复 deque + OpenCV 模板顺序问题）
#include <cstddef> // 修复 im2d.hpp 中 NULL 未声明
#include <deque>
#include <vector>

// 2. 再包含 OpenCV
#include <opencv2/opencv.hpp>

// 3. 最后包含 librga
#include <im2d.hpp>

#include "fusion_roi_predictor.h"

class RgaVisualizer {
public:
  RgaVisualizer();
  ~RgaVisualizer();

  bool init();
  cv::Mat draw(cv::Mat &frame, const PBASResult &pbas,
               const std::vector<ROI> &rois, const std::vector<Detection> &dets,
               bool hit);
  void reset_trail();

private:
  struct Buf {
    void *ptr = nullptr;
    rga_buffer_handle_t handle = 0;
    rga_buffer_t buffer = {};
    int w = 0, h = 0;
    size_t size = 0;
  };

  Buf alloc_buf(int w, int h, int fmt, size_t size);
  void free_buf(Buf &b);

  static inline int rga_color(uint8_t b, uint8_t g, uint8_t r) {
    return (b) | (g << 8) | (r << 16);
  }

  void rga_blend_mask(uint8_t *dst_ptr, int dst_w, int dst_h,
                      const uint8_t *mask_ptr, int mask_w, int mask_h);
  void rga_draw_roi_box(uint8_t *dst_ptr, int dst_w, int dst_h, int x, int y,
                        int w, int h, int color, int thickness);
  void rga_draw_crosshair(uint8_t *dst_ptr, int dst_w, int dst_h, float cx,
                          float cy, float dx, float dy);

  void cv_draw_dets(cv::Mat &frame, const std::vector<Detection> &dets);
  void cv_draw_roi_text(cv::Mat &frame, const std::vector<ROI> &rois);
  void cv_draw_hud(cv::Mat &frame, const PBASResult &pbas, bool hit);

  Buf mask_bgr_;
  std::deque<cv::Point2f> trail_;
  static constexpr size_t MAX_TRAIL = 40;
};

#endif
