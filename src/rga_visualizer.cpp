#include "rga_visualizer.h"
#include <iostream>

RgaVisualizer::RgaVisualizer() = default;
RgaVisualizer::~RgaVisualizer() { free_buf(mask_bgr_); }

RgaVisualizer::Buf RgaVisualizer::alloc_buf(int w, int h, int fmt,
                                            size_t size) {
  Buf b;
  b.w = w;
  b.h = h;
  b.size = size;
  if (posix_memalign(&b.ptr, 4096, size) != 0)
    return {};
  memset(b.ptr, 0, size);
  b.handle = importbuffer_virtualaddr(b.ptr, size);
  if (b.handle <= 0) {
    free(b.ptr);
    b.ptr = nullptr;
    return {};
  }
  b.buffer = wrapbuffer_handle(b.handle, w, h, fmt);
  return b;
}

void RgaVisualizer::free_buf(Buf &b) {
  if (b.handle > 0)
    releasebuffer_handle(b.handle);
  if (b.ptr)
    free(b.ptr);
  b = {};
}

bool RgaVisualizer::init() {
  mask_bgr_ = alloc_buf(1920, 1080, RK_FORMAT_BGR_888, 1920ull * 1080 * 3);
  if (!mask_bgr_.ptr)
    return false;
  return true;
}

void RgaVisualizer::reset_trail() { trail_.clear(); }

cv::Mat RgaVisualizer::draw(cv::Mat &frame, const PBASResult &pbas,
                            const std::vector<ROI> &rois,
                            const std::vector<Detection> &dets, bool hit) {
  int frame_w = frame.cols;
  int frame_h = frame.rows;
  uint8_t *frame_ptr = frame.data;

  if (!pbas.fg_mask.empty() && pbas.fg_mask.cols == 320 &&
      pbas.fg_mask.rows == 180) {
    rga_blend_mask(frame_ptr, frame_w, frame_h, pbas.fg_mask.data,
                   pbas.fg_mask.cols, pbas.fg_mask.rows);
  }

  const cv::Scalar spectrum[7] = {{0, 0, 255},  {0, 140, 255}, {0, 255, 255},
                                  {0, 255, 0},  {255, 255, 0}, {255, 0, 0},
                                  {255, 0, 255}};
  for (size_t i = 0; i < rois.size() && i < 7; ++i) {
    const ROI &r = rois[i];
    int color = rga_color((uint8_t)spectrum[i][0], (uint8_t)spectrum[i][1],
                          (uint8_t)spectrum[i][2]);
    rga_draw_roi_box(frame_ptr, frame_w, frame_h, r.x, r.y, r.w, r.h, 0x000000,
                     3);
    rga_draw_roi_box(frame_ptr, frame_w, frame_h, r.x, r.y, r.w, r.h, color, 4);
    rga_draw_roi_box(frame_ptr, frame_w, frame_h, r.x + 4, r.y + 4, r.w - 8,
                     r.h - 8, 0xFFFFFF, 1);
  }

  if (pbas.curr_centroid.x > 0 && pbas.curr_centroid.x < frame_w &&
      pbas.curr_centroid.y > 0 && pbas.curr_centroid.y < frame_h) {
    rga_draw_crosshair(frame_ptr, frame_w, frame_h, pbas.curr_centroid.x,
                       pbas.curr_centroid.y, pbas.bg_dx, pbas.bg_dy);
  }

  cv_draw_dets(frame, dets);
  cv_draw_roi_text(frame, rois);
  cv_draw_hud(frame, pbas, hit);

  return frame;
}

void RgaVisualizer::rga_blend_mask(uint8_t *dst_ptr, int dst_w, int dst_h,
                                   const uint8_t *mask_ptr, int mask_w,
                                   int mask_h) {
  cv::Mat mask_small(mask_h, mask_w, CV_8UC1, (void *)mask_ptr, mask_w);
  cv::Mat mask_large;
  cv::resize(mask_small, mask_large, cv::Size(dst_w, dst_h), 0, 0,
             cv::INTER_NEAREST);

  cv::Mat overlay(dst_h, dst_w, CV_8UC3, dst_ptr, dst_w * 3);
  for (int y = 0; y < dst_h; ++y) {
    uint8_t *mask_row = mask_large.ptr<uint8_t>(y);
    cv::Vec3b *dst_row = overlay.ptr<cv::Vec3b>(y);
    for (int x = 0; x < dst_w; ++x) {
      if (mask_row[x] > 128) {
        dst_row[x][0] = (uint8_t)(dst_row[x][0] * 0.8f);
        dst_row[x][1] = (uint8_t)(dst_row[x][1] * 0.8f + 51);
        dst_row[x][2] = (uint8_t)(dst_row[x][2] * 0.8f);
      }
    }
  }
}

// 辅助函数
static inline rga_buffer_t empty_rga_buffer() {
  rga_buffer_t buf = {};
  return buf;
}

static inline im_rect empty_im_rect() {
  im_rect rect = {0, 0, 0, 0};
  return rect;
}

// ============================================================================
// ROI 空心框：imfill 是值传递 im_rect，不是指针
// ============================================================================
void RgaVisualizer::rga_draw_roi_box(uint8_t *dst_ptr, int dst_w, int dst_h,
                                     int x, int y, int w, int h, int color,
                                     int thickness) {
  if (x < 0 || y < 0 || w <= 0 || h <= 0)
    return;

  rga_buffer_handle_t dst_hnd =
      importbuffer_virtualaddr(dst_ptr, (size_t)dst_w * dst_h * 3);
  if (dst_hnd <= 0)
    return;
  rga_buffer_t dst =
      wrapbuffer_handle(dst_hnd, dst_w, dst_h, RK_FORMAT_BGR_888);

  // imfill 签名：imfill(dst, im_rect rect, int color, int sync = 1, int
  // *release_fence_fd = NULL) im_rect 是值传递！
  im_rect top = {x, y, w, thickness};
  if (top.y + top.height <= dst_h)
    imfill(dst, top, color, 1, NULL);

  im_rect bottom = {x, y + h - thickness, w, thickness};
  if (bottom.y >= 0)
    imfill(dst, bottom, color, 1, NULL);

  im_rect left = {x, y + thickness, thickness, h - 2 * thickness};
  if (left.y + left.height <= dst_h)
    imfill(dst, left, color, 1, NULL);

  im_rect right = {x + w - thickness, y + thickness, thickness,
                   h - 2 * thickness};
  if (right.x + right.width <= dst_w)
    imfill(dst, right, color, 1, NULL);

  releasebuffer_handle(dst_hnd);

#ifdef __aarch64__
  __builtin___clear_cache((char *)dst_ptr,
                          (char *)dst_ptr + (size_t)dst_w * dst_h * 3);
#endif
}

// ============================================================================
// 质心十字线：同样 imfill 值传递
// ============================================================================
void RgaVisualizer::rga_draw_crosshair(uint8_t *dst_ptr, int dst_w, int dst_h,
                                       float cx, float cy, float dx, float dy) {
  int ix = (int)cx, iy = (int)cy;
  int len = 30, thick = 2;

  rga_buffer_handle_t dst_hnd =
      importbuffer_virtualaddr(dst_ptr, (size_t)dst_w * dst_h * 3);
  if (dst_hnd <= 0)
    return;
  rga_buffer_t dst =
      wrapbuffer_handle(dst_hnd, dst_w, dst_h, RK_FORMAT_BGR_888);

  int green = rga_color(0, 255, 0);

  im_rect h_line = {ix - len, iy - thick / 2, len * 2, thick};
  if (h_line.x >= 0 && h_line.x + h_line.width <= dst_w)
    imfill(dst, h_line, green, 1, NULL);

  im_rect v_line = {ix - thick / 2, iy - len, thick, len * 2};
  if (v_line.y >= 0 && v_line.y + v_line.height <= dst_h)
    imfill(dst, v_line, green, 1, NULL);

  releasebuffer_handle(dst_hnd);

#ifdef __aarch64__
  __builtin___clear_cache((char *)dst_ptr,
                          (char *)dst_ptr + (size_t)dst_w * dst_h * 3);
#endif

  // 运动箭头：斜线 RGA 不支持，回退 OpenCV
  float mag = std::sqrt(dx * dx + dy * dy);
  if (mag > 5.0f) {
    float scale = std::min(mag * 3.0f, 200.0f) / mag;
    cv::Point start(ix, iy);
    cv::Point end((int)(cx + dx * scale), (int)(cy + dy * scale));
    cv::Mat frame(dst_h, dst_w, CV_8UC3, dst_ptr, dst_w * 3);
    cv::arrowedLine(frame, start, end, cv::Scalar(0, 255, 0), 3, cv::LINE_AA, 0,
                    0.3);
  }
}

// ... 省略 cv_draw_dets / cv_draw_roi_text / cv_draw_hud（无变更）...
void RgaVisualizer::cv_draw_dets(cv::Mat &frame,
                                 const std::vector<Detection> &dets) {
  cv::Scalar blue(255, 0, 0);
  cv::Scalar white(255, 255, 255);

  for (const auto &d : dets) {
    int x1 = (int)d.x, y1 = (int)d.y;
    int x2 = (int)(d.x + d.w), y2 = (int)(d.y + d.h);

    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), blue, 2,
                  cv::LINE_AA);

    std::string label = cv::format("%.2f", d.score);
    int baseline = 0;
    cv::Size ts =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::Rect bg(x1, y1 - ts.height - 4, ts.width + 8, ts.height + 4);
    cv::rectangle(frame, bg, blue, -1);

    cv::putText(frame, label, cv::Point(x1 + 4, y1 - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv::LINE_AA);
  }
}

void RgaVisualizer::cv_draw_roi_text(cv::Mat &frame,
                                     const std::vector<ROI> &rois) {
  const cv::Scalar spectrum[7] = {{0, 0, 255},  {0, 140, 255}, {0, 255, 255},
                                  {0, 255, 0},  {255, 255, 0}, {255, 0, 0},
                                  {255, 0, 255}};

  for (size_t i = 0; i < rois.size() && i < 7; ++i) {
    const ROI &r = rois[i];
    std::string rank = cv::format("#%d", (int)i + 1);

    cv::Rect tag(r.x, r.y, 50, 28);
    cv::rectangle(frame, tag, spectrum[i], -1, cv::LINE_AA);

    cv::putText(frame, rank, cv::Point(r.x + 8, r.y + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2,
                cv::LINE_AA);
  }
}

void RgaVisualizer::cv_draw_hud(cv::Mat &frame, const PBASResult &pbas,
                                bool hit) {
  cv::Mat panel = frame(cv::Rect(10, 10, 420, 120));
  cv::Mat overlay;
  panel.copyTo(overlay);
  cv::rectangle(overlay, cv::Point(0, 0), cv::Point(420, 120),
                cv::Scalar(0, 0, 0), -1);
  cv::addWeighted(overlay, 0.6, panel, 0.4, 0, panel);

  cv::putText(
      frame,
      cv::format("PBAS fg:%d rel:%.2f", pbas.fg_pixels, pbas.reliability),
      cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0),
      2, cv::LINE_AA);
  cv::putText(frame, cv::format("bg_v:(%.1f,%.1f)", pbas.bg_dx, pbas.bg_dy),
              cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
  cv::putText(frame, cv::format("blobs:%d", (int)pbas.blobs.size()),
              cv::Point(20, 85), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

  if (hit) {
    cv::putText(frame, "TARGET HIT!", cv::Point(700, 540),
                cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 4,
                cv::LINE_AA);
  }
}
