#include "rga_engine.h"
#include <cstring>
#include <iostream>

RgaEngine::RgaEngine() = default;
RgaEngine::~RgaEngine() {
  free_buf(full_bgr_);
  free_buf(gray_nv12_);
  for (auto &b : roi_bufs_)
    free_buf(b);
}

RgaEngine::Buf RgaEngine::alloc_buf(int w, int h, int fmt, size_t total_size) {
  Buf b;
  b.w = w;
  b.h = h;
  b.fmt = fmt;
  b.size = total_size;
  if (posix_memalign(&b.ptr, 4096, total_size) != 0)
    return {};
  memset(b.ptr, 0, total_size);
  b.handle = importbuffer_virtualaddr(b.ptr, total_size);
  if (b.handle <= 0) {
    free(b.ptr);
    b.ptr = nullptr;
    return {};
  }
  b.buffer = wrapbuffer_handle(b.handle, w, h, fmt);
  return b;
}

void RgaEngine::free_buf(Buf &b) {
  if (b.handle > 0)
    releasebuffer_handle(b.handle);
  if (b.ptr)
    free(b.ptr);
  b = {};
}

bool RgaEngine::init(int num_rois, int roi_w, int roi_h, int gray_w,
                     int gray_h) {
  full_bgr_ = alloc_buf(1920, 1080, RK_FORMAT_BGR_888, 1920ull * 1080 * 3);
  if (!full_bgr_.ptr)
    return false;

  gray_nv12_ = alloc_buf(gray_w, gray_h, RK_FORMAT_YCbCr_420_SP,
                         (size_t)gray_w * gray_h * 3 / 2);
  if (!gray_nv12_.ptr)
    return false;

  roi_bufs_.resize(num_rois);
  for (int i = 0; i < num_rois; ++i) {
    roi_bufs_[i] =
        alloc_buf(roi_w, roi_h, RK_FORMAT_BGR_888, (size_t)roi_w * roi_h * 3);
    if (!roi_bufs_[i].ptr)
      return false;
  }
  return true;
}

// 辅助：构造空的 pat 和 prect
static inline rga_buffer_t empty_rga_buffer() {
  rga_buffer_t buf = {};
  return buf;
}

static inline im_rect empty_im_rect() {
  im_rect rect = {0, 0, 0, 0};
  return rect;
}

bool RgaEngine::convert_full_bgr(uint8_t *src_ptr, int src_stride, int src_w,
                                 int src_h, int src_fmt) {
  if (!src_ptr)
    return false;
  rga_buffer_handle_t h =
      importbuffer_virtualaddr(src_ptr, (size_t)src_stride * src_h * 2);
  if (h <= 0)
    return false;
  rga_buffer_t src = wrapbuffer_handle(h, src_stride, src_h, src_fmt);

  im_rect src_rect = {0, 0, src_w, src_h};
  im_rect dst_rect = {0, 0, 1920, 1080};

  IM_STATUS ret = improcess(src, full_bgr_.buffer, empty_rga_buffer(), src_rect,
                            dst_rect, empty_im_rect(), -1, NULL, NULL, 0);
  releasebuffer_handle(h);
  if (ret != IM_STATUS_SUCCESS)
    return false;

#ifdef __aarch64__
  __builtin___clear_cache((char *)full_bgr_.ptr,
                          (char *)full_bgr_.ptr + full_bgr_.size);
#endif
  return true;
}

uint8_t *RgaEngine::full_bgr_ptr() const { return (uint8_t *)full_bgr_.ptr; }

bool RgaEngine::convert_gray(uint8_t *src_ptr, int src_stride, int src_w,
                             int src_h, int src_fmt) {
  if (!src_ptr)
    return false;
  rga_buffer_handle_t h =
      importbuffer_virtualaddr(src_ptr, (size_t)src_stride * src_h * 2);
  if (h <= 0)
    return false;
  rga_buffer_t src = wrapbuffer_handle(h, src_stride, src_h, src_fmt);

  im_rect src_rect = {0, 0, src_w, src_h};
  im_rect dst_rect = {0, 0, gray_nv12_.w, gray_nv12_.h};

  IM_STATUS ret =
      improcess(src, gray_nv12_.buffer, empty_rga_buffer(), src_rect, dst_rect,
                empty_im_rect(), -1, NULL, NULL, 0);
  releasebuffer_handle(h);
  if (ret != IM_STATUS_SUCCESS)
    return false;

#ifdef __aarch64__
  __builtin___clear_cache((char *)gray_nv12_.ptr,
                          (char *)gray_nv12_.ptr + gray_nv12_.size);
#endif
  return true;
}

uint8_t *RgaEngine::gray_ptr() const { return (uint8_t *)gray_nv12_.ptr; }
int RgaEngine::gray_stride() const { return gray_nv12_.w; }

bool RgaEngine::crop_roi(uint8_t *src_ptr, int src_stride, int src_w, int src_h,
                         int src_fmt, int x, int y, int idx) {
  if (!src_ptr || idx >= (int)roi_bufs_.size())
    return false;
  rga_buffer_handle_t h =
      importbuffer_virtualaddr(src_ptr, (size_t)src_stride * src_h * 2);
  if (h <= 0)
    return false;
  rga_buffer_t src = wrapbuffer_handle(h, src_stride, src_h, src_fmt);

  im_rect src_rect = {x, y, roi_bufs_[idx].w, roi_bufs_[idx].h};
  im_rect dst_rect = {0, 0, roi_bufs_[idx].w, roi_bufs_[idx].h};

  IM_STATUS ret =
      improcess(src, roi_bufs_[idx].buffer, empty_rga_buffer(), src_rect,
                dst_rect, empty_im_rect(), -1, NULL, NULL, 0);
  releasebuffer_handle(h);
  if (ret != IM_STATUS_SUCCESS)
    return false;

#ifdef __aarch64__
  __builtin___clear_cache((char *)roi_bufs_[idx].ptr,
                          (char *)roi_bufs_[idx].ptr + roi_bufs_[idx].size);
#endif
  return true;
}

uint8_t *RgaEngine::roi_ptr(int idx) const {
  if (idx >= (int)roi_bufs_.size())
    return nullptr;
  return (uint8_t *)roi_bufs_[idx].ptr;
}

bool RgaEngine::crop_to_ptr(uint8_t *src_ptr, int src_stride, int src_w,
                            int src_h, int src_fmt, int x, int y,
                            uint8_t *dst_ptr, int dst_w, int dst_h) {
  if (!src_ptr || !dst_ptr)
    return false;

  rga_buffer_handle_t src_handle =
      importbuffer_virtualaddr(src_ptr, (size_t)src_stride * src_h * 2);
  if (src_handle <= 0)
    return false;
  rga_buffer_t src = wrapbuffer_handle(src_handle, src_stride, src_h, src_fmt);

  rga_buffer_handle_t dst_handle =
      importbuffer_virtualaddr(dst_ptr, (size_t)dst_w * dst_h * 3);
  if (dst_handle <= 0) {
    releasebuffer_handle(src_handle);
    return false;
  }
  rga_buffer_t dst =
      wrapbuffer_handle(dst_handle, dst_w, dst_h, RK_FORMAT_BGR_888);

  im_rect src_rect = {x, y, dst_w, dst_h};
  im_rect dst_rect = {0, 0, dst_w, dst_h};

  IM_STATUS ret = improcess(src, dst, empty_rga_buffer(), src_rect, dst_rect,
                            empty_im_rect(), -1, NULL, NULL, 0);
  releasebuffer_handle(src_handle);
  releasebuffer_handle(dst_handle);

  if (ret != IM_STATUS_SUCCESS) {
    std::cerr << "[RGA] crop_to_ptr failed\n";
    return false;
  }

#ifdef __aarch64__
  __builtin___clear_cache((char *)dst_ptr,
                          (char *)dst_ptr + (size_t)dst_w * dst_h * 3);
#endif
  return true;
}
