#include "rga_engine.h"
#include <cstring>
#include <iostream>

#define RGA_LOG(msg) std::cout << "[RGA] " << msg << "\n"

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
  return b;
}

void RgaEngine::free_buf(Buf &b) {
  if (b.ptr)
    free(b.ptr);
  b = {};
}

bool RgaEngine::init(int num_rois, int roi_w, int roi_h, int gray_w,
                     int gray_h) {
  RGA_LOG("init start: num_rois=" << num_rois << " roi=" << roi_w << "x"
                                  << roi_h << " gray=" << gray_w << "x"
                                  << gray_h);

  full_bgr_ = alloc_buf(1920, 1080, RK_FORMAT_BGR_888, 1920ull * 1080 * 3);
  if (!full_bgr_.ptr) {
    RGA_LOG("init failed: full_bgr");
    return false;
  }

  gray_nv12_ = alloc_buf(gray_w, gray_h, RK_FORMAT_YCbCr_420_SP,
                         (size_t)gray_w * gray_h * 3 / 2);
  if (!gray_nv12_.ptr) {
    RGA_LOG("init failed: gray_nv12");
    return false;
  }

  roi_bufs_.resize(num_rois);
  for (int i = 0; i < num_rois; ++i) {
    roi_bufs_[i] =
        alloc_buf(roi_w, roi_h, RK_FORMAT_BGR_888, (size_t)roi_w * roi_h * 3);
    if (!roi_bufs_[i].ptr) {
      RGA_LOG("init failed: roi_buf[" << i << "]");
      return false;
    }
  }
  RGA_LOG("init OK");
  return true;
}

static inline rga_buffer_t empty_rga_buffer() {
  rga_buffer_t buf = {};
  return buf;
}

static inline im_rect empty_im_rect() {
  im_rect rect = {0, 0, 0, 0};
  return rect;
}

// ============================================================================
// convert_full_bgr: YUYV -> BGR 1920x1080（同尺寸格式转换）
// 用 imcvtcolor 更简洁，sync 默认就是 1
// ============================================================================
bool RgaEngine::convert_full_bgr(uint8_t *src_ptr, int src_stride, int src_w,
                                 int src_h, int src_fmt) {
  if (!src_ptr) {
    RGA_LOG("convert_full_bgr failed: src_ptr is null");
    return false;
  }

  RGA_LOG("convert_full_bgr: src=" << (void *)src_ptr << " v4l2_stride="
                                   << src_stride << " img=" << src_w << "x"
                                   << src_h << " fmt=" << src_fmt);

  // 4 参数：像素宽、像素高、格式（wstride 默认 = width，单位像素）
  rga_buffer_t src = wrapbuffer_virtualaddr(src_ptr, src_w, src_h, src_fmt);
  rga_buffer_t dst = wrapbuffer_virtualaddr(full_bgr_.ptr, full_bgr_.w,
                                            full_bgr_.h, full_bgr_.fmt);

  // 同尺寸格式转换直接用 imcvtcolor，sync=1 默认同步
  IM_STATUS ret =
      imcvtcolor(src, dst, src_fmt, full_bgr_.fmt, IM_COLOR_SPACE_DEFAULT, 1);
  if (ret != IM_STATUS_SUCCESS) {
    RGA_LOG("convert_full_bgr failed: " << imStrError(ret));
    return false;
  }

#ifdef __aarch64__
  __builtin___clear_cache((char *)full_bgr_.ptr,
                          (char *)full_bgr_.ptr + full_bgr_.size);
#endif
  RGA_LOG("convert_full_bgr done");
  return true;
}

uint8_t *RgaEngine::full_bgr_ptr() const { return (uint8_t *)full_bgr_.ptr; }

// ============================================================================
// convert_gray: YUYV -> NV12 320x180（格式转换 + 缩放）
// 必须用 improcess，usage = IM_SYNC
// ============================================================================
bool RgaEngine::convert_gray(uint8_t *src_ptr, int src_stride, int src_w,
                             int src_h, int src_fmt) {
  if (!src_ptr) {
    RGA_LOG("convert_gray failed: src_ptr is null");
    return false;
  }

  RGA_LOG("convert_gray: src=" << (void *)src_ptr << " v4l2_stride="
                               << src_stride << " img=" << src_w << "x" << src_h
                               << " fmt=" << src_fmt);

  rga_buffer_t src = wrapbuffer_virtualaddr(src_ptr, src_w, src_h, src_fmt);
  rga_buffer_t dst = wrapbuffer_virtualaddr(gray_nv12_.ptr, gray_nv12_.w,
                                            gray_nv12_.h, gray_nv12_.fmt);

  im_rect src_rect = {0, 0, src_w, src_h};
  im_rect dst_rect = {0, 0, gray_nv12_.w, gray_nv12_.h};

  IM_STATUS ret = improcess(src, dst, empty_rga_buffer(), src_rect, dst_rect,
                            empty_im_rect(), -1, NULL, NULL, IM_SYNC);
  if (ret != IM_STATUS_SUCCESS) {
    RGA_LOG("convert_gray failed: " << imStrError(ret));
    return false;
  }

#ifdef __aarch64__
  __builtin___clear_cache((char *)gray_nv12_.ptr,
                          (char *)gray_nv12_.ptr + gray_nv12_.size);
#endif
  RGA_LOG("convert_gray done");
  return true;
}

uint8_t *RgaEngine::gray_ptr() const { return (uint8_t *)gray_nv12_.ptr; }
int RgaEngine::gray_stride() const { return gray_nv12_.w; }

// ============================================================================
// crop_roi: 裁剪 + 格式转换（YUYV -> BGR）
// ============================================================================
bool RgaEngine::crop_roi(uint8_t *src_ptr, int src_stride, int src_w, int src_h,
                         int src_fmt, int x, int y, int idx) {
  if (!src_ptr || idx >= (int)roi_bufs_.size()) {
    RGA_LOG("crop_roi failed: src_ptr=" << (void *)src_ptr << " idx=" << idx);
    return false;
  }

  RGA_LOG("crop_roi: src=" << (void *)src_ptr << " v4l2_stride=" << src_stride
                           << " roi=(" << x << "," << y << ")"
                           << " idx=" << idx);

  rga_buffer_t src = wrapbuffer_virtualaddr(src_ptr, src_w, src_h, src_fmt);
  rga_buffer_t dst =
      wrapbuffer_virtualaddr(roi_bufs_[idx].ptr, roi_bufs_[idx].w,
                             roi_bufs_[idx].h, roi_bufs_[idx].fmt);

  im_rect src_rect = {x, y, roi_bufs_[idx].w, roi_bufs_[idx].h};
  im_rect dst_rect = {0, 0, roi_bufs_[idx].w, roi_bufs_[idx].h};

  // [关键修复] usage = IM_SYNC
  IM_STATUS ret = improcess(src, dst, empty_rga_buffer(), src_rect, dst_rect,
                            empty_im_rect(), -1, NULL, NULL, IM_SYNC);
  if (ret != IM_STATUS_SUCCESS) {
    RGA_LOG("crop_roi failed: " << imStrError(ret));
    return false;
  }

#ifdef __aarch64__
  __builtin___clear_cache((char *)roi_bufs_[idx].ptr,
                          (char *)roi_bufs_[idx].ptr + roi_bufs_[idx].size);
#endif
  RGA_LOG("crop_roi done");
  return true;
}

uint8_t *RgaEngine::roi_ptr(int idx) const {
  if (idx >= (int)roi_bufs_.size())
    return nullptr;
  return (uint8_t *)roi_bufs_[idx].ptr;
}

// ============================================================================
// crop_to_ptr: 裁剪到外部指针
// ============================================================================
bool RgaEngine::crop_to_ptr(uint8_t *src_ptr, int src_stride, int src_w,
                            int src_h, int src_fmt, int x, int y,
                            uint8_t *dst_ptr, int dst_w, int dst_h) {
  if (!src_ptr || !dst_ptr) {
    RGA_LOG("crop_to_ptr failed: src_ptr=" << (void *)src_ptr
                                           << " dst_ptr=" << (void *)dst_ptr);
    return false;
  }

  RGA_LOG("crop_to_ptr: src=" << (void *)src_ptr << " dst=" << (void *)dst_ptr
                              << " roi=(" << x << "," << y << ") " << dst_w
                              << "x" << dst_h);

  rga_buffer_t src = wrapbuffer_virtualaddr(src_ptr, src_w, src_h, src_fmt);
  rga_buffer_t dst =
      wrapbuffer_virtualaddr(dst_ptr, dst_w, dst_h, RK_FORMAT_BGR_888);

  im_rect src_rect = {x, y, dst_w, dst_h};
  im_rect dst_rect = {0, 0, dst_w, dst_h};

  // [关键修复] usage = IM_SYNC
  IM_STATUS ret = improcess(src, dst, empty_rga_buffer(), src_rect, dst_rect,
                            empty_im_rect(), -1, NULL, NULL, IM_SYNC);
  if (ret != IM_STATUS_SUCCESS) {
    RGA_LOG("crop_to_ptr failed: " << imStrError(ret));
    return false;
  }

#ifdef __aarch64__
  __builtin___clear_cache((char *)dst_ptr,
                          (char *)dst_ptr + (size_t)dst_w * dst_h * 3);
#endif
  RGA_LOG("crop_to_ptr done");
  return true;
}
