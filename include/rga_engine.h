#ifndef RGA_ENGINE_H
#define RGA_ENGINE_H

#include <cstddef> // 修复 im2d.hpp NULL 问题
#include <cstdlib>
#include <vector>

#include <im2d.hpp>
// RGA 硬件图像处理引擎
// 负责所有图像变换：全图 BGR 转换、灰度缩放、ROI 裁剪+色转
// 输入为 Camera 输出的源指针，通过 IOMMU 映射实现零拷贝
class RgaEngine {
public:
  RgaEngine();
  ~RgaEngine();

  bool init(int num_rois, int roi_w, int roi_h, int gray_w, int gray_h);

  bool convert_full_bgr(uint8_t *src_ptr, int src_stride, int src_w, int src_h,
                        int src_fmt);
  uint8_t *full_bgr_ptr() const;

  bool convert_gray(uint8_t *src_ptr, int src_stride, int src_w, int src_h,
                    int src_fmt);
  uint8_t *gray_ptr() const;
  int gray_stride() const;

  bool crop_roi(uint8_t *src_ptr, int src_stride, int src_w, int src_h,
                int src_fmt, int x, int y, int idx);
  uint8_t *roi_ptr(int idx) const;

  bool crop_to_ptr(uint8_t *src_ptr, int src_stride, int src_w, int src_h,
                   int src_fmt, int x, int y, uint8_t *dst_ptr, int dst_w,
                   int dst_h);

private:
  struct Buf {
    void *ptr = nullptr;
    rga_buffer_handle_t handle = 0;
    rga_buffer_t buffer = {};
    int w = 0, h = 0, fmt = 0;
    size_t size = 0;
  };

  Buf alloc_buf(int w, int h, int fmt, size_t total_size);
  void free_buf(Buf &b);

  Buf full_bgr_;
  Buf gray_nv12_;
  std::vector<Buf> roi_bufs_;
};

#endif
