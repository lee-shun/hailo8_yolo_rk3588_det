#ifndef RGA_CROPPER_H
#define RGA_CROPPER_H

#include <stddef.h> // 修复 im2d.hpp NULL 问题
#include <stdlib.h>
#include <vector>

#include <im2d.hpp>
#include <string>
#include <vector>

class RgaCropper {
public:
  // overlap_w: 水平方向相邻块的重叠像素（默认0）
  // overlap_h: 垂直方向相邻块的重叠像素（默认0）
  RgaCropper(int src_w, int src_h, int tile_w, int tile_h, int tile_cols,
             int tile_rows, int overlap_w = 0, int overlap_h = 0);
  ~RgaCropper();

  bool init();
  bool process(int src_fd, int src_fmt, int src_w, int src_h);

  int dst_fd() const { return dst_fd_; }
  size_t dst_size() const { return dst_size_; }
  int tile_w() const { return tile_w_; }
  int tile_h() const { return tile_h_; }
  int tile_count() const { return tile_cols_ * tile_rows_; }
  int overlap_w() const { return overlap_w_; }
  int overlap_h() const { return overlap_h_; }

  static bool save_tiles(int dst_fd, size_t dst_size, int tile_w, int tile_h,
                         int tile_count, const std::string &prefix);

private:
  int src_w_, src_h_;
  int tile_w_, tile_h_;
  int tile_cols_, tile_rows_;
  int overlap_w_, overlap_h_;
  int dst_fd_ = -1;
  size_t dst_size_ = 0;
  std::vector<im_rect> src_rects_;

  bool alloc_dma_buf(size_t size);
  void release_dma_buf();
  static bool save_bmp(const std::string &path, const uint8_t *rgb_data, int w,
                       int h);
};

#endif
