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
    // tile_cols: 水平切几块, tile_rows: 垂直切几块
    RgaCropper(int src_w, int src_h, int tile_w, int tile_h, int tile_cols, int tile_rows);
    ~RgaCropper();

    // 申请 dst dma_buf（只需一次，循环复用）
    bool init();

    // 执行 RGA 色转 + Crop + 拼接（内部使用 Job API 批量提交）
    // src_fd:    YUV dma_buf fd（来自 CamDmaMmap::src_fd()）
    // src_fmt:   RK_FORMAT_YUYV_422 等（来自 CamDmaMmap::src_fmt()）
    // src_w/src_h: 图像宽高（像素，直接传 cam.src_w() / cam.src_h() 即可）
    bool process(int src_fd, int src_fmt, int src_w, int src_h);

    int dst_fd() const { return dst_fd_; }
    size_t dst_size() const { return dst_size_; }
    int tile_w() const { return tile_w_; }
    int tile_h() const { return tile_h_; }
    int tile_count() const { return tile_cols_ * tile_rows_; }

    // 从 dst_fd 按 tile 偏移保存为 BMP（只保存一次）
    static bool save_tiles(int dst_fd, size_t dst_size, int tile_w, int tile_h,
                           int tile_count, const std::string& prefix);

private:
    int src_w_, src_h_;
    int tile_w_, tile_h_;
    int tile_cols_, tile_rows_;
    int dst_fd_ = -1;
    size_t dst_size_ = 0;
    std::vector<im_rect> src_rects_;

    bool alloc_dma_buf(size_t size);
    void release_dma_buf();
    static bool save_bmp(const std::string& path, const uint8_t* rgb_data, int w, int h);
};

#endif
