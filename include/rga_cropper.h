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
    RgaCropper(int src_w, int src_h, int tile_w, int tile_h,
               int tile_cols, int tile_rows, int overlap_w, int overlap_h);
    ~RgaCropper();

    bool init();
    bool process(int src_fd, int src_fmt, int src_w, int src_h);

    // 保持兼容的旧接口（单一大 buffer 模式已废弃，dst_fd() 返回 -1）
    int dst_fd() const;
    size_t dst_size() const;
    int tile_count() const;

    // 新增：真零拷贝接口
    int tile_fd(int idx) const;          // 获取第 idx 个 tile 的独立 dmabuf fd
    size_t tile_size() const;            // 单个 tile 的字节数

    // 保持兼容的 static 版本（仅当传入有效 fd 时可用）
    static bool save_tiles(int dst_fd, size_t dst_size, int tile_w, int tile_h,
                           int tile_count, const std::string& prefix);
    // 新增：使用内部独立 tile fds 保存
    bool save_tiles(const std::string& prefix) const;

private:
    bool alloc_dma_buf(size_t size);
    bool alloc_tile_dma_bufs();
    void release_dma_buf();
    void release_tile_dma_bufs();
    static bool save_bmp(const std::string& path, const uint8_t* rgb_data, int w, int h);

    int src_w_, src_h_, tile_w_, tile_h_;
    int tile_cols_, tile_rows_;
    int overlap_w_, overlap_h_;
    std::vector<im_rect> src_rects_;

    // 兼容旧接口的占位（不再分配）
    int dst_fd_ = -1;
    size_t dst_size_ = 0;

    // 真零拷贝：每个 tile 独立的 dmabuf fd
    std::vector<int> tile_fds_;
    size_t tile_size_ = 0;
};

#endif
