#include "rga_cropper.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-heap.h>
#include <cstring>
#include <iostream>
#include <fstream>

#define RGA_LOG(msg) std::cout << "[RgaCropper] " << msg << "\n"
#define RGA_ERR(msg) std::cerr << "[RgaCropper] ERROR: " << msg << "\n"

RgaCropper::RgaCropper(int src_w, int src_h, int tile_w, int tile_h,
                       int tile_cols, int tile_rows,
                       int overlap_w, int overlap_h)
    : src_w_(src_w), src_h_(src_h), tile_w_(tile_w), tile_h_(tile_h),
      tile_cols_(tile_cols), tile_rows_(tile_rows),
      overlap_w_(overlap_w), overlap_h_(overlap_h)
{
    // 计算分块区域（支持重叠）
    int step_x = tile_w_ - overlap_w_;
    int step_y = tile_h_ - overlap_h_;

    for (int row = 0; row < tile_rows_; ++row) {
        for (int col = 0; col < tile_cols_; ++col) {
            im_rect rect;
            rect.x = col * step_x;
            rect.y = row * step_y;
            rect.width  = tile_w_;
            rect.height = tile_h_;
            src_rects_.push_back(rect);
        }
    }

    // 验证最后一个块不超出源图
    if (!src_rects_.empty()) {
        const im_rect& last = src_rects_.back();
        int max_x = last.x + last.width;
        int max_y = last.y + last.height;
        if (max_x > src_w_ || max_y > src_h_) {
            RGA_ERR("Tile grid exceeds source image! "
                    << "last_tile=(" << last.x << "," << last.y << ","
                    << last.width << "," << last.height << ")"
                    << " src=" << src_w_ << "x" << src_h_);
        }
    }

    RGA_LOG("Config: src=" << src_w_ << "x" << src_h_
            << " tile=" << tile_w_ << "x" << tile_h_
            << " grid=" << tile_cols_ << "x" << tile_rows_
            << " overlap=" << overlap_w_ << "x" << overlap_h_
            << " tiles=" << tile_count());
}

RgaCropper::~RgaCropper() {
    release_dma_buf();
}

bool RgaCropper::alloc_dma_buf(size_t size) {
    int heap_fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        RGA_ERR("open system failed: " << strerror(errno) << ", fallback to system-uncached");
        heap_fd = open("/dev/dma_heap/system-uncached", O_RDWR | O_CLOEXEC);
        if (heap_fd < 0) {
            RGA_ERR("open system-uncached failed: " << strerror(errno));
            return false;
        }
    }

    struct dma_heap_allocation_data alloc_data;
    memset(&alloc_data, 0, sizeof(alloc_data));
    alloc_data.len = size;
    alloc_data.fd = 0;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
    alloc_data.heap_flags = 0;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
        RGA_ERR("DMA_HEAP_IOCTL_ALLOC failed: " << strerror(errno));
        close(heap_fd);
        return false;
    }

    close(heap_fd);
    dst_fd_ = alloc_data.fd;
    dst_size_ = size;
    RGA_LOG("dst dma_buf allocated: fd=" << dst_fd_ << " size=" << size);
    return true;
}

void RgaCropper::release_dma_buf() {
    if (dst_fd_ >= 0) {
        close(dst_fd_);
        dst_fd_ = -1;
        dst_size_ = 0;
    }
}

bool RgaCropper::init() {
    dst_size_ = tile_w_ * tile_h_ * 3 * tile_count();
    return alloc_dma_buf(dst_size_);
}

bool RgaCropper::process(int src_fd, int src_fmt, int src_w, int src_h) {
    if (dst_fd_ < 0) {
        RGA_ERR("dst_fd not ready, call init() first");
        return false;
    }

    rga_buffer_t src = wrapbuffer_fd(src_fd, src_w, src_h, src_fmt, src_w, src_h);
    src.vir_addr = nullptr;
    src.phy_addr = nullptr;

    int dst_total_h = tile_h_ * tile_count();
    rga_buffer_t dst = wrapbuffer_fd(dst_fd_, tile_w_, dst_total_h,
                                      RK_FORMAT_RGB_888, tile_w_, dst_total_h);
    dst.vir_addr = nullptr;
    dst.phy_addr = nullptr;

    rga_buffer_t pat = {};
    memset(&pat, 0, sizeof(pat));

    for (size_t i = 0; i < src_rects_.size(); ++i) {
        im_rect dst_rect = {0, (int)(i * tile_h_), tile_w_, tile_h_};

        IM_STATUS ret = ::improcess(src, dst, pat,
                                    src_rects_[i], dst_rect, {},
                                    IM_SYNC);
        if (ret != IM_STATUS_SUCCESS) {
            RGA_ERR("improcess task " << i << " failed: " << imStrError(ret));
            return false;
        }
    }

    return true;
}

bool RgaCropper::save_bmp(const std::string& path, const uint8_t* rgb_data, int w, int h) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        RGA_ERR("Cannot open " << path);
        return false;
    }

    int row_stride = w * 3;
    int padding = (4 - (row_stride % 4)) % 4;
    int bmp_row_stride = row_stride + padding;
    int data_size = bmp_row_stride * h;
    int file_size = 14 + 40 + data_size;

    uint8_t fh[14] = { 'B','M', 0,0,0,0, 0,0,0,0, 54,0,0,0 };
    fh[2] = file_size & 0xFF;
    fh[3] = (file_size >> 8) & 0xFF;
    fh[4] = (file_size >> 16) & 0xFF;
    fh[5] = (file_size >> 24) & 0xFF;

    uint8_t ih[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,24,0,
                       0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
                       0,0,0,0, 0,0,0,0 };
    ih[4] = w & 0xFF; ih[5] = (w >> 8) & 0xFF; ih[6] = (w >> 16) & 0xFF; ih[7] = (w >> 24) & 0xFF;
    ih[8] = h & 0xFF; ih[9] = (h >> 8) & 0xFF; ih[10] = (h >> 16) & 0xFF; ih[11] = (h >> 24) & 0xFF;

    ofs.write((char*)fh, 14);
    ofs.write((char*)ih, 40);

    std::vector<uint8_t> row(bmp_row_stride, 0);
    for (int y = h - 1; y >= 0; --y) {
        const uint8_t* src = rgb_data + y * w * 3;
        for (int x = 0; x < w; ++x) {
            row[x * 3 + 0] = src[x * 3 + 2];
            row[x * 3 + 1] = src[x * 3 + 1];
            row[x * 3 + 2] = src[x * 3 + 0];
        }
        if (padding) memset(row.data() + row_stride, 0, padding);
        ofs.write((char*)row.data(), bmp_row_stride);
    }

    ofs.close();
    return true;
}

bool RgaCropper::save_tiles(int dst_fd, size_t dst_size, int tile_w, int tile_h,
                            int tile_count, const std::string& prefix) {
    if (dst_fd < 0) {
        RGA_ERR("Invalid dst_fd");
        return false;
    }

    void* ptr = mmap(nullptr, dst_size, PROT_READ, MAP_SHARED, dst_fd, 0);
    if (ptr == MAP_FAILED) {
        RGA_ERR("mmap dst_fd failed: " << strerror(errno));
        return false;
    }

    uint8_t* data = (uint8_t*)ptr;
    size_t tile_size = tile_w * tile_h * 3;
    bool ok = true;

    for (int i = 0; i < tile_count; ++i) {
        std::string path = prefix + "_" + std::to_string(i) + ".bmp";
        if (!save_bmp(path, data + i * tile_size, tile_w, tile_h)) {
            RGA_ERR("Failed to save " << path);
            ok = false;
            break;
        }
        RGA_LOG("Saved " << path);
    }

    munmap(ptr, dst_size);
    return ok;
}
