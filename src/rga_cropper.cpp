#include "rga_cropper.h"
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define RGA_LOG(msg) std::cout << "[RgaCropper] " << msg << "\n"
#define RGA_ERR(msg) std::cerr << "[RgaCropper] ERROR: " << msg << "\n"

RgaCropper::RgaCropper(int src_w, int src_h, int tile_w, int tile_h,
                       int tile_cols, int tile_rows, int overlap_w,
                       int overlap_h)
    : src_w_(src_w), src_h_(src_h), tile_w_(tile_w), tile_h_(tile_h),
      tile_cols_(tile_cols), tile_rows_(tile_rows), overlap_w_(overlap_w),
      overlap_h_(overlap_h) {
  int step_x = tile_w_ - overlap_w_;
  int step_y = tile_h_ - overlap_h_;

  for (int row = 0; row < tile_rows_; ++row) {
    for (int col = 0; col < tile_cols_; ++col) {
      im_rect rect;
      rect.x = col * step_x;
      rect.y = row * step_y;
      rect.width = tile_w_;
      rect.height = tile_h_;
      src_rects_.push_back(rect);
    }
  }

  if (!src_rects_.empty()) {
    const im_rect &last = src_rects_.back();
    int max_x = last.x + last.width;
    int max_y = last.y + last.height;
    if (max_x > src_w_ || max_y > src_h_) {
      RGA_ERR("Tile grid exceeds source image! "
              << "last_tile=(" << last.x << "," << last.y << "," << last.width
              << "," << last.height << ")"
              << " src=" << src_w_ << "x" << src_h_);
    }
  }

  tile_size_ = static_cast<size_t>(tile_w_) * tile_h_ * 3;

  RGA_LOG("Config: src=" << src_w_ << "x" << src_h_ << " tile=" << tile_w_
                         << "x" << tile_h_ << " grid=" << tile_cols_ << "x"
                         << tile_rows_ << " overlap=" << overlap_w_ << "x"
                         << overlap_h_ << " tiles=" << tile_count()
                         << " tile_size=" << tile_size_);
}

RgaCropper::~RgaCropper() {
  release_tile_dma_bufs();
  release_dma_buf();
}

bool RgaCropper::alloc_dma_buf(size_t size) {
  // 旧接口兼容：不再主动分配大 buffer，若需要可在此扩展
  (void)size;
  return true;
}

bool RgaCropper::alloc_tile_dma_bufs() {
  // 优先物理连续，失败则回退到普通页内存（Rockchip 上 system 完全可用）
  const char *heap_paths[] = {
      "/dev/dma_heap/linux,cma",       "/dev/dma_heap/cma",
      "/dev/dma_heap/system-uncached", "/dev/dma_heap/linux,system-uncached",
      "/dev/dma_heap/system",          nullptr};

  for (int i = 0; heap_paths[i] != nullptr; ++i) {
    int heap_fd = open(heap_paths[i], O_RDWR | O_CLOEXEC);
    if (heap_fd < 0)
      continue;

    std::vector<int> tmp_fds;
    bool ok = true;

    for (int t = 0; t < tile_count(); ++t) {
      struct dma_heap_allocation_data alloc_data = {};
      alloc_data.len = tile_size_;
      alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
      alloc_data.heap_flags = 0;

      if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
        RGA_LOG("heap " << heap_paths[i] << " tile " << t << " alloc failed ("
                        << strerror(errno) << "), trying next...");
        ok = false;
        break;
      }
      tmp_fds.push_back(alloc_data.fd);
    }

    close(heap_fd);

    if (ok) {
      tile_fds_ = std::move(tmp_fds);
      RGA_LOG("Using dma-heap: " << heap_paths[i] << ", allocated "
                                 << tile_count()
                                 << " tiles, size=" << tile_size_);
      return true;
    }

    // 清理本次部分分配的 fd，避免泄漏
    for (int fd : tmp_fds) {
      if (fd >= 0)
        close(fd);
    }
  }

  RGA_ERR("Failed to allocate tile dma-bufs from any heap");
  return false;
}
void RgaCropper::release_dma_buf() {
  if (dst_fd_ >= 0) {
    close(dst_fd_);
    dst_fd_ = -1;
    dst_size_ = 0;
  }
}

void RgaCropper::release_tile_dma_bufs() {
  for (int fd : tile_fds_) {
    if (fd >= 0)
      close(fd);
  }
  tile_fds_.clear();
}

bool RgaCropper::init() { return alloc_tile_dma_bufs(); }

bool RgaCropper::process(int src_fd, int src_fmt, int src_w, int src_h) {
  if (tile_fds_.size() != static_cast<size_t>(tile_count())) {
    RGA_ERR("tile_fds not ready, call init() first");
    return false;
  }

  rga_buffer_t src = wrapbuffer_fd(src_fd, src_w, src_h, src_fmt, src_w, src_h);
  src.vir_addr = nullptr;
  src.phy_addr = nullptr;

  rga_buffer_t pat = {};
  memset(&pat, 0, sizeof(pat));

  for (size_t i = 0; i < src_rects_.size(); ++i) {
    // 每个 tile 写入自己独立的 dmabuf，offset 永远为 0
    rga_buffer_t dst = wrapbuffer_fd(tile_fds_[i], tile_w_, tile_h_,
                                     RK_FORMAT_RGB_888, tile_w_, tile_h_);
    dst.vir_addr = nullptr;
    dst.phy_addr = nullptr;

    im_rect dst_rect = {0, 0, tile_w_, tile_h_};

    IM_STATUS ret =
        ::improcess(src, dst, pat, src_rects_[i], dst_rect, {}, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS) {
      RGA_ERR("improcess task " << i << " failed: " << imStrError(ret));
      return false;
    }
  }

  return true;
}

int RgaCropper::dst_fd() const {
  return dst_fd_; // 保持兼容，返回 -1 表示单一大 buffer 不再使用
}

size_t RgaCropper::dst_size() const {
  return dst_size_; // 保持兼容，返回 0
}

int RgaCropper::tile_count() const { return tile_cols_ * tile_rows_; }

int RgaCropper::tile_fd(int idx) const {
  if (idx < 0 || idx >= static_cast<int>(tile_fds_.size()))
    return -1;
  return tile_fds_[idx];
}

size_t RgaCropper::tile_size() const { return tile_size_; }

bool RgaCropper::save_bmp(const std::string &path, const uint8_t *rgb_data,
                          int w, int h) {
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

  uint8_t fh[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
  fh[2] = file_size & 0xFF;
  fh[3] = (file_size >> 8) & 0xFF;
  fh[4] = (file_size >> 16) & 0xFF;
  fh[5] = (file_size >> 24) & 0xFF;

  uint8_t ih[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  ih[4] = w & 0xFF;
  ih[5] = (w >> 8) & 0xFF;
  ih[6] = (w >> 16) & 0xFF;
  ih[7] = (w >> 24) & 0xFF;
  ih[8] = h & 0xFF;
  ih[9] = (h >> 8) & 0xFF;
  ih[10] = (h >> 16) & 0xFF;
  ih[11] = (h >> 24) & 0xFF;

  ofs.write((char *)fh, 14);
  ofs.write((char *)ih, 40);

  std::vector<uint8_t> row(bmp_row_stride, 0);
  for (int y = h - 1; y >= 0; --y) {
    const uint8_t *src = rgb_data + y * w * 3;
    for (int x = 0; x < w; ++x) {
      row[x * 3 + 0] = src[x * 3 + 2];
      row[x * 3 + 1] = src[x * 3 + 1];
      row[x * 3 + 2] = src[x * 3 + 0];
    }
    if (padding)
      memset(row.data() + row_stride, 0, padding);
    ofs.write((char *)row.data(), bmp_row_stride);
  }

  ofs.close();
  return true;
}

// 兼容旧接口：要求传入有效 fd，否则失败
bool RgaCropper::save_tiles(int dst_fd, size_t dst_size, int tile_w, int tile_h,
                            int tile_count, const std::string &prefix) {
  if (dst_fd < 0) {
    RGA_ERR("save_tiles with single fd is deprecated in zero-copy mode, "
            "use save_tiles(prefix) instead");
    return false;
  }

  void *ptr = mmap(nullptr, dst_size, PROT_READ, MAP_SHARED, dst_fd, 0);
  if (ptr == MAP_FAILED) {
    RGA_ERR("mmap dst_fd failed: " << strerror(errno));
    return false;
  }

  uint8_t *data = (uint8_t *)ptr;
  size_t single_tile_size = static_cast<size_t>(tile_w) * tile_h * 3;
  bool ok = true;

  for (int i = 0; i < tile_count; ++i) {
    std::string path = prefix + "_" + std::to_string(i) + ".bmp";
    if (!save_bmp(path, data + i * single_tile_size, tile_w, tile_h)) {
      RGA_ERR("Failed to save " << path);
      ok = false;
      break;
    }
    RGA_LOG("Saved " << path);
  }

  munmap(ptr, dst_size);
  return ok;
}

// 新增：利用内部独立 tile fds 保存
bool RgaCropper::save_tiles(const std::string &prefix) const {
  if (tile_fds_.size() != static_cast<size_t>(tile_count())) {
    RGA_ERR("tile_fds not ready");
    return false;
  }

  bool ok = true;
  for (int i = 0; i < tile_count(); ++i) {
    void *ptr =
        mmap(nullptr, tile_size_, PROT_READ, MAP_SHARED, tile_fds_[i], 0);
    if (ptr == MAP_FAILED) {
      RGA_ERR("mmap tile " << i << " failed: " << strerror(errno));
      ok = false;
      break;
    }

    std::string path = prefix + "_" + std::to_string(i) + ".bmp";
    if (!save_bmp(path, static_cast<uint8_t *>(ptr), tile_w_, tile_h_)) {
      RGA_ERR("Failed to save " << path);
      munmap(ptr, tile_size_);
      ok = false;
      break;
    }
    RGA_LOG("Saved " << path);
    munmap(ptr, tile_size_);
  }
  return ok;
}
