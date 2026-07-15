#ifndef CAM_DMA_MMAP_H
#define CAM_DMA_MMAP_H

#include <linux/videodev2.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_mpi.h>

#include <chrono>
#include <string>
#include <vector>

class CamDmaMmap {
public:
  enum class Mode { MMAP, DMA };

  enum class Format { MJPEG, YUYV };

  explicit CamDmaMmap(Mode mode);
  ~CamDmaMmap();

  bool init(const std::string &dev, int width, int height, int fps, Format fmt);
  bool start();
  void stop();

  bool grab();
  void release();

  // MJPEG: 无论 DMA/MMAP，均返回 MPP buffer 的 CPU 映射基地址
  // YUYV+MMAP: 返回 V4L2 mmap 地址
  // YUYV+DMA: 返回 nullptr（需用 src_fd）
  uint8_t *src_ptr() const;
  int src_fd() const;

  int src_stride() const; // 水平 stride（YUYV 为 width*2，MJPEG 为 MPP 对齐后）
  int src_ver_stride() const; // 垂直 stride（仅 MJPEG 有效，YUYV 返回 height）
  int src_w() const;
  int src_h() const;
  int src_fmt() const; // RK_FORMAT_YCbCr_420_SP 或 RK_FORMAT_YUYV_422

  int total_frames() const { return stats_total_; }
  int ok_frames() const { return stats_ok_; }
  int err_frames() const { return stats_err_; }

private:
  struct V4L2Buf {
    void *start = nullptr; // MMAP 模式使用
    size_t len = 0;
    int fd = -1;                 // DMA 模式使用
    MppBuffer mpp_buf = nullptr; // DMA 模式：持有 MPP buffer 引用
  };

  bool init_v4l2(int fps);
  bool init_v4l2_mmap();
  bool init_v4l2_dma();
  bool init_mpp();

  bool process_mjpeg(const struct v4l2_buffer &buf);
  bool process_mjpeg_dma(const struct v4l2_buffer &buf);
  bool process_mjpeg_mmap(const struct v4l2_buffer &buf);
  bool process_yuyv(const struct v4l2_buffer &buf);

  bool check_jpeg_header(const uint8_t *data, size_t len);
  void cleanup();
  void log_stats();

  Mode mode_;
  Format fmt_;
  int fd_ = -1;
  int width_ = 0, height_ = 0;
  int hor_stride_ = 0, ver_stride_ = 0;
  bool streaming_ = false;

  std::vector<V4L2Buf> v4l2_bufs_;
  int v4l2_buf_idx_ = -1;

  MppCtx mpp_ctx_ = nullptr;
  MppApi *mpp_api_ = nullptr;
  MppFrame mpp_frame_ = nullptr;
  MppFrame out_frame_ = nullptr;
  MppBuffer frm_buf_ = nullptr;
  MppBufferGroup frm_grp_ = nullptr;
  MppBufferGroup pkt_grp_ = nullptr;
  MppBufferGroup v4l2_dma_group_ = nullptr;

  int stats_total_ = 0;
  int stats_ok_ = 0;
  int stats_err_ = 0;
  std::chrono::steady_clock::time_point stats_start_;
};

#endif // CAM_DMA_MMAP_H
