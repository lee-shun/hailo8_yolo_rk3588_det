#ifndef CAMERA_H
#define CAMERA_H

#include <linux/videodev2.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_mpi.h>

#include <string>
#include <vector>

enum class CameraFormat {
    MJPEG,
    YUYV
};

class Camera {
public:
    Camera();
    ~Camera();

    bool init(const std::string& dev, int width, int height, int fps, CameraFormat fmt);
    bool start();
    void stop();

    bool grab();
    void release();

    uint8_t* src_ptr() const;
    int src_stride() const;      // hor_stride（预计算，可靠）
    int src_ver_stride() const;  // ver_stride（预计算，可靠）
    int src_w() const;
    int src_h() const;
    int src_fmt() const;

    int src_fd() const;

private:
    int fd_ = -1;
    int width_ = 0, height_ = 0;
    int hor_stride_ = 0, ver_stride_ = 0;  // MPP 对齐后的 stride（新增）
    CameraFormat fmt_ = CameraFormat::YUYV;
    bool streaming_ = false;

    struct V4L2Buf { void* start; size_t len; int fd; };
    std::vector<V4L2Buf> v4l2_bufs_;
    int v4l2_buf_idx_ = -1;

    MppCtx mpp_ctx_ = nullptr;
    MppApi* mpp_api_ = nullptr;
    MppFrame mpp_frame_ = nullptr;

    MppFrame out_frame_ = nullptr;
    MppBuffer frm_buf_ = nullptr;
    MppBufferGroup frm_grp_ = nullptr;
    MppBufferGroup pkt_grp_ = nullptr;
    MppBufferGroup pkt_grp_ext_ = nullptr;

    bool init_v4l2(int fps);
    bool init_mpp();
};

#endif
