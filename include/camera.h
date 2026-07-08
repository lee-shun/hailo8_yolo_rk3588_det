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
    int src_stride() const;
    int src_w() const;
    int src_h() const;
    int src_fmt() const;

private:
    int fd_ = -1;
    int width_ = 0, height_ = 0;
    CameraFormat fmt_ = CameraFormat::YUYV;
    bool streaming_ = false;

    struct V4L2Buf { void* start; size_t len; };
    std::vector<V4L2Buf> v4l2_bufs_;
    int v4l2_buf_idx_ = -1;

    MppCtx mpp_ctx_ = nullptr;
    MppApi* mpp_api_ = nullptr;
    MppFrame mpp_frame_ = nullptr;   // decoder 返回的 frame 指针（advanced 下等于 out_frame_）

    // MJPEG advanced 解码预分配资源（严格对应 mpi_dec_test 的 dec_advanced）
    MppFrame out_frame_ = nullptr;   // 预分配的输出 frame
    MppBuffer frm_buf_ = nullptr;    // 预分配的输出 buffer
    MppBufferGroup frm_grp_ = nullptr;
    MppBufferGroup pkt_grp_ = nullptr;

    bool init_v4l2(int fps);
    bool init_mpp();
};

#endif
