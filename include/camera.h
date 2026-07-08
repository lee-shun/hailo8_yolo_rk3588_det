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

// V4L2 取流 + MPP 硬件解码（MJPEG 模式）
// 统一输出：源数据指针、stride、格式，供 RGA 直接导入
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
    MppFrame mpp_frame_ = nullptr;

    bool init_v4l2(int fps);
    bool init_mpp();
};

#endif
