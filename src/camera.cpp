#include "camera.h"
#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <rga.h>

Camera::Camera() = default;
Camera::~Camera() { stop(); }

bool Camera::init(const std::string& dev, int width, int height, int fps, CameraFormat fmt) {
    width_ = width;
    height_ = height;
    fmt_ = fmt;

    fd_ = open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
        std::cerr << "[Camera] open " << dev << " failed\n";
        return false;
    }

    if (!init_v4l2(fps)) return false;
    if (fmt_ == CameraFormat::MJPEG && !init_mpp()) return false;
    return true;
}

bool Camera::init_v4l2(int fps) {
    struct v4l2_format fmt {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (fmt_ == CameraFormat::MJPEG) {
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    } else {
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    }

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        std::cerr << "[Camera] VIDIOC_S_FMT failed\n";
        return false;
    }

    struct v4l2_streamparm parm {};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    ioctl(fd_, VIDIOC_S_PARM, &parm);

    struct v4l2_requestbuffers req {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        std::cerr << "[Camera] VIDIOC_REQBUFS failed\n";
        return false;
    }

    v4l2_bufs_.resize(req.count);
    for (size_t i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        ioctl(fd_, VIDIOC_QUERYBUF, &buf);
        v4l2_bufs_[i].len = buf.length;
        v4l2_bufs_[i].start = mmap(nullptr, buf.length, PROT_READ, MAP_SHARED, fd_, buf.m.offset);
        ioctl(fd_, VIDIOC_QBUF, &buf);
    }
    return true;
}

bool Camera::init_mpp() {
    if (mpp_create(&mpp_ctx_, &mpp_api_) != MPP_OK) return false;
    if (mpp_init(mpp_ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG) != MPP_OK) return false;

    MppDecCfg cfg = nullptr;
    mpp_dec_cfg_init(&cfg);
    mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);
    mpp_dec_cfg_set_u32(cfg, "base:output_format", MPP_FMT_YUV420SP);
    mpp_api_->control(mpp_ctx_, MPP_DEC_SET_CFG, cfg);
    mpp_dec_cfg_deinit(cfg);
    return true;
}

bool Camera::start() {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) return false;
    streaming_ = true;
    return true;
}

void Camera::stop() {
    if (!streaming_) return;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
    streaming_ = false;

    release();

    for (auto& b : v4l2_bufs_) {
        if (b.start && b.start != MAP_FAILED) munmap(b.start, b.len);
    }
    v4l2_bufs_.clear();
    close(fd_);
    fd_ = -1;

    if (mpp_ctx_) {
        mpp_api_->reset(mpp_ctx_);
        mpp_destroy(mpp_ctx_);
        mpp_ctx_ = nullptr;
    }
}

bool Camera::grab() {
    release();

    struct v4l2_buffer buf {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    while (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) {
            usleep(1000);
            continue;
        }
        return false;
    }

    v4l2_buf_idx_ = buf.index;

    if (fmt_ == CameraFormat::MJPEG) {
        MppPacket packet = nullptr;
        mpp_packet_init(&packet, v4l2_bufs_[buf.index].start, buf.bytesused);
        mpp_packet_set_pts(packet, buf.timestamp.tv_sec * 1000000LL + buf.timestamp.tv_usec);

        mpp_api_->decode_put_packet(mpp_ctx_, packet);
        mpp_packet_deinit(&packet);
        ioctl(fd_, VIDIOC_QBUF, &buf);

        if (mpp_api_->decode_get_frame(mpp_ctx_, &mpp_frame_) != MPP_OK || !mpp_frame_)
            return false;

        if (mpp_frame_get_info_change(mpp_frame_)) {
            mpp_frame_deinit(&mpp_frame_);
            mpp_frame_ = nullptr;
            return false;
        }
    } else {
        ioctl(fd_, VIDIOC_QBUF, &buf);
    }
    return true;
}

void Camera::release() {
    if (v4l2_buf_idx_ >= 0) {
        struct v4l2_buffer buf {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = v4l2_buf_idx_;
        ioctl(fd_, VIDIOC_QBUF, &buf);
        v4l2_buf_idx_ = -1;
    }

    if (mpp_frame_) {
        mpp_frame_deinit(&mpp_frame_);
        mpp_frame_ = nullptr;
    }
}

uint8_t* Camera::src_ptr() const {
    if (fmt_ == CameraFormat::MJPEG) {
        if (!mpp_frame_) return nullptr;
        return (uint8_t*)mpp_buffer_get_ptr(mpp_frame_get_buffer(mpp_frame_));
    } else {
        if (v4l2_buf_idx_ < 0) return nullptr;
        return (uint8_t*)v4l2_bufs_[v4l2_buf_idx_].start;
    }
}

int Camera::src_stride() const {
    if (fmt_ == CameraFormat::MJPEG) {
        return mpp_frame_ ? mpp_frame_get_hor_stride(mpp_frame_) : 0;
    } else {
        return width_ * 2;
    }
}

int Camera::src_w() const { return width_; }
int Camera::src_h() const { return height_; }

int Camera::src_fmt() const {
    if (fmt_ == CameraFormat::MJPEG)
        return RK_FORMAT_YCbCr_420_SP;
    else
        return RK_FORMAT_YUYV_422;
}
