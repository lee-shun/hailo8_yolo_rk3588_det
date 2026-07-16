#include "camera.h"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <rga.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define CAM_LOG(msg) std::cout << "[Camera] " << msg << "\n"
#define CAM_ERR(msg) std::cerr << "[Camera] ERROR: " << msg << "\n"

#ifndef MPP_ALIGN
#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif

Camera::Camera() = default;
Camera::~Camera() { stop(); }

bool Camera::init(const std::string &dev, int width, int height, int fps,
                  CameraFormat fmt) {
  width_ = width;
  height_ = height;
  fmt_ = fmt;

  CAM_LOG("Opening " << dev << " at " << width << "x" << height
                     << " fps=" << fps << " fmt="
                     << (fmt == CameraFormat::MJPEG ? "MJPEG" : "YUYV"));

  fd_ = open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (fd_ < 0) {
    CAM_ERR("open failed, errno=" << errno);
    return false;
  }

  if (!init_v4l2(fps))
    return false;
  if (fmt_ == CameraFormat::MJPEG && !init_mpp())
    return false;

  CAM_LOG("Init OK");
  return true;
}

bool Camera::init_v4l2(int fps) {
  struct v4l2_format fmt{};
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
    CAM_ERR("VIDIOC_S_FMT failed, errno=" << errno);
    return false;
  }

  struct v4l2_format fmt_check{};
  fmt_check.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ioctl(fd_, VIDIOC_G_FMT, &fmt_check);
  CAM_LOG("V4L2 actual: " << fmt_check.fmt.pix.width << "x"
                          << fmt_check.fmt.pix.height << " fourcc="
                          << std::string((char *)&fmt_check.fmt.pix.pixelformat,
                                         4));

  struct v4l2_streamparm parm{};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator = fps;
  ioctl(fd_, VIDIOC_S_PARM, &parm);

  struct v4l2_requestbuffers req{};
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
    CAM_ERR("VIDIOC_REQBUFS failed, errno=" << errno);
    return false;
  }
  CAM_LOG("V4L2 got " << req.count << " buffers");

  v4l2_bufs_.resize(req.count);
  for (size_t i = 0; i < req.count; ++i) {
    struct v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
      CAM_ERR("VIDIOC_QUERYBUF failed on buf " << i);
      return false;
    }
    v4l2_bufs_[i].len = buf.length;
    v4l2_bufs_[i].start =
        mmap(nullptr, buf.length, PROT_READ, MAP_SHARED, fd_, buf.m.offset);
    if (v4l2_bufs_[i].start == MAP_FAILED) {
      CAM_ERR("mmap failed on buf " << i);
      return false;
    }

    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
      CAM_ERR("VIDIOC_QBUF failed on buf " << i);
      return false;
    }
  }
  return true;
}

bool Camera::init_mpp() {
  CAM_LOG("MPP init start...");

  hor_stride_ = MPP_ALIGN(width_, 16);
  ver_stride_ = MPP_ALIGN(height_, 16);
  RK_U32 buf_size = (RK_U32)hor_stride_ * ver_stride_ * 4;

  CAM_LOG("Prealloc output: " << width_ << "x" << height_
                              << " stride=" << hor_stride_ << "x" << ver_stride_
                              << " buf_size=" << buf_size);

  MppDecCfg cfg = nullptr;
  MppBufferType buf_type = MPP_BUFFER_TYPE_ION;
  bool ok = false;

  do {
    if (mpp_frame_init(&out_frame_) != MPP_OK) {
      CAM_ERR("mpp_frame_init failed");
      break;
    }

    mpp_frame_set_width(out_frame_, width_);
    mpp_frame_set_height(out_frame_, height_);
    mpp_frame_set_hor_stride(out_frame_, hor_stride_);
    mpp_frame_set_ver_stride(out_frame_, ver_stride_);
    mpp_frame_set_fmt(out_frame_, MPP_FMT_YUV420SP);

    if (mpp_buffer_group_get(&frm_grp_, buf_type, MPP_BUFFER_INTERNAL,
                             "cam_frm", __FUNCTION__) != MPP_OK) {
      buf_type = MPP_BUFFER_TYPE_NORMAL;
      if (mpp_buffer_group_get(&frm_grp_, buf_type, MPP_BUFFER_INTERNAL,
                               "cam_frm", __FUNCTION__) != MPP_OK) {
        CAM_ERR("mpp_buffer_group_get(frm, INTERNAL) failed");
        break;
      }
    }
    CAM_LOG("Buffer group type=" << (buf_type == MPP_BUFFER_TYPE_ION ? "ION"
                                                                       : "NORMAL"));

    if (mpp_buffer_group_limit_config(frm_grp_, buf_size, 4) != MPP_OK) {
      CAM_ERR("mpp_buffer_group_limit_config failed");
      break;
    }

    if (mpp_buffer_get(frm_grp_, &frm_buf_, buf_size) != MPP_OK) {
      CAM_ERR("mpp_buffer_get(frm_buf) failed");
      break;
    }
    CAM_LOG("frm_buf allocated: " << mpp_buffer_get_size(frm_buf_)
                                  << " ptr=" << mpp_buffer_get_ptr(frm_buf_));

    mpp_frame_set_buffer(out_frame_, frm_buf_);

    if (mpp_create(&mpp_ctx_, &mpp_api_) != MPP_OK) {
      CAM_ERR("mpp_create failed");
      break;
    }
    if (mpp_init(mpp_ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG) != MPP_OK) {
      CAM_ERR("mpp_init failed");
      break;
    }

    MppFrameFormat out_fmt = MPP_FMT_YUV420SP;
    if (mpp_api_->control(mpp_ctx_, MPP_DEC_SET_OUTPUT_FORMAT, &out_fmt) !=
        MPP_OK) {
      CAM_ERR("MPP_DEC_SET_OUTPUT_FORMAT failed");
      break;
    }

    mpp_dec_cfg_init(&cfg);
    if (mpp_api_->control(mpp_ctx_, MPP_DEC_GET_CFG, cfg) != MPP_OK) {
      CAM_ERR("MPP_DEC_GET_CFG failed");
      break;
    }
    if (mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1) != MPP_OK) {
      CAM_ERR("mpp_dec_cfg_set_u32 failed");
      break;
    }
    if (mpp_api_->control(mpp_ctx_, MPP_DEC_SET_CFG, cfg) != MPP_OK) {
      CAM_ERR("MPP_DEC_SET_CFG failed");
      break;
    }

    if (mpp_buffer_group_get(&pkt_grp_, MPP_BUFFER_TYPE_ION, MPP_BUFFER_INTERNAL,
                             "cam_pkt", __FUNCTION__) != MPP_OK) {
      if (mpp_buffer_group_get(&pkt_grp_, MPP_BUFFER_TYPE_NORMAL,
                               MPP_BUFFER_INTERNAL, "cam_pkt",
                               __FUNCTION__) != MPP_OK) {
        CAM_ERR("mpp_buffer_group_get(pkt, INTERNAL) failed");
        break;
      }
    }

    ok = true;
  } while (0);

  if (cfg) {
    mpp_dec_cfg_deinit(cfg);
    cfg = nullptr;
  }

  if (!ok) {
    if (mpp_ctx_) {
      mpp_destroy(mpp_ctx_);
      mpp_ctx_ = nullptr;
      mpp_api_ = nullptr;
    }
    if (frm_buf_) {
      mpp_buffer_put(frm_buf_);
      frm_buf_ = nullptr;
    }
    if (frm_grp_) {
      mpp_buffer_group_put(frm_grp_);
      frm_grp_ = nullptr;
    }
    if (out_frame_) {
      mpp_frame_deinit(&out_frame_);
      out_frame_ = nullptr;
    }
    return false;
  }

  CAM_LOG("MPP init OK");
  return true;
}

bool Camera::start() {
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
    CAM_ERR("VIDIOC_STREAMON failed, errno=" << errno);
    return false;
  }
  streaming_ = true;
  CAM_LOG("STREAMON OK");
  return true;
}

void Camera::stop() {
  if (!streaming_)
    return;
  CAM_LOG("Stopping...");

  release();

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ioctl(fd_, VIDIOC_STREAMOFF, &type);
  streaming_ = false;

  for (auto &b : v4l2_bufs_) {
    if (b.start && b.start != MAP_FAILED)
      munmap(b.start, b.len);
    if (b.fd >= 0)
      close(b.fd);
  }
  v4l2_bufs_.clear();
  close(fd_);
  fd_ = -1;

  if (mpp_ctx_) {
    mpp_api_->reset(mpp_ctx_);
    mpp_destroy(mpp_ctx_);
    mpp_ctx_ = nullptr;
    mpp_api_ = nullptr;
  }

  if (mpp_frame_ && mpp_frame_ != out_frame_) {
    mpp_frame_deinit(&mpp_frame_);
  }
  mpp_frame_ = nullptr;

  if (out_frame_) {
    mpp_frame_deinit(&out_frame_);
    out_frame_ = nullptr;
  }
  if (frm_buf_) {
    mpp_buffer_put(frm_buf_);
    frm_buf_ = nullptr;
  }
  if (frm_grp_) {
    mpp_buffer_group_put(frm_grp_);
    frm_grp_ = nullptr;
  }
  if (pkt_grp_) {
    mpp_buffer_group_put(pkt_grp_);
    pkt_grp_ = nullptr;
  }

  hor_stride_ = 0;
  ver_stride_ = 0;

  CAM_LOG("Stopped");
}

bool Camera::grab() {
  release();

  struct v4l2_buffer buf{};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;

  int retry = 0;
  while (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
    if (errno == EAGAIN) {
      usleep(1000);
      if (++retry > 5000) {
        CAM_ERR("VIDIOC_DQBUF timeout");
        return false;
      }
      continue;
    }
    CAM_ERR("VIDIOC_DQBUF failed, errno=" << errno);
    return false;
  }

  v4l2_buf_idx_ = buf.index;
  CAM_LOG("DQBUF idx=" << buf.index << " bytesused=" << buf.bytesused
                       << " seq=" << buf.sequence);

  if (buf.bytesused == 0) {
    CAM_ERR("V4L2 buffer empty");
    goto err_qbuf;
  }

  if (fmt_ == CameraFormat::MJPEG) {
    uint8_t *jpeg_ptr = (uint8_t *)v4l2_bufs_[buf.index].start;

    bool soi_ok =
        (buf.bytesused >= 2 && jpeg_ptr[0] == 0xFF && jpeg_ptr[1] == 0xD8);
    bool eoi_ok = (buf.bytesused >= 2 && jpeg_ptr[buf.bytesused - 2] == 0xFF &&
                   jpeg_ptr[buf.bytesused - 1] == 0xD9);
    CAM_LOG("JPEG SOI=" << (soi_ok ? "OK" : "FAIL")
                        << " EOI=" << (eoi_ok ? "OK" : "FAIL")
                        << " size=" << buf.bytesused);

    if (!soi_ok) {
      std::cout << "  Header: ";
      for (int i = 0; i < 8 && i < (int)buf.bytesused; i++)
        std::cout << std::hex << std::setw(2) << (int)jpeg_ptr[i] << " ";
      std::cout << std::dec << "\n";
      goto err_qbuf;
    }

    MppBuffer pkt_buf;
    MppPacket packet;
    MppFrame frame_ret;
    MppMeta meta;
    RK_S32 ret;
    int out_w;
    int out_h;
    int out_stride;

    pkt_buf = nullptr;
    packet = nullptr;
    frame_ret = nullptr;
    meta = nullptr;
    ret = MPP_OK;

    // 旧版 MPP 没有 mpp_buffer_import_with_fd，使用 memcpy 路径
    ret = mpp_buffer_get(pkt_grp_, &pkt_buf, buf.bytesused);
    if (ret != MPP_OK) {
      CAM_ERR("mpp_buffer_get(pkt_buf) failed, ret=" << ret);
      goto err_qbuf;
    }
    memcpy(mpp_buffer_get_ptr(pkt_buf), v4l2_bufs_[buf.index].start,
           buf.bytesused);

    ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
    if (ret != MPP_OK) {
      CAM_ERR("mpp_packet_init_with_buffer failed, ret=" << ret);
      mpp_buffer_put(pkt_buf);
      goto err_qbuf;
    }

    mpp_packet_set_size(packet, buf.bytesused);
    mpp_packet_set_length(packet, buf.bytesused);
    mpp_packet_set_pos(packet, mpp_packet_get_data(packet));

    meta = mpp_packet_get_meta(packet);
    if (meta) {
      mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, out_frame_);
      CAM_LOG("Bound KEY_OUTPUT_FRAME to packet meta");
    } else {
      CAM_ERR("mpp_packet_get_meta returned null");
    }

    ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);
    if (ret != MPP_OK) {
      CAM_ERR("decode_put_packet failed, ret=" << ret);
      goto done;
    }
    CAM_LOG("decode_put_packet OK");

    ret = mpp_api_->decode_get_frame(mpp_ctx_, &frame_ret);
    if (ret != MPP_OK || !frame_ret) {
      CAM_ERR("decode_get_frame failed, ret=" << ret << " frame=" << frame_ret);
      goto done;
    }

    if (frame_ret != out_frame_) {
      CAM_ERR("frame mismatch: expected " << out_frame_ << " got "
                                          << frame_ret);
    } else {
      CAM_LOG("frame match OK");
    }

    out_w = mpp_frame_get_width(frame_ret);
    out_h = mpp_frame_get_height(frame_ret);
    out_stride = mpp_frame_get_hor_stride(frame_ret);
    CAM_LOG("Decoded: " << out_w << "x" << out_h << " stride=" << out_stride);

    mpp_frame_ = frame_ret;

    meta = mpp_frame_get_meta(frame_ret);
    if (meta) {
      MppPacket packet_ret;
      RK_S32 meta_ret;
      packet_ret = nullptr;
      meta_ret = mpp_meta_get_packet(meta, KEY_INPUT_PACKET, &packet_ret);
      if (meta_ret != MPP_OK || !packet_ret) {
        CAM_ERR("mpp_meta_get_packet failed, ret=" << meta_ret);
      } else if (packet_ret != packet) {
        CAM_ERR("packet mismatch: expected " << packet << " got "
                                             << packet_ret);
      } else {
        CAM_LOG("packet match OK");
      }
    }

    if (mpp_frame_get_eos(frame_ret)) {
      CAM_LOG("found eos frame");
    }

  done:
    if (packet)
      mpp_packet_deinit(&packet);
    if (pkt_buf)
      mpp_buffer_put(pkt_buf);

    if (ret != MPP_OK || !mpp_frame_) {
      mpp_frame_ = nullptr;
      goto err_qbuf;
    }

    struct v4l2_buffer qbuf{};
    qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    qbuf.memory = V4L2_MEMORY_MMAP;
    qbuf.index = v4l2_buf_idx_;
    ioctl(fd_, VIDIOC_QBUF, &qbuf);
    v4l2_buf_idx_ = -1;
  }

  return true;

err_qbuf: {
  struct v4l2_buffer qbuf{};
  qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  qbuf.memory = V4L2_MEMORY_MMAP;
  qbuf.index = v4l2_buf_idx_;
  ioctl(fd_, VIDIOC_QBUF, &qbuf);
  v4l2_buf_idx_ = -1;
}
  return false;
}

void Camera::release() {
  if (v4l2_buf_idx_ >= 0) {
    struct v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = v4l2_buf_idx_;
    ioctl(fd_, VIDIOC_QBUF, &buf);
    v4l2_buf_idx_ = -1;
  }

  if (mpp_frame_ && mpp_frame_ != out_frame_) {
    mpp_frame_deinit(&mpp_frame_);
  }
  mpp_frame_ = nullptr;
}

uint8_t *Camera::src_ptr() const {
  if (fmt_ == CameraFormat::MJPEG) {
    if (!mpp_frame_)
      return nullptr;
    MppBuffer buffer = mpp_frame_get_buffer(mpp_frame_);
    if (!buffer)
      return nullptr;
    return (uint8_t *)mpp_buffer_get_ptr(buffer);
  } else {
    if (v4l2_buf_idx_ < 0)
      return nullptr;
    return (uint8_t *)v4l2_bufs_[v4l2_buf_idx_].start;
  }
}

int Camera::src_stride() const {
  return hor_stride_;
}

int Camera::src_ver_stride() const {
  return ver_stride_;
}

int Camera::src_w() const { return width_; }
int Camera::src_h() const { return height_; }

int Camera::src_fmt() const {
  if (fmt_ == CameraFormat::MJPEG)
    return RK_FORMAT_YCbCr_420_SP;
  else
    return RK_FORMAT_YUYV_422;
}

int Camera::src_fd() const {
  if (fmt_ == CameraFormat::MJPEG && frm_buf_) {
    return mpp_buffer_get_fd(frm_buf_);
  }
  return -1;
}
