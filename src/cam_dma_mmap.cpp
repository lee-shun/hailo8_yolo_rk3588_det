#include "cam_dma_mmap.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <rga.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define MODULE_TAG "CamDmaMmap"

#ifndef MPP_ALIGN
#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif

// ---------------------------------------------------------------------------
// 分级日志宏：自动带上 [DMA/MMAP][MJPEG/YUYV] 前缀
// ---------------------------------------------------------------------------
#define CAM_LOG(msg) \
    std::cout << "[CamDmaMmap][" \
              << (mode_ == Mode::DMA ? "DMA" : "MMAP") \
              << "][" << (fmt_ == Format::MJPEG ? "MJPEG" : "YUYV") \
              << "] " << msg << "\n"

#define CAM_ERR(msg) \
    std::cerr << "[CamDmaMmap][" \
              << (mode_ == Mode::DMA ? "DMA" : "MMAP") \
              << "][" << (fmt_ == Format::MJPEG ? "MJPEG" : "YUYV") \
              << "] ERROR: " << msg << "\n"

#define CAM_DBG(msg) \
    do { \
        std::cout << "[CamDmaMmap][" \
                  << (mode_ == Mode::DMA ? "DMA" : "MMAP") \
                  << "][" << (fmt_ == Format::MJPEG ? "MJPEG" : "YUYV") \
                  << "] DEBUG: " << msg << "\n"; \
    } while (0)

// ---------------------------------------------------------------------------
// 构造 / 析构
// ---------------------------------------------------------------------------
CamDmaMmap::CamDmaMmap(Mode mode)
    : mode_(mode)
    , stats_start_(std::chrono::steady_clock::now())
{}

CamDmaMmap::~CamDmaMmap() { stop(); }

// ---------------------------------------------------------------------------
// 初始化总入口
// ---------------------------------------------------------------------------
bool CamDmaMmap::init(const std::string& dev, int width, int height, int fps, Format fmt)
{
    width_  = width;
    height_ = height;
    fmt_    = fmt;

    CAM_LOG("Opening " << dev << " at " << width << "x" << height
                       << " fps=" << fps
                       << " mode=" << (mode_ == Mode::DMA ? "DMA" : "MMAP"));

    fd_ = open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
        CAM_ERR("open failed: " << strerror(errno));
        return false;
    }

    if (!init_v4l2(fps)) {
        stop();
        return false;
    }

    if (fmt_ == Format::MJPEG && !init_mpp()) {
        stop();
        return false;
    }

    CAM_LOG("Init OK");
    return true;
}

// ---------------------------------------------------------------------------
// V4L2 初始化
// ---------------------------------------------------------------------------
bool CamDmaMmap::init_v4l2(int fps)
{
    struct v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width  = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.field  = V4L2_FIELD_ANY;
    fmt.fmt.pix.pixelformat = (fmt_ == Format::MJPEG) ? V4L2_PIX_FMT_MJPEG
                                                      : V4L2_PIX_FMT_YUYV;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        CAM_ERR("VIDIOC_S_FMT failed: " << strerror(errno));
        return false;
    }

    struct v4l2_format fmt_check{};
    fmt_check.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_G_FMT, &fmt_check);
    CAM_LOG("V4L2 actual: " << fmt_check.fmt.pix.width << "x"
                            << fmt_check.fmt.pix.height
                            << " fourcc=" << std::string(
                                   (char*)&fmt_check.fmt.pix.pixelformat, 4));

    struct v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    ioctl(fd_, VIDIOC_S_PARM, &parm);

    struct v4l2_requestbuffers req{};
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = (mode_ == Mode::DMA) ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        CAM_ERR("VIDIOC_REQBUFS failed: " << strerror(errno));
        return false;
    }
    CAM_LOG("V4L2 got " << req.count << " buffers ("
            << (mode_ == Mode::DMA ? "DMABUF" : "MMAP") << ")");

    v4l2_bufs_.resize(req.count);

    if (mode_ == Mode::DMA)
        return init_v4l2_dma();
    else
        return init_v4l2_mmap();
}

// ---------------------------------------------------------------------------
// V4L2 MMAP 模式初始化
// ---------------------------------------------------------------------------
bool CamDmaMmap::init_v4l2_mmap()
{
    for (size_t i = 0; i < v4l2_bufs_.size(); ++i) {
        struct v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            CAM_ERR("VIDIOC_QUERYBUF failed on buf " << i << ": " << strerror(errno));
            return false;
        }

        v4l2_bufs_[i].len   = buf.length;
        v4l2_bufs_[i].start = mmap(nullptr, buf.length, PROT_READ,
                                   MAP_SHARED, fd_, buf.m.offset);
        if (v4l2_bufs_[i].start == MAP_FAILED) {
            CAM_ERR("mmap failed on buf " << i);
            return false;
        }
        v4l2_bufs_[i].fd      = -1;
        v4l2_bufs_[i].mpp_buf = nullptr;

        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            CAM_ERR("VIDIOC_QBUF failed on buf " << i << ": " << strerror(errno));
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// V4L2 DMA 模式初始化：预先分配 dma-buf，通过 DMABUF 交给 V4L2
// ---------------------------------------------------------------------------
bool CamDmaMmap::init_v4l2_dma()
{
    struct v4l2_format gfmt{};
    gfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_G_FMT, &gfmt) < 0) {
        CAM_ERR("VIDIOC_G_FMT failed: " << strerror(errno));
        return false;
    }

    size_t buf_size = gfmt.fmt.pix.sizeimage;
    if (buf_size == 0) {
        buf_size = (fmt_ == Format::MJPEG) ? (width_ * height_ * 2)
                                           : (width_ * height_ * 2); // YUYV
    }
    CAM_LOG("V4L2 DMA buffer size=" << buf_size);

    MppBufferType buf_type = MPP_BUFFER_TYPE_ION;
    if (mpp_buffer_group_get(&v4l2_dma_group_, buf_type, MPP_BUFFER_INTERNAL,
                             "v4l2_dma", __FUNCTION__) != MPP_OK) {
        buf_type = MPP_BUFFER_TYPE_DMA_HEAP;
        if (mpp_buffer_group_get(&v4l2_dma_group_, buf_type, MPP_BUFFER_INTERNAL,
                                 "v4l2_dma", __FUNCTION__) != MPP_OK) {
            buf_type = MPP_BUFFER_TYPE_DRM;
            if (mpp_buffer_group_get(&v4l2_dma_group_, buf_type, MPP_BUFFER_INTERNAL,
                                     "v4l2_dma", __FUNCTION__) != MPP_OK) {
                CAM_ERR("Failed to create V4L2 DMA buffer group");
                return false;
            }
        }
    }
    CAM_LOG("V4L2 DMA buffer group type=" << buf_type);

    if (mpp_buffer_group_limit_config(v4l2_dma_group_, buf_size,
                                      (RK_S32)v4l2_bufs_.size()) != MPP_OK) {
        CAM_ERR("mpp_buffer_group_limit_config failed for v4l2_dma");
        return false;
    }

    for (size_t i = 0; i < v4l2_bufs_.size(); ++i) {
        MppBuffer buf = nullptr;
        if (mpp_buffer_get(v4l2_dma_group_, &buf, buf_size) != MPP_OK) {
            CAM_ERR("mpp_buffer_get failed for V4L2 DMA buf " << i);
            return false;
        }

        v4l2_bufs_[i].mpp_buf = buf;
        v4l2_bufs_[i].fd      = mpp_buffer_get_fd(buf);
        v4l2_bufs_[i].len    = buf_size;
        v4l2_bufs_[i].start  = nullptr;

        CAM_LOG("V4L2 DMA buf[" << i << "] fd=" << v4l2_bufs_[i].fd
                << " size=" << buf_size);

        struct v4l2_buffer vbuf{};
        vbuf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vbuf.memory = V4L2_MEMORY_DMABUF;
        vbuf.index  = i;
        vbuf.m.fd   = v4l2_bufs_[i].fd;
        vbuf.length = (uint32_t)buf_size;

        if (ioctl(fd_, VIDIOC_QBUF, &vbuf) < 0) {
            CAM_ERR("VIDIOC_QBUF failed on DMA buf " << i << ": " << strerror(errno));
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// MPP 解码器初始化（仅 MJPEG）
// ---------------------------------------------------------------------------
bool CamDmaMmap::init_mpp()
{
    CAM_LOG("MPP init start...");

    hor_stride_ = MPP_ALIGN(width_, 16);
    ver_stride_ = MPP_ALIGN(height_, 16);
    RK_U32 buf_size = (RK_U32)hor_stride_ * ver_stride_ * 4;

    CAM_LOG("Prealloc output: " << width_ << "x" << height_
            << " stride=" << hor_stride_ << "x" << ver_stride_
            << " buf_size=" << buf_size);

    MppDecCfg cfg = nullptr;
    bool ok = false;

    MppBufferType frm_type = MPP_BUFFER_TYPE_ION;

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

        // 统一尝试 ION → DMA_HEAP → DRM，不再区分 MMAP/DMA
        if (mpp_buffer_group_get(&frm_grp_, frm_type, MPP_BUFFER_INTERNAL,
                                 "cam_frm", __FUNCTION__) != MPP_OK) {
            CAM_ERR("ION group failed, try DMA_HEAP...");
            frm_type = MPP_BUFFER_TYPE_DMA_HEAP;
            if (mpp_buffer_group_get(&frm_grp_, frm_type, MPP_BUFFER_INTERNAL,
                                     "cam_frm", __FUNCTION__) != MPP_OK) {
                CAM_ERR("DMA_HEAP group failed, try DRM...");
                frm_type = MPP_BUFFER_TYPE_DRM;
                if (mpp_buffer_group_get(&frm_grp_, frm_type, MPP_BUFFER_INTERNAL,
                                         "cam_frm", __FUNCTION__) != MPP_OK) {
                    CAM_ERR("DRM group failed, all hardware allocators exhausted");
                    break;
                }
            }
        }
        CAM_LOG("Frame buffer group type=" << frm_type);

        if (mpp_buffer_group_limit_config(frm_grp_, buf_size, 4) != MPP_OK) {
            CAM_ERR("mpp_buffer_group_limit_config failed");
            break;
        }

        if (mpp_buffer_get(frm_grp_, &frm_buf_, buf_size) != MPP_OK) {
            CAM_ERR("mpp_buffer_get(frm_buf) failed");
            break;
        }
        CAM_LOG("frm_buf allocated: size=" << mpp_buffer_get_size(frm_buf_)
                << " ptr=" << mpp_buffer_get_ptr(frm_buf_)
                << " fd=" << mpp_buffer_get_fd(frm_buf_));

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
        if (mpp_api_->control(mpp_ctx_, MPP_DEC_SET_OUTPUT_FORMAT, &out_fmt) != MPP_OK) {
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

        // MMAP 模式需要 packet buffer 做 memcpy（V4L2 MMAP 数据拷贝到 MPP）
        // DMA 模式直接 import V4L2 dma-buf，无需 pkt_grp_
        if (mode_ == Mode::MMAP) {
            if (mpp_buffer_group_get(&pkt_grp_, MPP_BUFFER_TYPE_ION, MPP_BUFFER_INTERNAL,
                                     "cam_pkt", __FUNCTION__) != MPP_OK) {
                CAM_ERR("mpp_buffer_group_get(pkt) failed");
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
        cleanup();
        return false;
    }

    CAM_LOG("MPP init OK");
    return true;
}
// ---------------------------------------------------------------------------
// 开始 / 停止 视频流
// ---------------------------------------------------------------------------
bool CamDmaMmap::start()
{
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        CAM_ERR("VIDIOC_STREAMON failed: " << strerror(errno));
        return false;
    }
    streaming_ = true;
    CAM_LOG("STREAMON OK");
    return true;
}

void CamDmaMmap::stop()
{
    if (!streaming_) return;

    CAM_LOG("Stopping...");
    release();

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
    streaming_ = false;

    for (auto& b : v4l2_bufs_) {
        if (mode_ == Mode::MMAP) {
            if (b.start && b.start != MAP_FAILED)
                munmap(b.start, b.len);
        } else {
            if (b.mpp_buf) {
                mpp_buffer_put(b.mpp_buf);
                b.mpp_buf = nullptr;
            }
        }
        b.fd = -1;
    }
    v4l2_bufs_.clear();

    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }

    cleanup();

    if (stats_total_ > 0) {
        CAM_LOG("Stopped. Total=" << stats_total_
                << " OK=" << stats_ok_
                << " Err=" << stats_err_);
    }
}

// ---------------------------------------------------------------------------
// 取图主逻辑
// ---------------------------------------------------------------------------
bool CamDmaMmap::grab()
{
    release();

    struct v4l2_buffer buf{};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = (mode_ == Mode::DMA) ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;

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
        CAM_ERR("VIDIOC_DQBUF failed: " << strerror(errno));
        return false;
    }

    v4l2_buf_idx_ = buf.index;
    stats_total_++;

    CAM_LOG("DQBUF idx=" << buf.index
                         << " bytesused=" << buf.bytesused
                         << " seq=" << buf.sequence);

    if (buf.bytesused == 0) {
        CAM_ERR("V4L2 buffer empty");
        release();
        stats_err_++;
        return false;
    }

    bool ok = false;
    if (fmt_ == Format::MJPEG) {
        ok = process_mjpeg(buf);
    } else {
        ok = process_yuyv(buf);
    }

    if (!ok) {
        release();
        stats_err_++;
        return false;
    }

    stats_ok_++;
    if (stats_total_ % 30 == 0) {
        log_stats();
    }
    return true;
}

// ---------------------------------------------------------------------------
// MJPEG 处理分发
// ---------------------------------------------------------------------------
bool CamDmaMmap::process_mjpeg(const struct v4l2_buffer& buf)
{
    if (mode_ == Mode::DMA)
        return process_mjpeg_dma(buf);
    else
        return process_mjpeg_mmap(buf);
}

// ---------------------------------------------------------------------------
// MJPEG + DMA：零拷贝路径。直接 import V4L2 的 dma-buf fd 给 MPP
// ---------------------------------------------------------------------------
bool CamDmaMmap::process_mjpeg_dma(const struct v4l2_buffer& buf)
{
    // import V4L2 dma-buf 到 MPP（零拷贝）
    MppBufferInfo info{};
    info.type = MPP_BUFFER_TYPE_EXT_DMA;
    info.fd   = v4l2_bufs_[buf.index].fd;
    info.size = buf.bytesused;
    info.ptr  = nullptr;
    info.hnd  = nullptr;

    MppBuffer pkt_buf = nullptr;
    MPP_RET ret = mpp_buffer_import(&pkt_buf, &info);
    if (ret != MPP_OK) {
        CAM_ERR("mpp_buffer_import failed, ret=" << ret);
        return false;
    }

    // 调试：检查 JPEG 头（dma-buf 通常支持 CPU 读）
    uint8_t* ptr = (uint8_t*)mpp_buffer_get_ptr(pkt_buf);
    if (ptr && !check_jpeg_header(ptr, buf.bytesused)) {
        mpp_buffer_put(pkt_buf);
        return false;
    }

    MppPacket packet = nullptr;
    ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
    if (ret != MPP_OK) {
        CAM_ERR("mpp_packet_init_with_buffer failed, ret=" << ret);
        mpp_buffer_put(pkt_buf);
        return false;
    }

    mpp_packet_set_size(packet, buf.bytesused);
    mpp_packet_set_length(packet, buf.bytesused);
    mpp_packet_set_pos(packet, mpp_packet_get_data(packet));

    MppMeta meta = mpp_packet_get_meta(packet);
    if (meta) {
        mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, out_frame_);
        CAM_DBG("Bound KEY_OUTPUT_FRAME to packet meta");
    } else {
        CAM_ERR("mpp_packet_get_meta returned null");
    }

    auto t0 = std::chrono::steady_clock::now();

    ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);
    if (ret != MPP_OK) {
        CAM_ERR("decode_put_packet failed, ret=" << ret);
        mpp_packet_deinit(&packet);
        mpp_buffer_put(pkt_buf);
        return false;
    }
    CAM_DBG("decode_put_packet OK");

    ret = mpp_api_->decode_get_frame(mpp_ctx_, &mpp_frame_);
    if (ret != MPP_OK || !mpp_frame_) {
        CAM_ERR("decode_get_frame failed, ret=" << ret << " frame=" << mpp_frame_);
        mpp_packet_deinit(&packet);
        mpp_buffer_put(pkt_buf);
        return false;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    CAM_DBG("MPP decode took " << ms << " ms");

    mpp_packet_deinit(&packet);
    mpp_buffer_put(pkt_buf);

    if (mpp_frame_ != out_frame_) {
        CAM_ERR("frame mismatch: expected " << out_frame_ << " got " << mpp_frame_);
    } else {
        CAM_DBG("frame match OK");
    }

    int out_w = mpp_frame_get_width(mpp_frame_);
    int out_h = mpp_frame_get_height(mpp_frame_);
    int out_stride = mpp_frame_get_hor_stride(mpp_frame_);
    CAM_LOG("Decoded: " << out_w << "x" << out_h << " stride=" << out_stride);

    if (mpp_frame_get_eos(mpp_frame_)) {
        CAM_LOG("found eos frame");
    }

    return true;
}

// ---------------------------------------------------------------------------
// MJPEG + MMAP：传统路径，需要 memcpy 到 MPP packet buffer
// ---------------------------------------------------------------------------
bool CamDmaMmap::process_mjpeg_mmap(const struct v4l2_buffer& buf)
{
    uint8_t* jpeg_ptr = (uint8_t*)v4l2_bufs_[buf.index].start;
    if (!check_jpeg_header(jpeg_ptr, buf.bytesused)) {
        return false;
    }

    MppBuffer pkt_buf = nullptr;
    MPP_RET ret = mpp_buffer_get(pkt_grp_, &pkt_buf, buf.bytesused);
    if (ret != MPP_OK) {
        CAM_ERR("mpp_buffer_get(pkt_buf) failed, ret=" << ret);
        return false;
    }

    memcpy(mpp_buffer_get_ptr(pkt_buf), jpeg_ptr, buf.bytesused);

    MppPacket packet = nullptr;
    ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
    if (ret != MPP_OK) {
        CAM_ERR("mpp_packet_init_with_buffer failed, ret=" << ret);
        mpp_buffer_put(pkt_buf);
        return false;
    }

    mpp_packet_set_size(packet, buf.bytesused);
    mpp_packet_set_length(packet, buf.bytesused);
    mpp_packet_set_pos(packet, mpp_packet_get_data(packet));

    MppMeta meta = mpp_packet_get_meta(packet);
    if (meta) {
        mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, out_frame_);
        CAM_DBG("Bound KEY_OUTPUT_FRAME to packet meta");
    } else {
        CAM_ERR("mpp_packet_get_meta returned null");
    }

    auto t0 = std::chrono::steady_clock::now();

    ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);
    if (ret != MPP_OK) {
        CAM_ERR("decode_put_packet failed, ret=" << ret);
        mpp_packet_deinit(&packet);
        mpp_buffer_put(pkt_buf);
        return false;
    }
    CAM_DBG("decode_put_packet OK");

    ret = mpp_api_->decode_get_frame(mpp_ctx_, &mpp_frame_);
    if (ret != MPP_OK || !mpp_frame_) {
        CAM_ERR("decode_get_frame failed, ret=" << ret << " frame=" << mpp_frame_);
        mpp_packet_deinit(&packet);
        mpp_buffer_put(pkt_buf);
        return false;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    CAM_DBG("MPP decode took " << ms << " ms");

    mpp_packet_deinit(&packet);
    mpp_buffer_put(pkt_buf);

    if (mpp_frame_ != out_frame_) {
        CAM_ERR("frame mismatch: expected " << out_frame_ << " got " << mpp_frame_);
    } else {
        CAM_DBG("frame match OK");
    }

    int out_w = mpp_frame_get_width(mpp_frame_);
    int out_h = mpp_frame_get_height(mpp_frame_);
    int out_stride = mpp_frame_get_hor_stride(mpp_frame_);
    CAM_LOG("Decoded: " << out_w << "x" << out_h << " stride=" << out_stride);

    if (mpp_frame_get_eos(mpp_frame_)) {
        CAM_LOG("found eos frame");
    }

    return true;
}

// ---------------------------------------------------------------------------
// YUYV 处理：MMAP 直接返回指针；DMA 直接返回 fd，均零拷贝
// ---------------------------------------------------------------------------
bool CamDmaMmap::process_yuyv(const struct v4l2_buffer& buf)
{
    if (mode_ == Mode::MMAP) {
        CAM_DBG("YUYV MMAP mode, direct ptr=" << v4l2_bufs_[buf.index].start);
    } else {
        CAM_DBG("YUYV DMA mode, fd=" << v4l2_bufs_[buf.index].fd
                << " bytesused=" << buf.bytesused);
    }
    return true;
}

// ---------------------------------------------------------------------------
// JPEG 头详细检查
// ---------------------------------------------------------------------------
bool CamDmaMmap::check_jpeg_header(const uint8_t* data, size_t len)
{
    if (len < 2) {
        CAM_ERR("JPEG data too short: " << len);
        return false;
    }

    bool soi_ok = (data[0] == 0xFF && data[1] == 0xD8);
    bool eoi_ok = (len >= 2 && data[len - 2] == 0xFF && data[len - 1] == 0xD9);

    CAM_LOG("JPEG SOI=" << (soi_ok ? "OK" : "FAIL")
                         << " EOI=" << (eoi_ok ? "OK" : "FAIL")
                         << " size=" << len);

    if (!soi_ok) {
        std::cerr << "  Header bytes: ";
        for (size_t i = 0; i < std::min(len, size_t(16)); ++i) {
            std::cerr << std::hex << std::setw(2) << std::setfill('0')
                      << (int)data[i] << " ";
        }
        std::cerr << std::dec << "\n";
        return false;
    }

    // 扫描 SOF 标记，打印内嵌宽高
    for (size_t i = 2; i + 9 < len; ++i) {
        if (data[i] == 0xFF && (data[i + 1] & 0xF0) == 0xC0 &&
            data[i + 1] != 0xC4 && data[i + 1] != 0xC8 && data[i + 1] != 0xCC)
        {
            int h = (data[i + 5] << 8) | data[i + 6];
            int w = (data[i + 7] << 8) | data[i + 8];
            CAM_LOG("JPEG SOF marker 0x" << std::hex << (int)data[i + 1]
                                          << std::dec << " found: " << w << "x" << h);
            break;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 释放当前帧：QBUF 回 V4L2，释放 MPP frame
// ---------------------------------------------------------------------------
void CamDmaMmap::release()
{
    if (v4l2_buf_idx_ >= 0) {
        struct v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = (mode_ == Mode::DMA) ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;
        buf.index  = v4l2_buf_idx_;
        if (mode_ == Mode::DMA) {
            buf.m.fd   = v4l2_bufs_[v4l2_buf_idx_].fd;
            buf.length = (uint32_t)v4l2_bufs_[v4l2_buf_idx_].len;
        }

        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            CAM_ERR("VIDIOC_QBUF in release failed: " << strerror(errno));
        } else {
            CAM_DBG("QBUF idx=" << v4l2_buf_idx_);
        }
        v4l2_buf_idx_ = -1;
    }

    if (mpp_frame_ && mpp_frame_ != out_frame_) {
        mpp_frame_deinit(&mpp_frame_);
    }
    mpp_frame_ = nullptr;
}

// ---------------------------------------------------------------------------
// 清理 MPP / Buffer 资源
// ---------------------------------------------------------------------------
void CamDmaMmap::cleanup()
{
    if (mpp_ctx_) {
        mpp_api_->reset(mpp_ctx_);
        mpp_destroy(mpp_ctx_);
        mpp_ctx_  = nullptr;
        mpp_api_  = nullptr;
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

    if (v4l2_dma_group_) {
        mpp_buffer_group_put(v4l2_dma_group_);
        v4l2_dma_group_ = nullptr;
    }

    hor_stride_ = 0;
    ver_stride_ = 0;
}

// ---------------------------------------------------------------------------
// 两套数据访问接口
// ---------------------------------------------------------------------------
uint8_t* CamDmaMmap::src_ptr() const
{
    if (mode_ != Mode::MMAP) {
        CAM_ERR("src_ptr() called in DMA mode");
        return nullptr;
    }
    if (v4l2_buf_idx_ < 0) return nullptr;

    if (fmt_ == Format::MJPEG) {
        if (!mpp_frame_) return nullptr;
        MppBuffer buffer = mpp_frame_get_buffer(mpp_frame_);
        if (!buffer) return nullptr;
        return (uint8_t*)mpp_buffer_get_ptr(buffer);
    } else {
        return (uint8_t*)v4l2_bufs_[v4l2_buf_idx_].start;
    }
}

int CamDmaMmap::src_fd() const
{
    if (mode_ != Mode::DMA) return -1;
    if (v4l2_buf_idx_ < 0) return -1;

    if (fmt_ == Format::MJPEG) {
        if (!mpp_frame_) return -1;
        MppBuffer buffer = mpp_frame_get_buffer(mpp_frame_);
        if (!buffer) return -1;
        return mpp_buffer_get_fd(buffer);
    } else {
        return v4l2_bufs_[v4l2_buf_idx_].fd;
    }
}

int CamDmaMmap::src_stride() const
{
    if (fmt_ == Format::MJPEG) return hor_stride_;
    return width_ * 2; // YUYV
}

int CamDmaMmap::src_ver_stride() const
{
    if (fmt_ == Format::MJPEG) return ver_stride_;
    return height_;
}

int CamDmaMmap::src_w() const { return width_; }
int CamDmaMmap::src_h() const { return height_; }

int CamDmaMmap::src_fmt() const
{
    if (fmt_ == Format::MJPEG)
        return RK_FORMAT_YCbCr_420_SP;
    else
        return RK_FORMAT_YUYV_422;
}

// ---------------------------------------------------------------------------
// 运行期统计
// ---------------------------------------------------------------------------
void CamDmaMmap::log_stats()
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - stats_start_).count();
    if (elapsed <= 0) elapsed = 1;

    double fps = static_cast<double>(stats_ok_) / elapsed;
    double err_rate = (stats_total_ > 0) ? (100.0 * stats_err_ / stats_total_) : 0.0;

    CAM_LOG("Stats: total=" << stats_total_
                            << " ok=" << stats_ok_
                            << " err=" << stats_err_
                            << " fps=" << fps
                            << " err_rate=" << err_rate << "%");
}
