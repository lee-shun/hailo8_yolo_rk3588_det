// fusion_roi_predictor.h
// 融合 ROI 预测引擎：PBAS + 运动学跟踪 + 7-Zone 概率融合
// 依赖: OpenCV (core, imgproc), OpenMP

#ifndef FUSION_ROI_PREDICTOR_H
#define FUSION_ROI_PREDICTOR_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

// ==================== 数据结构 ====================
struct ROI {
  int x, y;
  int w = 640, h = 640;
  float score = 0.0f;
  float prob_motion = 0.0f;
  float prob_bg_motion = 0.0f;
  float prob_blob = 0.0f;
  float prob_center = 0.0f;
  float prob_history = 0.0f;
  float prob_neighbor = 0.0f;
};

struct Detection {
  float x, y, w, h;
  float score;
  int cls;
};

struct PBASResult {
  cv::Mat fg_mask;
  float bg_dx = 0.0f, bg_dy = 0.0f;
  cv::Point2f curr_centroid;
  std::vector<cv::Point2f> blobs;
  int fg_pixels = 0;
  float reliability = 0.0f;
};

// ==================== 7-Zone 几何配置 ====================
struct ZoneGeometry {
  static constexpr int IMG_W = 1920;
  static constexpr int IMG_H = 1080;
  static constexpr int R = 640;

  struct Zone {
    int id, x, y;
    float cx, cy;
    float center_weight;
  };

  static const std::array<Zone, 7> &zones() {
    static std::array<Zone, 7> Z = {{
        {0, 0, 0, 320, 320, 0.0f},
        {1, 640, 0, 960, 320, 0.0f},
        {2, 1280, 0, 1600, 320, 0.0f},
        {3, 640, 220, 960, 540, 0.0f},
        {4, 0, 440, 320, 760, 0.0f},
        {5, 640, 440, 960, 760, 0.0f},
        {6, 1280, 440, 1600, 760, 0.0f},
    }};
    static bool init = false;
    if (!init) {
      for (auto &z : Z) {
        float dx = z.cx - IMG_W / 2.0f;
        float dy = z.cy - IMG_H / 2.0f;
        z.center_weight = std::exp(-std::sqrt(dx * dx + dy * dy) / 800.0f);
      }
      init = true;
    }
    return Z;
  }

  static int point_to_zone(float px, float py) {
    const auto &Z = zones();
    if (px >= Z[3].x && px < Z[3].x + R && py >= Z[3].y && py < Z[3].y + R)
      return 3;
    for (int i = 0; i < 7; ++i) {
      if (i == 3)
        continue;
      if (px >= Z[i].x && px < Z[i].x + R && py >= Z[i].y && py < Z[i].y + R)
        return i;
    }
    return 3;
  }

  static bool is_neighbor(int a, int b) {
    static const bool adj[7][7] = {
        {0, 1, 0, 0, 1, 0, 0}, {1, 0, 1, 1, 0, 1, 0}, {0, 1, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 0, 1, 0}, {0, 1, 0, 1, 1, 0, 1},
        {0, 0, 1, 0, 0, 1, 0},
    };
    return adj[a][b];
  }

  static ROI get_roi(int id) {
    const auto &Z = zones();
    return {Z[id].x, Z[id].y, 640, 640, 0, 0, 0, 0, 0, 0};
  }

  // [关键更新] 获取两Zone共享边界方向：0=左,1=右,2=上,3=下,-1=非邻接
  static int shared_edge_direction(int a, int b) {
    const auto &Za = zones()[a];
    const auto &Zb = zones()[b];
    if (Za.y == Zb.y && std::abs(Za.x - Zb.x) == 640) {
      return (Za.x < Zb.x) ? 1 : 0;
    }
    if (Za.x == Zb.x && std::abs(Za.y - Zb.y) > 0) {
      return (Za.y < Zb.y) ? 3 : 2;
    }
    if (a == 3 && b == 1)
      return 2;
    if (a == 3 && b == 5)
      return 3;
    if (a == 1 && b == 3)
      return 3;
    if (a == 5 && b == 3)
      return 2;
    return -1;
  }

  // [关键更新] 计算点(px,py)到zone_id某边界的距离
  static float distance_to_edge(float px, float py, int zone_id, int edge_dir) {
    const auto &z = zones()[zone_id];
    switch (edge_dir) {
    case 0:
      return px - z.x;
    case 1:
      return (z.x + 640) - px;
    case 2:
      return py - z.y;
    case 3:
      return (z.y + 640) - py;
    default:
      return 1e6f;
    }
  }
};

// ==================== PBAS (OpenMP 加速) ====================
class PBAS {
public:
  PBAS(int width, int height, int N = 15)
      : W(width), H(height), N_samples(N), RADIUS(5), MIN_MATCH(2) {
    samples_.resize(N);
    for (int i = 0; i < N; ++i) {
      samples_[i] = cv::Mat::zeros(H, W, CV_8U);
    }
    T_ = cv::Mat(H, W, CV_8U, cv::Scalar(RADIUS));
    R_ = cv::Mat(H, W, CV_8U, cv::Scalar(15));
    mean_dist_ = cv::Mat::zeros(H, W, CV_32F);
    init_ = false;
  }

  PBASResult update(const cv::Mat &gray) {
    CV_Assert(gray.type() == CV_8U && gray.cols == W && gray.rows == H);
    PBASResult res;
    res.fg_mask = cv::Mat::zeros(H, W, CV_8U);

    if (!init_) {
      for (int i = 0; i < N_samples; ++i)
        gray.copyTo(samples_[i]);
      init_ = true;
      return res;
    }

    int total_fg = 0;

#pragma omp parallel for schedule(dynamic, 16) reduction(+ : total_fg)
    for (int y = 0; y < H; ++y) {
      unsigned int seed =
          (unsigned int)(y * 12345 + 0xDEADBEEF + omp_get_thread_num());
      const uint8_t *gray_row = gray.ptr<uint8_t>(y);
      uint8_t *fg_row = res.fg_mask.ptr<uint8_t>(y);
      uint8_t *T_row = T_.ptr<uint8_t>(y);
      uint8_t *R_row = R_.ptr<uint8_t>(y);
      float *md_row = mean_dist_.ptr<float>(y);

      for (int x = 0; x < W; ++x) {
        uint8_t val = gray_row[x];
        int match = 0;
        float dist_sum = 0.0f;

        for (int n = 0; n < N_samples; ++n) {
          int d = std::abs(val - samples_[n].at<uint8_t>(y, x));
          dist_sum += d;
          if (d < T_row[x])
            match++;
        }
        float avg_dist = dist_sum / N_samples;

        bool is_fg = (match < MIN_MATCH);

        if (!is_fg) {
          float new_T = 0.7f * T_row[x] + 0.3f * avg_dist;
          T_row[x] = (uint8_t)std::clamp(new_T, 3.0f, 80.0f);
        } else {
          T_row[x] = (uint8_t)std::min(80, (int)T_row[x] + 1);
        }

        if (!is_fg) {
          float new_R = 10.0f + avg_dist * 2.0f;
          R_row[x] = (uint8_t)std::clamp(new_R, 5.0f, 60.0f);
        }

        if (!is_fg) {
          if ((rand_r(&seed) & 0xFF) < (255 / std::max(1, (int)R_row[x]))) {
            int nidx = rand_r(&seed) % N_samples;
            samples_[nidx].at<uint8_t>(y, x) = val;
          }
          fg_row[x] = 0;
        } else {
          fg_row[x] = 255;
          total_fg++;
        }

        md_row[x] = 0.9f * md_row[x] + 0.1f * avg_dist;
      }
    }

    res.fg_pixels = total_fg;

    if (total_fg < W * H * 0.5f) {
      std::vector<cv::Point> bg_points;
      bg_points.reserve(W * H / 2);
      for (int y = 1; y < H - 1; ++y) {
        const uint8_t *fg_row = res.fg_mask.ptr<uint8_t>(y);
        for (int x = 1; x < W - 1; ++x) {
          if (fg_row[x] == 0)
            bg_points.emplace_back(x, y);
        }
      }
      unsigned int seed_sp = 0x12345678;
      int propagate_count = std::min((int)bg_points.size() / 30, 1000);
      for (int i = 0; i < propagate_count; ++i) {
        int idx = rand_r(&seed_sp) % bg_points.size();
        const auto &p = bg_points[idx];
        int nx = p.x + (rand_r(&seed_sp) % 3 - 1);
        int ny = p.y + (rand_r(&seed_sp) % 3 - 1);
        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
          uint8_t val = gray.at<uint8_t>(p.y, p.x);
          samples_[rand_r(&seed_sp) % N_samples].at<uint8_t>(ny, nx) = val;
        }
      }
    }

    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(res.fg_mask, res.fg_mask, kernel, cv::Point(-1, -1), 1);
    cv::dilate(res.fg_mask, res.fg_mask, kernel, cv::Point(-1, -1), 1);

    if (total_fg > 30) {
      cv::Moments m = cv::moments(res.fg_mask, true);
      if (m.m00 > 0) {
        float cx = (float)(m.m10 / m.m00);
        float cy = (float)(m.m01 / m.m00);
        float scale_x = 1920.0f / W;
        float scale_y = 1080.0f / H;
        res.curr_centroid = cv::Point2f(cx * scale_x, cy * scale_y);
        res.bg_dx = (cx - W / 2.0f) * scale_x;
        res.bg_dy = (cy - H / 2.0f) * scale_y;
      }
      res.reliability = std::min(1.0f, total_fg / 2000.0f);
    }

    cv::Mat labels, stats, centroids;
    int ncomps = cv::connectedComponentsWithStats(res.fg_mask, labels, stats,
                                                  centroids, 8, CV_32S);
    float sx = 1920.0f / W;
    float sy = 1080.0f / H;
    for (int i = 1; i < ncomps; ++i) {
      int area = stats.at<int>(i, cv::CC_STAT_AREA);
      if (area < 5)
        continue;
      float cx = (float)centroids.at<double>(i, 0) * sx;
      float cy = (float)centroids.at<double>(i, 1) * sy;
      res.blobs.emplace_back(cx, cy);
    }

    return res;
  }

private:
  int W, H;
  int N_samples;
  int RADIUS;
  int MIN_MATCH;
  std::vector<cv::Mat> samples_;
  cv::Mat T_;
  cv::Mat R_;
  cv::Mat mean_dist_;
  bool init_;
};

// ==================== 运动学跟踪器（位置优先，几乎不外推）
// ====================
class KinematicTracker {
  struct Track {
    float x, y, vx, vy, ax, ay;
    int miss;
    float confidence;
  };
  std::optional<Track> track_;

public:
  void update(const Detection &det) {
    if (!track_) {
      track_ = Track{det.x, det.y, 0, 0, 0, 0, 0, 0.5f};
    } else {
      auto &t = *track_;
      // [关键更新] 残差 = 观测 - 当前位置（不再用外推预测做残差）
      float ex = det.x - t.x;
      float ey = det.y - t.y;

      // 位置更新：高度信任观测（α=0.85）
      t.x += 0.85f * ex;
      t.y += 0.85f * ey;

      // 速度更新：中等信任（β=0.60）
      t.vx += 0.60f * ex;
      t.vy += 0.60f * ey;

      // 加速度更新：低信任（γ=0.15），仅作平滑
      t.ax += 0.15f * ex;
      t.ay += 0.15f * ey;

      t.confidence = std::min(1.0f, t.confidence + 0.1f);
      t.miss = 0;
    }
  }

  void mark_miss() {
    if (!track_)
      return;
    track_->miss++;
    track_->confidence *= 0.7f;
    if (track_->miss > 5)
      track_.reset();
  }

  struct Prediction {
    float x, y;  // 预测位置（以当前位置为主，几乎不外推）
    float sigma; // 不确定性（由速度+丢失帧决定）
    float confidence;
  };

  Prediction predict() const {
    if (!track_)
      return {960.0f, 540.0f, 400.0f, 0.0f};

    const auto &t = *track_;

    // [关键更新] 位置预测：90% 相信当前位置，仅 10% 速度外推
    // 加速度不再直接叠加到位置，只通过 sigma 体现
    float px = t.x + 0.1f * t.vx;
    float py = t.y + 0.1f * t.vy;

    // 速度大小
    float v = std::sqrt(t.vx * t.vx + t.vy * t.vy);

    // [关键更新] sigma 加大，让速度和加速度转化为"搜索范围"而非"位置偏移"
    float sigma = 80.0f + v * 0.8f + t.miss * 60.0f;
    sigma = std::min(sigma, 500.0f);

    return {px, py, sigma, t.confidence};
  }

  bool has_track() const { return track_.has_value(); }
  int last_zone() const {
    return track_ ? ZoneGeometry::point_to_zone(track_->x, track_->y) : 3;
  }
};

// ==================== Zone 历史记忆 ====================
class ZoneHistory {
  std::array<float, 7> hist_ = {0};

public:
  void update(int hit_zone, bool detected) {
    for (auto &h : hist_)
      h *= 0.7f;
    if (detected && hit_zone >= 0 && hit_zone < 7)
      hist_[hit_zone] += 1.0f;
  }
  float get(int z) const { return hist_[z]; }
};

// ==================== 融合 ROI 预测引擎 ====================
class FusionROIPredictor {
  KinematicTracker tracker_;
  ZoneHistory history_;
  PBAS pbas_;
  int frame_id_ = 0;

  static constexpr float W_MOTION = 0.35f;
  static constexpr float W_BG_MOTION = 0.10f;
  static constexpr float W_BLOB = 0.15f;
  static constexpr float W_CENTER = 0.10f;
  static constexpr float W_HISTORY = 0.10f;
  static constexpr float W_NEIGHBOR = 0.20f;

public:
  FusionROIPredictor() : pbas_(320, 180, 15) {}

  struct Result {
    std::vector<ROI> rois;
    PBASResult pbas_res;
  };

  Result predict(const std::optional<Detection> &last_det,
                 const cv::Mat &gray_320x180) {
    frame_id_++;
    Result res;

    // 1. 更新目标跟踪器
    if (last_det) {
      tracker_.update(*last_det);
      history_.update(ZoneGeometry::point_to_zone(last_det->x, last_det->y),
                      true);
    } else {
      tracker_.mark_miss();
      history_.update(-1, false);
    }

    // 2. PBAS 背景减除
    res.pbas_res = pbas_.update(gray_320x180);
    const auto &motion = res.pbas_res;

    // 3. 目标运动预测 + 背景运动补偿
    auto pred = tracker_.predict();

    // [关键更新] 位置几乎不外推，背景运动补偿后仍贴近当前位置
    float fused_x = pred.x - 0.2 * motion.bg_dx;
    float fused_y = pred.y - 0.2 * motion.bg_dy;

    // sigma 包含 PBAS 可靠性惩罚（可靠性低时扩大搜索范围）
    float fused_sigma = pred.sigma + (1.0f - motion.reliability) * 100.0f;

    // 4. [关键更新] 计算每个 Zone 的"目标区域重叠概率"（尺寸感知）
    const auto &Z = ZoneGeometry::zones();
    std::vector<float> raw_motion(7, 0.0f);
    for (int i = 0; i < 7; ++i) {
      raw_motion[i] = compute_area_overlap(fused_x, fused_y, fused_sigma, Z[i]);
    }

    // 5. [关键更新] 邻接扩散：高概率 Zone 的相邻 Zone 获得加成
    // 当目标靠近共享边界时，让相邻 Zone 同时获得高概率
    std::vector<float> adj_boost(7, 0.0f);
    for (int i = 0; i < 7; ++i) {
      if (raw_motion[i] < 0.2f)
        continue; // 只扩散高概率 Zone

      for (int j = 0; j < 7; ++j) {
        if (i == j)
          continue;
        if (!ZoneGeometry::is_neighbor(i, j))
          continue;

        // 检查目标是否靠近 Zone i 与 Zone j 的共享边界
        int edge_dir = ZoneGeometry::shared_edge_direction(i, j);
        if (edge_dir < 0)
          continue;

        float dist =
            ZoneGeometry::distance_to_edge(fused_x, fused_y, i, edge_dir);
        // 如果目标靠近共享边界（< 40px），Zone j 获得显著加成
        if (dist < 40.0f) {
          // 越靠近边界，加成越大；紧贴边界时，Zone j 获得 Zone i 概率的 50%
          float boost = raw_motion[i] * 0.5f * (1.0f - dist / 40.0f);
          adj_boost[j] = std::max(adj_boost[j], boost);
        }
      }
    }

    // 6. [关键更新] 合并概率并生成 ROI
    std::vector<ROI> candidates;
    for (int i = 0; i < 7; ++i) {
      ROI roi = ZoneGeometry::get_roi(i);
      const auto &z = Z[i];

      // p_motion = max(自身重叠, 邻接扩散)
      // 这样当目标在交界处时，两个 Zone 都能获得高概率
      float p_motion = std::max(raw_motion[i], adj_boost[i]);

      float p_bg = compute_bg_motion_prob(i, motion, z);
      float p_blob = 0.0f;
      for (const auto &b : motion.blobs) {
        if (b.x >= z.x && b.x < z.x + 640 && b.y >= z.y && b.y < z.y + 640) {
          p_blob = 1.0f;
          break;
        }
      }

      float p_center = z.center_weight;
      float p_hist = std::min(1.0f, history_.get(i));

      // [关键更新] p_neighbor：边界感知
      // 不仅判断是否邻接，还判断是否靠近共享边界
      float p_neighbor = 0.5f;
      if (tracker_.has_track()) {
        int last_z = tracker_.last_zone();
        if (last_z == i) {
          p_neighbor = 1.0f; // 同一 Zone
        } else if (ZoneGeometry::is_neighbor(last_z, i)) {
          // 检查目标是否靠近从 last_z 到 i 的共享边界
          int edge_dir = ZoneGeometry::shared_edge_direction(last_z, i);
          float dist = ZoneGeometry::distance_to_edge(fused_x, fused_y, last_z,
                                                      edge_dir);
          if (dist < 30.0f) {
            // 非常靠近边界，高概率
            p_neighbor = 0.85f;
          } else {
            p_neighbor = 0.6f;
          }
        } else {
          p_neighbor = 0.15f; // 非邻接，强惩罚
        }
      }

      roi.score = W_MOTION * p_motion + W_BG_MOTION * p_bg + W_BLOB * p_blob +
                  W_CENTER * p_center + W_HISTORY * p_hist +
                  W_NEIGHBOR * p_neighbor;

      roi.prob_motion = p_motion;
      roi.prob_bg_motion = p_bg;
      roi.prob_blob = p_blob;
      roi.prob_center = p_center;
      roi.prob_history = p_hist;
      roi.prob_neighbor = p_neighbor;

      candidates.push_back(roi);
    }

    // 7. 按总分降序排序
    std::sort(candidates.begin(), candidates.end(),
              [](const ROI &a, const ROI &b) { return a.score > b.score; });

    // 8. [关键更新] 确保中心 Zone 3 在 Top-4（而非 Top-3）
    ensure_center_in_top4(candidates);

    // 9. [关键更新] 返回全部 7 个 ROI，不截断
    // 这样即使 Top-1 预测错误，后续 ROI 仍能兜底检测
    res.rois = candidates;

    return res;
  }

private:
  // [关键更新] 替换 compute_gaussian_overlap → compute_area_overlap
  // 使用 20x20 目标区域与 Zone 640x640 的重叠面积比例
  float compute_area_overlap(float mx, float my, float sigma,
                             const ZoneGeometry::Zone &z) {
    // 目标区域：10x10 目标 + 运动余量 = 20x20
    float target_r = 10.0f;
    float t_x1 = mx - target_r, t_x2 = mx + target_r;
    float t_y1 = my - target_r, t_y2 = my + target_r;
    float target_area = (2 * target_r) * (2 * target_r);

    // Zone 矩形
    float z_x1 = z.x, z_x2 = z.x + 640;
    float z_y1 = z.y, z_y2 = z.y + 640;

    // 重叠面积
    float overlap_x =
        std::max(0.0f, std::min(t_x2, z_x2) - std::max(t_x1, z_x1));
    float overlap_y =
        std::max(0.0f, std::min(t_y2, z_y2) - std::max(t_y1, z_y1));
    float overlap_area = overlap_x * overlap_y;
    float overlap_ratio = overlap_area / target_area;

    // 高斯距离（中心点到 Zone 最近点）
    float closest_x = std::max(z_x1, std::min(mx, z_x2));
    float closest_y = std::max(z_y1, std::min(my, z_y2));
    float dx = mx - closest_x;
    float dy = my - closest_y;
    float dist = std::sqrt(dx * dx + dy * dy);
    float gauss_prob = std::exp(-dist * dist / (2.0f * sigma * sigma + 1e-6f));

    // 综合：在 Zone 内时以重叠比例为主，在 Zone 外时以高斯尾概率为主
    if (mx >= z_x1 && mx < z_x2 && my >= z_y1 && my < z_y2) {
      // 目标在 Zone 内：0.5~1.0，完全覆盖时 1.0
      return 0.5f + 0.5f * overlap_ratio;
    } else {
      // 目标在 Zone 外：靠重叠比例和高斯共同决定
      // 如果跨越边界（overlap_ratio > 0），概率显著 > 0
      return 0.3f * gauss_prob + 0.7f * overlap_ratio;
    }
  }

  float compute_bg_motion_prob(int zone_id, const PBASResult &motion,
                               const ZoneGeometry::Zone &z) {
    if (motion.reliability < 0.1f)
      return 0.5f;
    float affected_x = z.cx + motion.bg_dx;
    float affected_y = z.cy + motion.bg_dy;
    float dx = affected_x - z.cx;
    float dy = affected_y - z.cy;
    float align = std::exp(-(dx * dx + dy * dy) / (2.0f * 200.0f * 200.0f));
    return 0.3f + 0.7f * align * motion.reliability;
  }

  // [关键更新] 确保中心 Zone 3 在 Top-4
  void ensure_center_in_top4(std::vector<ROI> &candidates) {
    bool center_in_top4 = false;
    for (int i = 0; i < std::min(4, (int)candidates.size()); ++i) {
      if (candidates[i].x == 640 && candidates[i].y == 220) {
        center_in_top4 = true;
        break;
      }
    }
    if (!center_in_top4) {
      // 找到中心 Zone 3，与第 4 名（索引 3）交换
      for (size_t i = 4; i < candidates.size(); ++i) {
        if (candidates[i].x == 640 && candidates[i].y == 220) {
          std::swap(candidates[3], candidates[i]);
          break;
        }
      }
    }
  }
};

#endif // FUSION_ROI_PREDICTOR_H
