#pragma once

#include <vector>
#include <memory>
#include <random>
#include <cstdint>
#include <Eigen/Dense>
#include <sensor_msgs/msg/laser_scan.hpp>

#include "occupancy_grid_map.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Structs cơ bản
// ─────────────────────────────────────────────────────────────────────────────
struct Pose2D {
    double x     = 0.0;
    double y     = 0.0;
    double theta = 0.0;  // radian [-pi, pi]
};

// Mỗi hạt mang theo bản đồ riêng qua shared_ptr
// shared_ptr cho phép resample rẻ (chỉ copy con trỏ)
// Deep copy chỉ xảy ra khi cần ghi (updateMapForParticle)
struct Particle {
    Pose2D                          pose;
    double                          weight = 1.0;
    std::shared_ptr<OccupancyGridMap> map;  // bản đồ riêng của hạt
};

struct Correspondence {
    int    src_gx, src_gy;
    int    tgt_gx, tgt_gy;
    double d_cell;
    double d_meter;
};

struct ICPResult {
    bool   converged   = false;
    double tx          = 0.0;
    double ty          = 0.0;
    double delta_theta = 0.0;
    double error       = 1e9;
};

struct PFParams {
    int    num_particles      = 20;
    int    top_k_update       = 5;     // chỉ update map cho K hạt tốt nhất
    double init_sigma_xy      = 0.3;
    double init_sigma_theta   = 0.2;
    double motion_sigma_xy    = 0.05;
    double motion_sigma_theta = 0.02;
    double icp_sigma          = 0.1;
    int    min_occ_points     = 10;
    int    icp_max_iter       = 10;
    double icp_tolerance      = 1e-4;
    double icp_max_corr_dist  = 0.3;
    double min_trans_update   = 0.02;
    double min_rot_update     = 0.02;
};

// ─────────────────────────────────────────────────────────────────────────────
// ParticleFilter — FastSLAM 1.0: mỗi hạt có bản đồ riêng
// ─────────────────────────────────────────────────────────────────────────────
class ParticleFilter {
public:
    explicit ParticleFilter(const PFParams& params = PFParams{});

    // Khởi tạo: mỗi hạt nhận một OccupancyGridMap rỗng riêng
    void init(const Pose2D& initial_pose,
              int map_width, int map_height,
              double map_resolution,
              double map_origin_x, double map_origin_y,
              double map_occ_thresh);

    // Trả về true nếu chuyển động đủ lớn để update
    bool predict(const Pose2D& delta_pose_robot_frame);

    // update():
    //   1. ICP mỗi hạt dùng map của chính hạt đó
    //   2. Tính weight từ map của chính hạt
    //   3. Normalize weights
    //   4. Update map cho top-K hạt (copy-on-write với shared_ptr)
    //   5. Resample (chỉ copy shared_ptr, không deep copy map)
    void update(const sensor_msgs::msg::LaserScan& scan);

    // Hạt có weight cao nhất
    const Particle& getBestParticle() const;

    // Pose hạt tốt nhất
    Pose2D getBestPose() const { return getBestParticle().pose; }

    // Map data hạt tốt nhất (để publish)
    const std::vector<int8_t>& getBestMap() const {
        return getBestParticle().map->getData();
    }

    const std::vector<Particle>& getParticles() const { return particles_; }
    bool isInitialized() const { return initialized_; }

private:
    // ── Grid helpers (giữ nguyên từ phiên bản trước) ─────────────────────────
    void buildSourceHits(
        const sensor_msgs::msg::LaserScan& scan,
        const Pose2D& pose,
        double resolution, double origin_x, double origin_y,
        int map_width, int map_height,
        std::vector<uint8_t>& source_hits) const;

    Correspondence findCorrespondence(
        int src_gx, int src_gy,
        const std::vector<int8_t>& map_data,
        int map_width, int map_height,
        int occ_thresh, int max_radius_cells,
        double resolution) const;

    std::vector<Correspondence> buildCorrespondences(
        const std::vector<uint8_t>& source_hits,
        const std::vector<int8_t>&  map_data,
        int map_width, int map_height,
        int occ_thresh, int max_radius_cells,
        double resolution, double max_corr_dist_meter) const;

    ICPResult icpFromCorrespondences(
        const std::vector<Correspondence>& corrs,
        double resolution,
        double origin_x, double origin_y) const;

    double computeLogLikelihood(
        const std::vector<Correspondence>& corrs,
        double sigma_meter, double resolution) const;

    // Update map riêng của hạt p (copy-on-write nếu shared)
    void updateMapForParticle(
        Particle& p,
        const sensor_msgs::msg::LaserScan& scan) const;

    // ── Weight helpers ────────────────────────────────────────────────────────
    void normalizeWeights();
    void resample();

    // ── State ─────────────────────────────────────────────────────────────────
    PFParams              params_;
    std::vector<Particle> particles_;
    bool                  initialized_ = false;

    // Thông số bản đồ — dùng khi init map cho mỗi hạt
    int    map_width_    = 250;
    int    map_height_   = 250;
    double map_resolution_ = 0.1;
    double map_origin_x_ = -12.5;
    double map_origin_y_ = -12.5;
    double map_occ_thresh_ = 0.7;

    double accum_trans_ = 0.0;
    double accum_rot_   = 0.0;

    // Pre-allocated source_hits buffer — tái dùng cho tất cả hạt
    mutable std::vector<uint8_t> source_hits_buf_;

    mutable std::mt19937                   rng_;
    std::normal_distribution<double>       dist_xy_;
    std::normal_distribution<double>       dist_th_;
    std::uniform_real_distribution<double> dist_uni_;
};