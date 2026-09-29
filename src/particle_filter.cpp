#include "particle_filter.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <rclcpp/rclcpp.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
ParticleFilter::ParticleFilter(const PFParams& params)
    : params_(params)
    , rng_(std::random_device{}())
    , dist_xy_(0.0, params.motion_sigma_xy)
    , dist_th_(0.0, params.motion_sigma_theta)
    , dist_uni_(0.0, 1.0)
{}

// ─────────────────────────────────────────────────────────────────────────────
// Init — mỗi hạt nhận shared_ptr<OccupancyGridMap> riêng (rỗng, unknown)
// ─────────────────────────────────────────────────────────────────────────────
void ParticleFilter::init(const Pose2D& initial_pose,
                          int map_width, int map_height,
                          double map_resolution,
                          double map_origin_x, double map_origin_y,
                          double map_occ_thresh)
{
    // Lưu thông số map để dùng trong updateMapForParticle
    map_width_     = map_width;
    map_height_    = map_height;
    map_resolution_ = map_resolution;
    map_origin_x_  = map_origin_x;
    map_origin_y_  = map_origin_y;
    map_occ_thresh_ = map_occ_thresh;

    std::normal_distribution<double> dxy(0.0, params_.init_sigma_xy);
    std::normal_distribution<double> dth(0.0, params_.init_sigma_theta);

    particles_.resize(params_.num_particles);
    double w0 = 1.0 / params_.num_particles;

    for (auto& p : particles_) {
        p.pose.x     = initial_pose.x + dxy(rng_);
        p.pose.y     = initial_pose.y + dxy(rng_);
        p.pose.theta = initial_pose.theta + dth(rng_);
        while (p.pose.theta >  M_PI) p.pose.theta -= 2.0 * M_PI;
        while (p.pose.theta < -M_PI) p.pose.theta += 2.0 * M_PI;
        p.weight = w0;

        // Mỗi hạt có bản đồ riêng — dùng make_shared để tạo mới
        p.map = std::make_shared<OccupancyGridMap>(
            map_width, map_height, map_resolution,
            map_origin_x, map_origin_y,
            0.0, map_occ_thresh);
    }

    // Pre-allocate source_hits buffer
    source_hits_buf_.assign(map_width * map_height, 0);

    initialized_ = true;
    accum_trans_ = 0.0;
    accum_rot_   = 0.0;

    RCLCPP_INFO(rclcpp::get_logger("particle_filter"),
        "FastSLAM init: %d particles, top_k=%d, map %dx%d @ %.2fm",
        params_.num_particles, params_.top_k_update,
        map_width, map_height, map_resolution);
}

// ─────────────────────────────────────────────────────────────────────────────
// Predict
// ─────────────────────────────────────────────────────────────────────────────
bool ParticleFilter::predict(const Pose2D& delta)
{
    for (auto& p : particles_) {
        double ct = std::cos(p.pose.theta);
        double st = std::sin(p.pose.theta);

        p.pose.x     += ct * delta.x - st * delta.y + dist_xy_(rng_);
        p.pose.y     += st * delta.x + ct * delta.y + dist_xy_(rng_);
        p.pose.theta += delta.theta + dist_th_(rng_);

        while (p.pose.theta >  M_PI) p.pose.theta -= 2.0 * M_PI;
        while (p.pose.theta < -M_PI) p.pose.theta += 2.0 * M_PI;
    }

    accum_trans_ += std::hypot(delta.x, delta.y);
    accum_rot_   += std::fabs(delta.theta);

    return (accum_trans_ >= params_.min_trans_update ||
            accum_rot_   >= params_.min_rot_update);
}

// ─────────────────────────────────────────────────────────────────────────────
// buildSourceHits
// ─────────────────────────────────────────────────────────────────────────────
void ParticleFilter::buildSourceHits(
    const sensor_msgs::msg::LaserScan& scan,
    const Pose2D& pose,
    double resolution, double origin_x, double origin_y,
    int map_width, int map_height,
    std::vector<uint8_t>& source_hits) const
{
    std::fill(source_hits.begin(), source_hits.end(), 0);

    double cos_t = std::cos(pose.theta);
    double sin_t = std::sin(pose.theta);

    for (size_t i = 0; i < scan.ranges.size(); ++i) {
        double r = scan.ranges[i];
        if (r < scan.range_min || r > scan.range_max ||
            std::isinf(r) || std::isnan(r)) continue;

        double a  = scan.angle_min + i * scan.angle_increment;
        double lx = r * std::cos(a);
        double ly = r * std::sin(a);

        double wx = pose.x + lx * cos_t - ly * sin_t;
        double wy = pose.y + lx * sin_t + ly * cos_t;

        int gx = static_cast<int>(std::floor((wx - origin_x) / resolution));
        int gy = static_cast<int>(std::floor((wy - origin_y) / resolution));

        if (gx < 0 || gx >= map_width || gy < 0 || gy >= map_height) continue;
        source_hits[gy * map_width + gx] = 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// findCorrespondence — shell Chebyshev
// ─────────────────────────────────────────────────────────────────────────────
Correspondence ParticleFilter::findCorrespondence(
    int src_gx, int src_gy,
    const std::vector<int8_t>& map_data,
    int map_width, int map_height,
    int occ_thresh, int max_radius_cells,
    double resolution) const
{
    for (int r = 0; r <= max_radius_cells; ++r) {
        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                if (std::max(std::abs(dx), std::abs(dy)) != r) continue;
                int tgx = src_gx + dx;
                int tgy = src_gy + dy;
                if (tgx < 0 || tgx >= map_width ||
                    tgy < 0 || tgy >= map_height) continue;
                if (map_data[tgy * map_width + tgx] >
                    static_cast<int8_t>(occ_thresh)) {
                    double dc = std::sqrt(static_cast<double>(dx*dx + dy*dy));
                    return {src_gx, src_gy, tgx, tgy, dc, dc * resolution};
                }
            }
        }
    }
    double dmax = static_cast<double>(max_radius_cells);
    return {src_gx, src_gy, src_gx, src_gy, dmax, dmax * resolution};
}

// ─────────────────────────────────────────────────────────────────────────────
// buildCorrespondences
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Correspondence> ParticleFilter::buildCorrespondences(
    const std::vector<uint8_t>& source_hits,
    const std::vector<int8_t>&  map_data,
    int map_width, int map_height,
    int occ_thresh, int max_radius_cells,
    double resolution, double max_corr_dist_meter) const
{
    std::vector<Correspondence> corrs;
    corrs.reserve(256);

    for (int gy = 0; gy < map_height; ++gy) {
        for (int gx = 0; gx < map_width; ++gx) {
            if (source_hits[gy * map_width + gx] == 0) continue;
            auto c = findCorrespondence(gx, gy, map_data,
                                        map_width, map_height,
                                        occ_thresh, max_radius_cells,
                                        resolution);
            if (c.d_meter <= max_corr_dist_meter)
                corrs.push_back(c);
        }
    }
    return corrs;
}

// ─────────────────────────────────────────────────────────────────────────────
// icpFromCorrespondences — một bước SVD
// ─────────────────────────────────────────────────────────────────────────────
ICPResult ParticleFilter::icpFromCorrespondences(
    const std::vector<Correspondence>& corrs,
    double resolution,
    double origin_x, double origin_y) const
{
    if (corrs.size() < 3)
        return ICPResult{false, 0, 0, 0, 1e9};

    std::vector<Eigen::Vector2d> src_pts, tgt_pts;
    src_pts.reserve(corrs.size());
    tgt_pts.reserve(corrs.size());

    for (const auto& c : corrs) {
        src_pts.emplace_back(origin_x + (c.src_gx + 0.5) * resolution,
                             origin_y + (c.src_gy + 0.5) * resolution);
        tgt_pts.emplace_back(origin_x + (c.tgt_gx + 0.5) * resolution,
                             origin_y + (c.tgt_gy + 0.5) * resolution);
    }

    Eigen::Vector2d c_src = Eigen::Vector2d::Zero();
    Eigen::Vector2d c_tgt = Eigen::Vector2d::Zero();
    for (size_t i = 0; i < src_pts.size(); ++i) {
        c_src += src_pts[i];
        c_tgt += tgt_pts[i];
    }
    c_src /= static_cast<double>(src_pts.size());
    c_tgt /= static_cast<double>(src_pts.size());

    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    for (size_t i = 0; i < src_pts.size(); ++i)
        H += (src_pts[i] - c_src) * (tgt_pts[i] - c_tgt).transpose();

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix2d V = svd.matrixV();
        V.col(1) *= -1;
        R = V * svd.matrixU().transpose();
    }
    Eigen::Vector2d t = c_tgt - R * c_src;

    double error = 0.0;
    for (size_t i = 0; i < src_pts.size(); ++i)
        error += (R * src_pts[i] + t - tgt_pts[i]).squaredNorm();
    error /= static_cast<double>(src_pts.size());

    return ICPResult{true, t.x(), t.y(), std::atan2(R(1,0), R(0,0)), error};
}

// ─────────────────────────────────────────────────────────────────────────────
// computeLogLikelihood — log trung bình trên d_cell
// ─────────────────────────────────────────────────────────────────────────────
double ParticleFilter::computeLogLikelihood(
    const std::vector<Correspondence>& corrs,
    double sigma_meter, double resolution) const
{
    if (corrs.empty()) return -1e6;
    double sigma_cell  = sigma_meter / resolution;
    double inv_2sigma2 = 1.0 / (2.0 * sigma_cell * sigma_cell);
    double log_sum = 0.0;
    for (const auto& c : corrs)
        log_sum += -c.d_cell * c.d_cell * inv_2sigma2;
    return log_sum / static_cast<double>(corrs.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// updateMapForParticle — copy-on-write với shared_ptr
// Nếu map đang được share với hạt khác → deep copy trước khi ghi
// ─────────────────────────────────────────────────────────────────────────────
void ParticleFilter::updateMapForParticle(
    Particle& p,
    const sensor_msgs::msg::LaserScan& scan) const
{
    // Copy-on-write: nếu map đang được share (use_count > 1) → tách ra
    if (p.map.use_count() > 1)
        p.map = std::make_shared<OccupancyGridMap>(*p.map);

    // Gọi updateMap của OccupancyGridMap với pose đã hiệu chỉnh
    p.map->updateMap(p.pose, scan);
}

// ─────────────────────────────────────────────────────────────────────────────
// update — vòng lặp FastSLAM chính
// ─────────────────────────────────────────────────────────────────────────────
void ParticleFilter::update(const sensor_msgs::msg::LaserScan& scan)
{
    int max_radius = static_cast<int>(
        std::ceil(params_.icp_max_corr_dist / map_resolution_));
    if (max_radius < 1) max_radius = 1;

    int occ_thresh = static_cast<int>(map_occ_thresh_ * 100);

    using Clock = std::chrono::steady_clock;
    double icp_sum_ms = 0.0, icp_min_ms = 1e9, icp_max_ms = -1e9;

    // ── Bước 1: ICP + tính weight cho mỗi hạt dùng map CỦA CHÍNH HẠT ────────
    std::vector<double> log_weights(particles_.size(), -1e6);
    double max_lw = -std::numeric_limits<double>::max();

    for (size_t k = 0; k < particles_.size(); ++k) {
        Particle& p = particles_[k];

        // Lấy map data của chính hạt này
        const std::vector<int8_t>& map_data = p.map->getData();

        // Kiểm tra bản đồ đủ dữ liệu chưa (min_occ_points)
        int occ_count = 0;
        for (auto v : map_data)
            if (v > static_cast<int8_t>(occ_thresh)) ++occ_count;
        if (occ_count < params_.min_occ_points) {
            log_weights[k] = 0.0;  // bản đồ trống: weight đều nhau
            if (0.0 > max_lw) max_lw = 0.0;
            continue;
        }

        // ICP vòng lặp
        Pose2D corrected = p.pose;
        auto t0 = Clock::now();
        double prev_error = std::numeric_limits<double>::max();

        for (int iter = 0; iter < params_.icp_max_iter; ++iter) {
            buildSourceHits(scan, corrected,
                            map_resolution_, map_origin_x_, map_origin_y_,
                            map_width_, map_height_,
                            source_hits_buf_);

            auto corrs = buildCorrespondences(
                source_hits_buf_, map_data,
                map_width_, map_height_,
                occ_thresh, max_radius,
                map_resolution_, params_.icp_max_corr_dist);

            if (corrs.size() < 3) break;

            ICPResult res = icpFromCorrespondences(
                corrs, map_resolution_, map_origin_x_, map_origin_y_);

            if (!res.converged) break;

            corrected.x     += res.tx;
            corrected.y     += res.ty;
            corrected.theta += res.delta_theta;
            while (corrected.theta >  M_PI) corrected.theta -= 2.0 * M_PI;
            while (corrected.theta < -M_PI) corrected.theta += 2.0 * M_PI;

            if (iter > 0 &&
                std::fabs(prev_error - res.error) < params_.icp_tolerance)
                break;
            prev_error = res.error;
        }

        double icp_ms = std::chrono::duration<double, std::milli>(
                            Clock::now() - t0).count();
        icp_sum_ms += icp_ms;
        if (icp_ms < icp_min_ms) icp_min_ms = icp_ms;
        if (icp_ms > icp_max_ms) icp_max_ms = icp_ms;

        // Cập nhật pose hạt theo kết quả ICP
        p.pose = corrected;

        // Tính likelihood trên map của chính hạt
        buildSourceHits(scan, corrected,
                        map_resolution_, map_origin_x_, map_origin_y_,
                        map_width_, map_height_,
                        source_hits_buf_);

        auto final_corrs = buildCorrespondences(
            source_hits_buf_, map_data,
            map_width_, map_height_,
            occ_thresh, max_radius,
            map_resolution_, params_.icp_max_corr_dist);

        double lw = computeLogLikelihood(
            final_corrs, params_.icp_sigma, map_resolution_);

        log_weights[k] = lw;
        if (lw > max_lw) max_lw = lw;
    }

    RCLCPP_INFO(rclcpp::get_logger("particle_filter"),
        "[ICP timing] particles=%zu | avg=%.2fms | min=%.2fms | max=%.2fms",
        particles_.size(),
        icp_sum_ms / static_cast<double>(particles_.size()),
        icp_min_ms, icp_max_ms);

    // ── Bước 2: log-sum-exp → linear weights ─────────────────────────────────
    for (size_t k = 0; k < particles_.size(); ++k)
        particles_[k].weight = std::exp(log_weights[k] - max_lw);

    normalizeWeights();

    // ── Bước 3: Chọn top-K hạt để update map ─────────────────────────────────
    // Sắp xếp index theo weight giảm dần, lấy top_k_update đầu
    std::vector<int> indices(particles_.size());
    std::iota(indices.begin(), indices.end(), 0);
    int k = std::min(params_.top_k_update,
                     static_cast<int>(particles_.size()));
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&](int a, int b) {
            return particles_[a].weight > particles_[b].weight;
        });

    // Update map cho top-K — copy-on-write tự xử lý trong updateMapForParticle
    for (int i = 0; i < k; ++i)
        updateMapForParticle(particles_[indices[i]], scan);

    // ── Bước 4: Resample ─────────────────────────────────────────────────────
    // shared_ptr: copy hạt chỉ copy con trỏ map (rẻ), không deep copy
    resample();

    accum_trans_ = 0.0;
    accum_rot_   = 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// getBestParticle
// ─────────────────────────────────────────────────────────────────────────────
const Particle& ParticleFilter::getBestParticle() const
{
    return *std::max_element(
        particles_.begin(), particles_.end(),
        [](const Particle& a, const Particle& b) {
            return a.weight < b.weight;
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// normalizeWeights
// ─────────────────────────────────────────────────────────────────────────────
void ParticleFilter::normalizeWeights()
{
    double sum = 0.0;
    for (const auto& p : particles_) sum += p.weight;
    if (sum < 1e-12) {
        double w = 1.0 / particles_.size();
        for (auto& p : particles_) p.weight = w;
        return;
    }
    for (auto& p : particles_) p.weight /= sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// Resample — systematic resampling
// shared_ptr: copy Particle chỉ copy con trỏ 8 byte, không deep copy map
// Deep copy chỉ xảy ra khi updateMapForParticle ghi vào map bị share
// ─────────────────────────────────────────────────────────────────────────────
void ParticleFilter::resample()
{
    int N = static_cast<int>(particles_.size());
    std::vector<Particle> next;
    next.reserve(N);

    double step = 1.0 / N;
    double r    = dist_uni_(rng_) * step;
    double c    = particles_[0].weight;
    int    i    = 0;

    for (int m = 0; m < N; ++m) {
        double u = r + m * step;
        while (u > c && i < N - 1) { ++i; c += particles_[i].weight; }

        // Copy Particle: pose + weight copy bình thường
        // map: chỉ copy shared_ptr (tăng use_count, không deep copy)
        next.push_back(particles_[i]);
        next.back().weight = 1.0 / N;
    }
    particles_ = std::move(next);
}