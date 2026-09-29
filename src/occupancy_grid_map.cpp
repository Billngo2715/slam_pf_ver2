#include "occupancy_grid_map.hpp"
#include "particle_filter.hpp"   // định nghĩa đầy đủ struct Pose2D
#include <algorithm>
#include <cmath>
#include <limits>

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
OccupancyGridMap::OccupancyGridMap(int width, int height, double resolution,
                                   double origin_x, double origin_y,
                                   double prior_log_odds, double occ_prob_thresh)
    : width_(width), height_(height), resolution_(resolution),
      origin_x_(origin_x), origin_y_(origin_y),
      prior_log_odds_(prior_log_odds),
      data_valid_(false)
{
    occupied_thresh_log_ = std::log(occ_prob_thresh / (1.0 - occ_prob_thresh));
    // int8 threshold: prob → 0-100
    occ_thresh_int_ = static_cast<int>(occ_prob_thresh * 100.0);

    log_odds_.assign(height_, std::vector<double>(width_, prior_log_odds_));
    data_.resize(width_ * height_, -1);  // -1 = unknown
}

// ─────────────────────────────────────────────────────────────────────────────
// worldToGrid
// ─────────────────────────────────────────────────────────────────────────────
bool OccupancyGridMap::worldToGrid(double wx, double wy, int& gx, int& gy) const
{
    gx = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
    gy = static_cast<int>(std::floor((wy - origin_y_) / resolution_));
    return (gx >= 0 && gx < width_ && gy >= 0 && gy < height_);
}

// ─────────────────────────────────────────────────────────────────────────────
// updateCellLogOdds
// ─────────────────────────────────────────────────────────────────────────────
void OccupancyGridMap::updateCellLogOdds(int gx, int gy, double inc)
{
    if (gx < 0 || gx >= width_ || gy < 0 || gy >= height_) return;
    double& lo = log_odds_[gy][gx];
    lo = std::clamp(lo + inc, -20.0, 20.0);
    data_valid_ = false;  // flat cache cần rebuild
}

// ─────────────────────────────────────────────────────────────────────────────
// getRayCells — Bresenham với endpoint clamp
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::pair<int,int>> OccupancyGridMap::getRayCells(
    double x0, double y0, double x1, double y1) const
{
    std::vector<std::pair<int,int>> cells;
    int gx0, gy0, gx1, gy1;
    if (!worldToGrid(x0, y0, gx0, gy0)) return cells;

    if (!worldToGrid(x1, y1, gx1, gy1)) {
        gx1 = std::clamp(gx1, 0, width_  - 1);
        gy1 = std::clamp(gy1, 0, height_ - 1);
    }

    int dx = std::abs(gx1 - gx0);
    int dy = std::abs(gy1 - gy0);
    int sx = (gx0 < gx1) ? 1 : -1;
    int sy = (gy0 < gy1) ? 1 : -1;
    int err = dx - dy;
    int x = gx0, y = gy0;

    while (true) {
        cells.emplace_back(x, y);
        if (x == gx1 && y == gy1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }
    }
    return cells;
}

// ─────────────────────────────────────────────────────────────────────────────
// updateMap — inverse sensor model
// ─────────────────────────────────────────────────────────────────────────────
void OccupancyGridMap::updateMap(const Pose2D& pose,
                                 const sensor_msgs::msg::LaserScan& scan)
{
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

        auto cells = getRayCells(pose.x, pose.y, wx, wy);
        if (cells.empty()) continue;

        for (size_t j = 0; j + 1 < cells.size(); ++j)
            updateCellLogOdds(cells[j].first, cells[j].second, -0.7);
        updateCellLogOdds(cells.back().first, cells.back().second, 0.85);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// rebuildData — chuyển log_odds_ 2D → flat int8_t
// 0 = free, 100 = occupied, -1 = unknown
// ─────────────────────────────────────────────────────────────────────────────
void OccupancyGridMap::rebuildData() const
{
    data_.resize(width_ * height_);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            double prob = 1.0 - 1.0 / (1.0 + std::exp(log_odds_[y][x]));
            int8_t val;
            if (prob < 0.2)       val = 0;    // free
            else if (prob > 0.65) val = 100;  // occupied
            else                   val = -1;   // unknown
            data_[y * width_ + x] = val;
        }
    }
    data_valid_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// getData — trả về flat cache, rebuild nếu cần
// ─────────────────────────────────────────────────────────────────────────────
const std::vector<int8_t>& OccupancyGridMap::getData() const
{
    if (!data_valid_) rebuildData();
    return data_;
}