#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <sensor_msgs/msg/laser_scan.hpp>

// Forward declare để tránh circular dependency
// Pose2D được định nghĩa trong particle_filter.hpp
struct Pose2D;

class OccupancyGridMap {
public:
    OccupancyGridMap() = default;
    OccupancyGridMap(int width, int height, double resolution,
                     double origin_x, double origin_y,
                     double prior_log_odds  = 0.0,
                     double occ_prob_thresh = 0.7);

    // Cập nhật bản đồ với inverse sensor model (Bresenham ray tracing)
    void updateMap(const Pose2D& pose, const sensor_msgs::msg::LaserScan& scan);

    // Trả về flat array int8_t [0-100 | -1] — cùng format nav_msgs/OccupancyGrid
    // Cache tự động rebuild khi log_odds_ thay đổi
    const std::vector<int8_t>& getData() const;

    // Ngưỡng occ (int8, 0-100)
    int getOccThresh() const { return occ_thresh_int_; }

    const std::vector<std::vector<double>>& getLogOdds() const { return log_odds_; }

    double probabilityFromLogOdds(double l) const {
        return 1.0 - 1.0 / (1.0 + std::exp(l));
    }

    int    getWidth()      const { return width_;      }
    int    getHeight()     const { return height_;     }
    double getResolution() const { return resolution_; }
    double getOriginX()    const { return origin_x_;   }
    double getOriginY()    const { return origin_y_;   }

private:
    bool worldToGrid(double wx, double wy, int& gx, int& gy) const;

    std::vector<std::pair<int,int>> getRayCells(
        double x0, double y0, double x1, double y1) const;

    void updateCellLogOdds(int gx, int gy, double inc);

    void rebuildData() const;

    int    width_ = 0, height_ = 0;
    double resolution_ = 0.1;
    double origin_x_ = 0.0, origin_y_ = 0.0;
    double prior_log_odds_    = 0.0;
    double occupied_thresh_log_ = 0.847;  // log(0.7/0.3)
    int    occ_thresh_int_    = 70;

    std::vector<std::vector<double>> log_odds_;

    mutable std::vector<int8_t> data_;
    mutable bool                data_valid_ = false;
};