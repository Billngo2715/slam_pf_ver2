#pragma once

#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "particle_filter.hpp"

class SlamNode : public rclcpp::Node
{
public:
    SlamNode();

private:
    void syncCallback(
        const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg,
        const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg);

    // publishAll dùng thẳng pf_.getBestMap() và pf_.getBestPose()
    void publishAll(const Pose2D& robot_pose);

    // ── Tham số ──────────────────────────────────────────────────────────────
    int    num_particles_;
    int    top_k_update_;
    double init_sigma_xy_, init_sigma_theta_;
    double motion_sigma_xy_, motion_sigma_theta_;
    double icp_sigma_;
    int    min_occ_points_, icp_max_iter_;
    double icp_tolerance_, icp_max_corr_dist_;
    double min_trans_update_, min_rot_update_;
    bool   init_at_zero_;

    int    map_width_, map_height_;
    double map_resolution_;
    double map_origin_x_, map_origin_y_;
    double map_occ_thresh_;

    // ── SLAM core — chỉ còn ParticleFilter, không còn map_ riêng ─────────────
    PFParams       pf_params_;
    ParticleFilter pf_;

    // Template header/info cho OccupancyGrid publish — tránh tạo lại mỗi frame
    nav_msgs::msg::OccupancyGrid map_template_;

    Pose2D last_odom_pose_;
    bool   first_odom_ = true;

    // ── ROS interfaces ────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr   map_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr  particles_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster>               tf_broadcaster_;

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        nav_msgs::msg::Odometry, sensor_msgs::msg::LaserScan>;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>>     odom_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::LaserScan>> scan_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>>                sync_;
};