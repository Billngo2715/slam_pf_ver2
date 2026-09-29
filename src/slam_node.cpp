#include "slam_node.hpp"

#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <cmath>
#include <memory>

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
SlamNode::SlamNode()
    : Node("slam_node"),
      pf_(PFParams{})
{
    // ── Khai báo tham số ─────────────────────────────────────────────────────
    declare_parameter<int>   ("num_particles",       20);
    declare_parameter<int>   ("top_k_update",         5);
    declare_parameter<double>("init_sigma_xy",        0.3);
    declare_parameter<double>("init_sigma_theta",     0.2);
    declare_parameter<double>("motion_sigma_xy",      0.05);
    declare_parameter<double>("motion_sigma_theta",   0.02);
    declare_parameter<double>("icp_sigma",            0.1);
    declare_parameter<int>   ("min_occ_points",       10);
    declare_parameter<int>   ("icp_max_iter",         10);
    declare_parameter<double>("icp_tolerance",        1e-4);
    declare_parameter<double>("icp_max_corr_dist",    0.3);
    declare_parameter<double>("min_trans_update",     0.02);
    declare_parameter<double>("min_rot_update",       0.02);
    declare_parameter<bool>  ("init_at_zero",         true);

    declare_parameter<int>   ("map_width",            250);
    declare_parameter<int>   ("map_height",           250);
    declare_parameter<double>("map_resolution",       0.1);
    declare_parameter<double>("map_origin_x",        -12.5);
    declare_parameter<double>("map_origin_y",        -12.5);
    declare_parameter<double>("map_occ_thresh",       0.7);

    // ── Đọc tham số ──────────────────────────────────────────────────────────
    get_parameter("num_particles",      num_particles_);
    get_parameter("top_k_update",       top_k_update_);
    get_parameter("init_sigma_xy",      init_sigma_xy_);
    get_parameter("init_sigma_theta",   init_sigma_theta_);
    get_parameter("motion_sigma_xy",    motion_sigma_xy_);
    get_parameter("motion_sigma_theta", motion_sigma_theta_);
    get_parameter("icp_sigma",          icp_sigma_);
    get_parameter("min_occ_points",     min_occ_points_);
    get_parameter("icp_max_iter",       icp_max_iter_);
    get_parameter("icp_tolerance",      icp_tolerance_);
    get_parameter("icp_max_corr_dist",  icp_max_corr_dist_);
    get_parameter("min_trans_update",   min_trans_update_);
    get_parameter("min_rot_update",     min_rot_update_);
    get_parameter("init_at_zero",       init_at_zero_);

    get_parameter("map_width",          map_width_);
    get_parameter("map_height",         map_height_);
    get_parameter("map_resolution",     map_resolution_);
    get_parameter("map_origin_x",       map_origin_x_);
    get_parameter("map_origin_y",       map_origin_y_);
    get_parameter("map_occ_thresh",     map_occ_thresh_);

    // ── Build PFParams ────────────────────────────────────────────────────────
    pf_params_.num_particles      = num_particles_;
    pf_params_.top_k_update       = top_k_update_;
    pf_params_.init_sigma_xy      = init_sigma_xy_;
    pf_params_.init_sigma_theta   = init_sigma_theta_;
    pf_params_.motion_sigma_xy    = motion_sigma_xy_;
    pf_params_.motion_sigma_theta = motion_sigma_theta_;
    pf_params_.icp_sigma          = icp_sigma_;
    pf_params_.min_occ_points     = min_occ_points_;
    pf_params_.icp_max_iter       = icp_max_iter_;
    pf_params_.icp_tolerance      = icp_tolerance_;
    pf_params_.icp_max_corr_dist  = icp_max_corr_dist_;
    pf_params_.min_trans_update   = min_trans_update_;
    pf_params_.min_rot_update     = min_rot_update_;

    pf_ = ParticleFilter(pf_params_);

    // ── Map template — dùng để fill header/info khi publish ──────────────────
    map_template_.info.width               = map_width_;
    map_template_.info.height              = map_height_;
    map_template_.info.resolution          = map_resolution_;
    map_template_.info.origin.position.x   = map_origin_x_;
    map_template_.info.origin.position.y   = map_origin_y_;
    map_template_.info.origin.position.z   = 0.0;
    map_template_.info.origin.orientation.w = 1.0;
    map_template_.header.frame_id          = "map";

    // ── Publishers ───────────────────────────────────────────────────────────
    map_pub_       = create_publisher<nav_msgs::msg::OccupancyGrid>("map", 1);
    particles_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
        "particle_cloud", rclcpp::QoS(1));
    pose_pub_      = create_publisher<geometry_msgs::msg::PoseStamped>(
        "slam_pose", rclcpp::QoS(1));
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // ── Subscribers ───────────────────────────────────────────────────────────
    odom_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(
        this, "/odom");
    scan_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>(
        this, "/scan");
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), *odom_sub_, *scan_sub_);
    sync_->registerCallback(
        std::bind(&SlamNode::syncCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(),
        "SlamNode (FastSLAM) started: %d particles, top_k=%d, map %dx%d @ %.2fm/cell",
        num_particles_, top_k_update_, map_width_, map_height_, map_resolution_);
}

// ─────────────────────────────────────────────────────────────────────────────
// syncCallback
// ─────────────────────────────────────────────────────────────────────────────
void SlamNode::syncCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg)
{
    // 1. Lấy pose từ odometry
    double odom_x = odom_msg->pose.pose.position.x;
    double odom_y = odom_msg->pose.pose.position.y;
    tf2::Quaternion q(
        odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y,
        odom_msg->pose.pose.orientation.z,
        odom_msg->pose.pose.orientation.w);
    double odom_theta = tf2::getYaw(q);

    // 2. Lần đầu tiên: init PF — mỗi hạt tạo OccupancyGridMap riêng
    if (first_odom_) {
        Pose2D init = init_at_zero_
            ? Pose2D{0.0, 0.0, 0.0}
            : Pose2D{odom_x, odom_y, odom_theta};

        pf_.init(init,
                 map_width_, map_height_, map_resolution_,
                 map_origin_x_, map_origin_y_, map_occ_thresh_);

        last_odom_pose_ = {odom_x, odom_y, odom_theta};
        first_odom_     = false;

        RCLCPP_INFO(get_logger(),
            "PF initialized at (%.2f, %.2f, %.2f) [init_at_zero=%s]",
            init.x, init.y, init.theta,
            init_at_zero_ ? "true" : "false");
        return;
    }

    // 3. Delta odom → robot frame
    double dx_w   = odom_x     - last_odom_pose_.x;
    double dy_w   = odom_y     - last_odom_pose_.y;
    double dtheta = odom_theta - last_odom_pose_.theta;
    while (dtheta >  M_PI) dtheta -= 2.0 * M_PI;
    while (dtheta < -M_PI) dtheta += 2.0 * M_PI;

    double cl = std::cos(last_odom_pose_.theta);
    double sl = std::sin(last_odom_pose_.theta);
    Pose2D delta{cl * dx_w + sl * dy_w,
                -sl * dx_w + cl * dy_w,
                 dtheta};

    last_odom_pose_ = {odom_x, odom_y, odom_theta};

    // 4. Predict
    bool should_update = pf_.predict(delta);
    if (!should_update) {
        RCLCPP_DEBUG(get_logger(), "Skip update: motion below threshold");
        publishAll(pf_.getBestPose());
        return;
    }
    
    // 5. Update — mỗi hạt tự dùng map của chính mình
    // Không còn global map_, không còn getData() từ ngoài
    pf_.update(*scan_msg);

    // 6. Publish bản đồ và pose của hạt tốt nhất
    publishAll(pf_.getBestPose());
}

// ─────────────────────────────────────────────────────────────────────────────
// publishAll
// ─────────────────────────────────────────────────────────────────────────────
void SlamNode::publishAll(const Pose2D& robot_pose)
{
    auto stamp = this->now();

    // ── OccupancyGrid — map của hạt tốt nhất ─────────────────────────────────
    nav_msgs::msg::OccupancyGrid grid = map_template_;  // copy header/info
    grid.header.stamp = stamp;
    grid.data         = pf_.getBestMap();               // map data hạt tốt nhất
    map_pub_->publish(grid);

    // ── PoseArray (tất cả hạt) ────────────────────────────────────────────────
    geometry_msgs::msg::PoseArray pa;
    pa.header.stamp    = stamp;
    pa.header.frame_id = "map";
    pa.poses.reserve(pf_.getParticles().size());
    for (const auto& p : pf_.getParticles()) {
        geometry_msgs::msg::Pose pose;
        pose.position.x = p.pose.x;
        pose.position.y = p.pose.y;
        tf2::Quaternion qp; qp.setRPY(0, 0, p.pose.theta);
        pose.orientation = tf2::toMsg(qp);
        pa.poses.push_back(pose);
    }
    particles_pub_->publish(pa);

    // ── PoseStamped (hạt tốt nhất) ───────────────────────────────────────────
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp    = stamp;
    ps.header.frame_id = "map";
    ps.pose.position.x = robot_pose.x;
    ps.pose.position.y = robot_pose.y;
    {
        tf2::Quaternion qe; qe.setRPY(0, 0, robot_pose.theta);
        ps.pose.orientation = tf2::toMsg(qe);
    }
    pose_pub_->publish(ps);

    // ── TF map → odom ────────────────────────────────────────────────────────
    tf2::Transform T_map_robot, T_odom_robot;
    T_map_robot.setOrigin(tf2::Vector3(robot_pose.x, robot_pose.y, 0.0));
    { tf2::Quaternion q; q.setRPY(0, 0, robot_pose.theta); T_map_robot.setRotation(q); }
    T_odom_robot.setOrigin(tf2::Vector3(last_odom_pose_.x, last_odom_pose_.y, 0.0));
    { tf2::Quaternion q; q.setRPY(0, 0, last_odom_pose_.theta); T_odom_robot.setRotation(q); }

    tf2::Transform T_map_odom = T_map_robot * T_odom_robot.inverse();

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp            = stamp;
    tf_msg.header.frame_id         = "map";
    tf_msg.child_frame_id          = "odom";
    tf_msg.transform.translation.x = T_map_odom.getOrigin().x();
    tf_msg.transform.translation.y = T_map_odom.getOrigin().y();
    tf_msg.transform.translation.z = 0.0;
    tf_msg.transform.rotation       = tf2::toMsg(T_map_odom.getRotation());
    tf_broadcaster_->sendTransform(tf_msg);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SlamNode>());
    rclcpp::shutdown();
    return 0;
}