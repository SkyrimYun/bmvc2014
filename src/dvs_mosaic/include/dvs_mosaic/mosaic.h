#pragma once

#include <ros/ros.h>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <opencv2/core/core.hpp>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <deque>

#include <kindr/minimal/quat-transformation.h>
#include <image_geometry/pinhole_camera_model.h>

#include <matplotlibcpp.h>

#include <sophus/so3.hpp>

namespace dvs_mosaic
{

using Transformation = kindr::minimal::QuatTransformation;

class Mosaic {
public:
  Mosaic(ros::NodeHandle & nh, ros::NodeHandle nh_private);
  virtual ~Mosaic();

private:
  ros::NodeHandle nh_;   // Node handle used to subscribe to ROS topics
  ros::NodeHandle pnh_;  // Private node handle for reading parameters

  // Callback functions
  void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);

  // Subscribers
  ros::Subscriber event_sub_;

  // Publishers
  image_transport::Publisher mosaic_pub_;
  image_transport::Publisher time_map_pub_;
  image_transport::Publisher mosaic_gradx_pub_, mosaic_grady_pub_, mosaic_tracecov_pub_;
  ros::Publisher pose_pub_;
  void publishMap();
  void publishPose();
  ros::Time time_packet_;
  image_transport::Publisher pose_cop_pub_; // For compare estimated pose and GT pose on tracker standalone mode


  // Sliding window of events
  std::deque<dvs_msgs::Event> events_;
  std::vector<dvs_msgs::Event> events_subset_;

  // Camera
  int sensor_width_, sensor_height_;
  int sensor_bottom_right, sensor_upper_left;
  image_geometry::PinholeCameraModel dvs_cam_;
  cv::Mat time_map_;

  // Mosaic parameters
  int mosaic_width_, mosaic_height_;
  cv::Size mosaic_size_;
  float fx_, fy_; // speed-up equiareal projection

  // Measurement function
  bool measure_contrast_;
  double var_R_mapping_;
  double var_R_tracking_;
  double C_th_;

  // Viusalization & Debugging setting
  const bool visualize = true;
  const bool extra_log_debugging = true;
  bool display_accuracy_;
  bool use_partial_mosaic_; // set true to enable partial mosaic map intput on tracker standalone mode
  int partial_mosaic_dur_;  // 1 -> 0.1s; 3 -> 0.3s; 5 -> 0.5s

  // Mapping / mosaicing
  int num_packet_reconstrct_mosaic_;
  int idx_first_ev_map_;  // index of first event of processing window
  const double dNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<cv::Matx33d> map_of_last_rotations_;
  cv::Mat grad_map_, grad_map_covar_, mosaic_img_, mosaic_img_recons_, pano_ev;
  std::map<ros::Time, Transformation> poses_, poses_est_;
  std::vector<double> pose_covar_est_;
  void loadPoses();
  double gaussian_blur_sigma_;
  bool use_gaussian_blur_;

  // Packet thresholds and statistics
  unsigned int packet_number = 0;
  int skip_count_polygon_;
  int skip_count_grad_;
  int skip_count_bright_;
  cv::Matx33d Rot_gt;
  int init_packet_num_;
  std::vector<Sophus::SO3d> recorded_pose_gt_;
  std::vector<Sophus::SO3d> recorded_pose_est_;

  // Tracking
  int num_events_update_;
  cv::Mat rot_vec_;   // state for the tracker
  cv::Mat covar_rot_; // 3x3 covariance matrix
  double var_process_noise_;
  std::vector<cv::Point> tracking_polygon_; // 4 points polygon for tracking area limitation
  bool tracker_standalone_; // set true to enable independent tracker operation
  bool use_grad_thres_;
  double grad_thres_;
  bool use_polygon_thres_;
  double tracking_area_percent_;
  bool use_bright_thres_;
  double bright_thres_;

  cv::Mat project_EquirectangularProjection(const cv::Point3d &pt_3d, cv::Point2f &pt_on_mosaic, bool calculate_d2d3 = false);
  bool rotationAt(const ros::Time& t_query, cv::Matx33d& Rot_interp);
  void processEventForMap(const dvs_msgs::Event &ev, const cv::Matx33d Rot_prev);

  // Precomputed bearing vectors for each camera pixel
  std::vector<cv::Point3d> precomputed_bearing_vectors_;
  void precomputeBearingVectors();

  // Calculate Tracking Polygon pixel location
  void calculatePacketPoly();

  void processEventForTrack(const dvs_msgs::Event &ev, const cv::Matx33d Rot_prev);

  // Obtain brightness or gradient value on mosiac map
  const int get_mosaic_map = 0;
  const int get_grad_x = 1;
  const int get_grad_y = 2;
  double getMapBrightnessAt(const cv::Point2f &pm, int mode);

  // Compuate constrast
  double computePredictedConstrastOfEvent(
      const cv::Point2f &pm,
      const cv::Point2f &pm_prev);

  // Compute derivative
  void computeDeriv(
      const cv::Point2f pm,
      const cv::Mat dpm_d3d,
      const cv::Point3d rotated_bvec,
      cv::Mat &Jac);

  // Collect estimated and ground truth pose for rmse 
  void dataCollect();
};

} // namespace
