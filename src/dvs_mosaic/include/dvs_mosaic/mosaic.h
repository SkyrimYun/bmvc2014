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

#include <thread>
#include <mutex>
#include <condition_variable>

namespace dvs_mosaic
{

using Transformation = kindr::minimal::QuatTransformation;
//using Transformation = kindr::minimal::RotationQuaternion;

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

  // Sliding window of events
  std::deque<dvs_msgs::Event> events_;
  std::vector<dvs_msgs::Event> events_subset_;

  // Camera
  int sensor_width_, sensor_height_;
  image_geometry::PinholeCameraModel dvs_cam_;
  cv::Mat time_map_;

  // Mosaic parameters
  int mosaic_width_, mosaic_height_;
  cv::Size mosaic_size_;
  float fx_, fy_; // speed-up equiareal projection

  // Measurement function
  double var_R_mapping;
  double var_R_tracking;
  double C_th_;

  // reference ground truth value
  cv::Matx33d Rot_gt;

  // Mapping / mosaicing
  int num_packet_reconstrct_mosaic_;
  const double dNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<cv::Matx33d> map_of_last_rotations_;
  cv::Mat grad_map_, grad_map_covar_, mosaic_img_, mosaic_img_vis_;
  std::map<ros::Time, Transformation> poses_;
  void loadPoses();


  // Tracking 
  int num_events_update_;
  std::map<ros::Time, Transformation> poses_est_;
  cv::Mat rot_vec_;   // state for the tracker
  cv::Mat covar_rot_; // 3x3 covariance matrix
  double var_process_noise_;

  // packet
  int packet_number;
  cv::Matx33d Rot_packet;
  cv::Point2f pm_packet_min, pm_packet_max;

  // Threads
  std::thread reconstruct_thread_;
  void reconstuctMosaic();
  std::mutex data_lock_;
  std::condition_variable reconstruct_;

  // Debugging
  const bool visualize = true;
  const bool extra_log_debugging = true;
  

  void processEventForMap(const dvs_msgs::Event &ev, const cv::Matx33d Rot_prev);
  bool rotationAt(const ros::Time &t_query, cv::Matx33d &Rot_interp);
  //void project_EquirectangularProjection(const cv::Point3d &pt_3d, cv::Point2f &pt_on_mosaic);
  cv::Mat project_EquirectangularProjection(const cv::Point3d &pt_3d, cv::Point2f &pt_on_mosaic, bool calculate_d2d3 = false);

  // Precomputed bearing vectors for each camera pixel
  std::vector<cv::Point3d> precomputed_bearing_vectors_;
  void precomputeBearingVectors();

  void processEventForTrack(const dvs_msgs::Event &ev, const cv::Matx33d Rot_prev);

  const int get_mosaic_map = 0;
  const int get_grad_x = 1;
  const int get_grad_y = 2;
  double getMapBrightnessAt(const cv::Point2f &pm, int mode);

  double computePredictedConstrastOfEvent(
      const cv::Point2f &pm,
      const cv::Point2f &pm_prev);

  void computeDeriv(
      const cv::Point2f pm,
      const cv::Mat dpm_d3d,
      const cv::Point3d rotated_bvec,
      cv::Mat &Jac,
      bool is_analytic);
};

} // namespace
