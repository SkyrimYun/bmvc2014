#include <dvs_mosaic/mosaic.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <glog/logging.h>
#include <camera_info_manager/camera_info_manager.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <opencv2/highgui.hpp> //cv::imwrite
#include <opencv2/calib3d.hpp> // Rodrigues
#include <fstream>
#include <dvs_mosaic/image_util.h>
#include <dvs_mosaic/reconstruction.h>

namespace dvs_mosaic
{

Mosaic::Mosaic(ros::NodeHandle & nh, ros::NodeHandle nh_private)
 : nh_(nh)
 , pnh_("~")
{
  // Get parameters
  // nh_private.param<int>("Num_ev_map_update", num_events_map_update_, 10000);
  // nh_private.param<int>("Num_ev_pose_update", num_events_pose_update_, 500);
  num_packet_reconstrct_mosaic_ = 20;
  num_events_update_ = 500;

  // Set up subscribers
  event_sub_ = nh_.subscribe("events", 0, &Mosaic::eventsCallback, this);


  // Set up publishers
  image_transport::ImageTransport it_(nh_);
  mosaic_pub_ = it_.advertise("mosaic", 1);
  time_map_pub_ = it_.advertise("time_map", 1);
  mosaic_gradx_pub_ = it_.advertise("mosaic_gx", 1);
  mosaic_grady_pub_ = it_.advertise("mosaic_gy", 1);
  mosaic_tracecov_pub_ = it_.advertise("mosaic_trace_cov", 1);
  pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("mosaic_pose", 1);


  // Event processing in batches / packets
  time_packet_ = ros::Time(0);

  // Camera information
  std::string cam_name("DVS-synthetic"); // yaml file should be in placed in /home/ggb/.ros/camera_info
  camera_info_manager::CameraInfoManager cam_info (nh_, cam_name);
  dvs_cam_.fromCameraInfo(cam_info.getCameraInfo());
  const cv::Size sensor_resolution = dvs_cam_.fullResolution();
  sensor_width_ = sensor_resolution.width;
  sensor_height_ = sensor_resolution.height;
  precomputeBearingVectors();

  // Mosaic size (in pixels)
  mosaic_height_ = 1024; // 512 or 256 for prototyping
  mosaic_width_ = 2 * mosaic_height_;
  mosaic_size_ = cv::Size(mosaic_width_, mosaic_height_);
  fx_ = mosaic_width_ / (2 * M_PI);
  fy_ = mosaic_height_ / M_PI;
  mosaic_img_ = cv::Mat::zeros(mosaic_size_, CV_32FC1);

  // Ground-truth poses for prototyping
  poses_.clear();
  loadPoses();

  // Observation / Measurement function
  var_process_noise_ = 1e-3;
  C_th_ = 0.45; // dataset
  var_R_tracking = 0.17*0.17; // units [C_th]^2, (contrast)
  var_R_mapping = 1e4; // units [1/second]^2, (event rate)

  // Tracking variables
  rot_vec_ = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::randn(rot_vec_, cv::Scalar(0.0), cv::Scalar(1e-5));
  covar_rot_ = cv::Mat::eye(3, 3, CV_64FC1) * 1e-3;

  // Mapping variables
  grad_map_ = cv::Mat::zeros(mosaic_size_, CV_32FC2);
  const float grad_init_variance = 10.f;
  grad_map_covar_ = cv::Mat(mosaic_size_, CV_32FC3, cv::Scalar(grad_init_variance, 0.f, grad_init_variance));
  
  
  // Estimated poses
  VLOG(1)
      << "Set initial pose: ";
  poses_est_.insert(std::pair<ros::Time, Transformation>(poses_.begin()->first, poses_.begin()->second));
  // Print initial time and pose (the pose should be the identity)
  VLOG(1) << "--Estimated pose "
          << ". time = " << poses_est_.begin()->first;
  VLOG(1) << "--T = ";
  VLOG(1) << poses_est_.begin()->second;
  VLOG(1) << "Set initial pose... done!";
}


Mosaic::~Mosaic()
{
  // shut down all publishers
  mosaic_pub_.shutdown();
  time_map_pub_.shutdown();
  mosaic_gradx_pub_.shutdown();
  mosaic_grady_pub_.shutdown();
  mosaic_tracecov_pub_.shutdown();
  pose_pub_.shutdown();
}

/**
* \brief Function to process event messages received by the ROS node
*/
void Mosaic::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  
  // Append events of current message to the queue
  for(const dvs_msgs::Event& ev : msg->events)
    events_.push_back(ev);

  static unsigned long total_event_count = 0;
  total_event_count += msg->events.size();
  VLOG(1) << "Packet # " << packet_number << "  event# " << total_event_count << "  queue_size:" << events_.size();

  if (packet_number == 0)
  {
    // Initialize time map and last rotation map
    time_map_ = cv::Mat(sensor_height_,sensor_width_,CV_64FC1,cv::Scalar(-0.01));
    map_of_last_rotations_ = std::vector<cv::Matx33d>(sensor_width_ * sensor_height_, cv::Matx33d(dNaN));
  }
  packet_number++;

  
  while (num_events_update_ <= events_.size())
  {
    VLOG(1) << "TRACK using ev= " << num_events_update_ << " events.  Queue size()=" << events_.size();

    // Get subset of events
    events_subset_ = std::vector<dvs_msgs::Event> (events_.begin(),
                                                   events_.begin() + num_events_update_);

    // Compute time span of the events
    const ros::Time time_first = events_subset_.front().ts;
    const ros::Time time_last = events_subset_.back().ts;
    const ros::Duration time_dt = time_last - time_first;
    time_packet_ = time_first + time_dt * 0.5;
    VLOG(2) << "MAP: duration [s]= "<< time_dt.toSec();

    //visualization
    if(visualize)
    {
      mosaic_img_vis_ = mosaic_img_.clone();
      image_util::normalize(mosaic_img_vis_, mosaic_img_vis_, 1.);
      cv::cvtColor(mosaic_img_vis_, mosaic_img_vis_, cv::COLOR_GRAY2BGR);
    }

    // Compute ground truth rotation matrix (shared by all events in the batch)
    rotationAt(time_packet_, Rot_gt);

    // initilize rotation vector with ground truth
    if(packet_number<100)
      cv::Rodrigues(Rot_gt, rot_vec_);


    int packet_events_count = 0;
    // Loop through the events
    for (const dvs_msgs::Event& ev : events_subset_)
    {
      // update rotation map
      const int idx = ev.y * sensor_width_ + ev.x;
      const cv::Matx33d Rot_cur;
      cv::Rodrigues(rot_vec_, Rot_cur);
      cv::Matx33d Rot_prev = map_of_last_rotations_.at(idx);
      map_of_last_rotations_[idx] = Rot_cur;
      if (std::isnan(Rot_prev(0, 0)))
      {
        VLOG(3) << "Uninitialized event. Continue";
        continue;
      }

      //if(packet_number>=100)
        processEventForTrack(ev, Rot_prev);
      processEventForMap(ev, Rot_prev);

      ++packet_events_count;

    }


    if(packet_number % num_packet_reconstrct_mosaic_ == 0)
    {
      VLOG(1) << "---- Reconstruct Mosaic ----";
      poisson::reconstructBrightnessFromGradientMap(grad_map_, mosaic_img_);
    }

    publishMap();

    // Debugging
    if (extra_log_debugging)
    {
      if(packet_number>=100)
      {
        cv::Matx31d rot_vec_gt;
        cv::Rodrigues(Rot_gt, rot_vec_gt);
        static std::ofstream ofs("/home/yunfan/work_spaces/master_thesis/bmvc2014/log_rot", std::ofstream::trunc);
        static int count2 = 0;
        ofs << "###########################################" << std::endl;
        ofs << "packet number: " << packet_number << std::endl;
        ofs << "packet count: " << packet_events_count << std::endl;
        ofs << "GT rotation vec: [" << rot_vec_gt(0, 0) << ", " << rot_vec_gt(1, 0) << ", " << rot_vec_gt(2, 0) << std::endl;
        ofs << "rotation vec: " << std::endl;
        ofs << rot_vec_ << std::endl;
        count2++;
        if (count2 == 100)
          ofs.close();
      }
    }

    // Slide
    events_.erase(events_.begin(), events_.begin() + num_events_update_);
  }

}




}
