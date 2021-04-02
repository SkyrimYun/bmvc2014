#include <dvs_mosaic/mosaic.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <glog/logging.h>
#include <camera_info_manager/camera_info_manager.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <opencv2/highgui.hpp> //cv::imwrite


namespace dvs_mosaic
{

Mosaic::Mosaic(ros::NodeHandle & nh, ros::NodeHandle nh_private)
 : nh_(nh)
 , pnh_("~")
{
  // Get parameters
  nh_private.param<int>("Num_ev_map_update", num_events_map_update_, 10000);
  nh_private.param<int>("Num_ev_pose_update", num_events_pose_update_, 200);

  // Set up subscribers
  // FILL IN...
  event_sub_ = nh_.subscribe("events", 0, &Mosaic::eventsCallback, this);


  // Set up publishers
  image_transport::ImageTransport it_(nh_);
  // FILL IN...
  // my topics names: time_map, mosaic, mosaic_gx, mosaic_gy, mosaic_trace_cov,
  mosaic_pub_ = it_.advertise("mosaic", 1);
  time_map_pub_ = it_.advertise("time_map", 1);
  mosaic_gradx_pub_ = it_.advertise("mosaic_gx", 1);
  mosaic_grady_pub_ = it_.advertise("mosaic_gy", 1);
  mosaic_tracecov_pub_ = it_.advertise("mosaic_trace_cov", 1);

  pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("mosaic_pose", 1);

  // Dynamic reconfigure
  dynamic_reconfigure_callback_ = boost::bind(&Mosaic::reconfigureCallback, this, _1, _2);
  server_.reset(new dynamic_reconfigure::Server<dvs_mosaic::dvs_mosaicConfig>(nh_private));
  server_->setCallback(dynamic_reconfigure_callback_);

  // Event processing in batches / packets
  idx_first_ev_map_ = 0;   // Index of first event of processing window
  time_packet_ = ros::Time(0);

  // Camera information
  // This yaml file is provided in folder data/synth1/
  std::string cam_name("DVS-synthetic"); // yaml file should be in placed in /home/ggb/.ros/camera_info
  camera_info_manager::CameraInfoManager cam_info (nh_, cam_name);
  dvs_cam_.fromCameraInfo(cam_info.getCameraInfo());
  const cv::Size sensor_resolution = dvs_cam_.fullResolution();
  // FILL IN ...
  // Set sensor_width_, sensor_height_ and precompute bearing vectors
  sensor_width_ = sensor_resolution.width;
  sensor_height_ = sensor_resolution.height;
  precomputeBearingVectors();

  // Mosaic size (in pixels)
  mosaic_height_ = 1024; // 512 or 256 for prototyping
  mosaic_width_ = 2 * mosaic_height_;
  // FILL IN ...
  // Set mosaic_size_, fx_ and fy_ and Initialize mosaic_img_ (if needed)
  mosaic_size_ = cv::Size(mosaic_width_, mosaic_height_);
  fx_ = mosaic_width_ / (2 * M_PI);
  fy_ = mosaic_height_ / M_PI;
  mosaic_img_ = cv::Mat::zeros(mosaic_size_, CV_32FC1);
  mosaic_img_save_ = cv::Mat::zeros(mosaic_size_, CV_32FC1);

  // Observation / Measurement function
  C_th_ = 0.45; // dataset
  measure_contrast_ = false;
  if (measure_contrast_)
    var_R_ = 0.17*0.17; // units [C_th]^2, (contrast)
  else
    var_R_ = 1e4; // units [1/second]^2, (event rate)

  // Mapping variables
  map_of_last_rotations_.resize(sensor_width_*sensor_height_); // for speed-up
  grad_map_ = cv::Mat::zeros(mosaic_size_, CV_32FC2);
  const float grad_init_variance = 10.f;
  grad_map_covar_ = cv::Mat(mosaic_size_, CV_32FC3, cv::Scalar(grad_init_variance,0.f,grad_init_variance));

  // Ground-truth poses for prototyping
  poses_.clear();
  loadPoses();
}


Mosaic::~Mosaic()
{
  // FILL IN ...
  // shut down all publishers
  mosaic_pub_.shutdown();
  time_map_pub_.shutdown();
  mosaic_gradx_pub_.shutdown();
  mosaic_grady_pub_.shutdown();
  mosaic_tracecov_pub_.shutdown();
  pose_pub_.shutdown();

  // save binary image
  std::string filename = "/home/yunfan/work_spaces/EventVision/exe7/mosaic";
  // cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  // fs << "mosaic recons map" << mosaic_img_save_;

  const char *filenamechar = filename.c_str();
  FILE *fpw = fopen(filenamechar, "wb"); //如果没有则创建，如果存在则从头开始写
  int cols = mosaic_img_save_.cols;  //四个字节存 列数
  int rows = mosaic_img_save_.rows; //四个字节存 行数
  fwrite(&cols, sizeof(int), 1, fpw);
  fwrite(&rows, sizeof(int), 1, fpw);
  float *dp = (float *)mosaic_img_save_.data;
  int res = fwrite(dp, sizeof(float), cols*rows, fpw);
  VLOG(1)<< "Write " << res << " blocks to Mosaic Map";
  fclose(fpw);

  cv::imwrite("/home/yunfan/work_spaces/EventVision/exe7/mosaic.png", mosaic_img_save_);
}

/**
* \brief Function to process event messages received by the ROS node
*/
void Mosaic::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  // Assume events are sorted in time and event messages come in correct order.
  // Read events, split the events into packets of constant number of events,...

  // Append events of current message to the queue
  for(const dvs_msgs::Event& ev : msg->events)
    events_.push_back(ev);

  static unsigned int packet_number = 0;
  static unsigned long total_event_count = 0;
  total_event_count += msg->events.size();
  VLOG(1) << "Packet # " << packet_number << "  event# " << total_event_count << "  queue_size:" << events_.size();

  if (packet_number == 0)
  {
    // Initialize time map
    time_map_ = cv::Mat(sensor_height_,sensor_width_,CV_64FC1,cv::Scalar(-0.01));
    // Let negative time indicate un-initialized pixels
  }
  packet_number++;

  // -----------------------------------------------------------------
  // Call pose tracker (if time allows)


  // -----------------------------------------------------------------
  // Call mapper (EKF + Poisson)
  while (idx_first_ev_map_ + num_events_map_update_ <= events_.size())
  {
    VLOG(1) << "MAP ev= " << idx_first_ev_map_ << " -- "
            << idx_first_ev_map_ + num_events_map_update_ << ", events_.size()=" << events_.size();
    // Get subset of events
    events_subset_ = std::vector<dvs_msgs::Event> (events_.begin() + idx_first_ev_map_,
                                                   events_.begin() + idx_first_ev_map_ + num_events_map_update_);

    // Compute time span of the events
    ros::Time time_first = events_subset_.front().ts;
    ros::Time time_last = events_subset_.back().ts;
    ros::Duration time_dt = time_last - time_first;
    time_packet_ = time_first + time_dt * 0.5;
    VLOG(2) << "MAP: duration [s]= "<< time_dt.toSec();


    // Call the mapper

    // Split into smaller subsets (about 200 events to share rotation matrix)
    // Share the same rotation matrix for all events in the batch
    // because it is expensive to compute this rotation matrix
    const int num_events_share_Rot = 300;
    int idx_first_ev_batch = 0;
    //static int iCount = 0;
    while ( idx_first_ev_batch < events_subset_.size() )
    {
      int num_ev_batch = std::min(num_events_share_Rot, int(events_subset_.size())-idx_first_ev_batch);
      VLOG(2) << "MAP: idx_first_ev_batch = " << idx_first_ev_batch << "  num_ev_batch = " << num_ev_batch;

      // Select batch of events that will share the same rotation matrix
      std::vector<dvs_msgs::Event> events_batch =
        std::vector<dvs_msgs::Event>(events_subset_.begin() + idx_first_ev_batch,
                                     events_subset_.begin() + idx_first_ev_batch + num_ev_batch);

      // Get (interpolated) rotation
      // Compute interpolation time
      ros::Time t_first = events_batch.front().ts;
      ros::Time t_last = events_batch.back().ts;
      ros::Duration t_dt = t_last - t_first;
      ros::Time t_batch = t_first + t_dt * 0.5;
      VLOG(2) << "MAP: t = " << t_batch.toSec() << ". Batch duration [s] = " << t_dt.toSec();
      // Compute rotation matrix, shared for all events in the batch
      cv::Matx33d Rot;
      rotationAt(t_batch, Rot);

      // Loop through the events
      //cv::Mat pano_ev = cv::Mat::zeros(mosaic_size_, CV_32FC1);
      for (const dvs_msgs::Event& ev : events_batch)
      {
        // Get time of current and last event at the pixel
        const double t_ev = ev.ts.toSec();
        double t_prev; 
        // FILL IN: get it from time_map_
        t_prev = time_map_.at<double>(ev.y, ev.x);
        // FILL IN: Update time map
        time_map_.at<double>(ev.y, ev.x) = t_ev;

        // Get last rotation at the event
        const int idx = ev.y*sensor_width_ + ev.x; 
        cv::Matx33d Rot_prev; 
        // FILL IN get it from map_of_last_rotations_
        Rot_prev = map_of_last_rotations_.at(idx);
        // FILL IN update (prepare for next iteration)
        map_of_last_rotations_.at(idx) = Rot;

        /*
        // Example of plotting events on mosaic
        // Get map point corresponding to current event
        cv::Point3d rotated_bvec = Rot * precomputed_bearing_vectors_.at(idx);
        cv::Point2f pm;
        project_EquirectangularProjection(rotated_bvec, pm);
        const int ic = pm.x, ir = pm.y; // integer position
        if (0 <= ir && ir < mosaic_height_ && 0 <= ic && ic < mosaic_width_)
        {
          pano_ev.at<float>(ir,ic) = (ev.polarity ? 1. : -1.);
        }
        */

        if (t_prev < 0)
        {
          VLOG(3) << "Uninitialized pixel. Continue";
          continue;
        }
        processEventForMap(ev, t_ev, t_prev, Rot, Rot_prev);
      }
      idx_first_ev_batch += num_ev_batch;
      //iCount++;

      /*
      // Example of saving images (for prototyping / debugging)
      std::stringstream ss;
      cv::Mat out_img;
      ss.str(std::string());
      ss << "/tmp/pano_ev_" << std::setfill('0') << std::setw(8) << iCount << ".png";
      cv::normalize(pano_ev, out_img, 0, 255.0, cv::NORM_MINMAX, CV_32FC1);
      cv::imwrite(ss.str(), out_img );
      */
    }

    publishMap();
    idx_first_ev_map_ += num_events_map_update_;

    // Slide
    events_.erase(events_.begin(), events_.begin() + num_events_map_update_);
    idx_first_ev_map_ -= num_events_map_update_;
    VLOG(1) << "-- idx_first_ev_map_ = " << idx_first_ev_map_;
  }

}


void Mosaic::reconfigureCallback(dvs_mosaic::dvs_mosaicConfig &config, uint32_t level)
/**
* \brief Load dynamic parameters
*/
{
  num_events_map_update_ = config.Num_ev_map_update;
  num_events_pose_update_ = config.Num_ev_pose_update;
}

}
