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

  Mosaic::Mosaic(ros::NodeHandle &nh, ros::NodeHandle nh_private)
      : nh_(nh), pnh_("~")
  {
    nh_private.param<int>("num_events_update_", num_events_update_, 1000);
    nh_private.param<int>("num_packet_reconstrct_mosaic_", num_packet_reconstrct_mosaic_, 100);
    nh_private.param<int>("mosaic_height_", mosaic_height_, 512); // 1024,512,256
    nh_private.param<double>("var_process_noise_", var_process_noise_, 1e-4); // if input mosaic is from Ex7; use 1e-4; if input mosaic is from matlab, use 1e-3
    nh_private.param<double>("var_R_tracking_", var_R_tracking_, 0.0289);
    nh_private.param<double>("var_R_mapping_", var_R_mapping_, 1e4);
    nh_private.param<int>("init_packet_num_", init_packet_num_, 300);
    nh_private.param<double>("gaussian_blur_sigma_", gaussian_blur_sigma_, 2);
    nh_private.param<bool>("use_gaussian_blur_", use_gaussian_blur_, true);
    nh_private.param<bool>("tracker_standalone_", tracker_standalone_, false);
    nh_private.param<bool>("use_grad_thres_", use_grad_thres_, true);
    nh_private.param<double>("grad_thres_", grad_thres_, 1);
    nh_private.param<bool>("use_polygon_thres_", use_polygon_thres_, true);
    nh_private.param<double>("tracking_area_percent_", tracking_area_percent_, 0.75);
    nh_private.param<bool>("use_bright_thres_", use_bright_thres_, true);
    nh_private.param<double>("bright_thres_", bright_thres_, 0.15);

    // Set up subscribers
    event_sub_ = nh_.subscribe("events", 0, &Mosaic::eventsCallback, this);
    // set queue_size to 0 to avoid discarding messages (for correctness).

    // Set up publishers
    image_transport::ImageTransport it_(nh_);
    time_map_pub_ = it_.advertise("time_map", 1);
    mosaic_pub_ = it_.advertise("mosaic", 1);
    mosaic_gradx_pub_ = it_.advertise("mosaic_gx", 1);
    mosaic_grady_pub_ = it_.advertise("mosaic_gy", 1);
    mosaic_tracecov_pub_ = it_.advertise("mosaic_trace_cov", 1);
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("mosaic_pose", 1);
    pose_cop_pub_ = it_.advertise("pose_compare", 1);

    // Event processing in batches / packets
    time_packet_ = ros::Time(0);

    // Camera information
    std::string cam_name("DVS-synthetic"); // yaml file should be in /home/ggb/.ros/camera_info
    camera_info_manager::CameraInfoManager cam_info(nh_, cam_name);
    dvs_cam_.fromCameraInfo(cam_info.getCameraInfo());
    const cv::Size sensor_resolution = dvs_cam_.fullResolution();
    sensor_width_ = sensor_resolution.width;
    sensor_height_ = sensor_resolution.height;
    sensor_bottom_right = sensor_width_ - 1;
    sensor_upper_left = sensor_width_ * (sensor_height_ - 1);
    precomputeBearingVectors();

    // Mosaic size (in pixels)
    mosaic_width_ = 2 * mosaic_height_;
    mosaic_size_ = cv::Size(mosaic_width_, mosaic_height_);
    fx_ = static_cast<float>(mosaic_width_) / (2. * M_PI);
    fy_ = static_cast<float>(mosaic_height_) / M_PI;
    mosaic_img_ = cv::Mat::zeros(mosaic_size_, CV_32FC1);

    // Ground-truth poses for prototyping
    poses_.clear();
    loadPoses();

  
    // Observation / Measurement function
    C_th_ = 0.45;                 // dataset

    // Initialize tracker's state and covariance
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
    pose_covar_est_.push_back(sqrt(cv::sum(covar_rot_ * cv::Mat::eye(3, 3, CV_64FC1))[0]) * 180 / M_PI);

    // Print initial time and pose (the pose should be the identity)
    VLOG(1) << "--Estimated pose "
            << ". time = " << poses_est_.begin()->first;
    VLOG(1) << "--T = ";
    VLOG(1) << poses_est_.begin()->second;
    VLOG(1) << "Set initial pose... done!";

    VLOG(1) << "var_process_noise_: " << var_process_noise_;
    VLOG(1) << "var_R_tracking_: " << var_R_tracking_;
    VLOG(1) << "var_R_mapping_: " << var_R_mapping_;
    VLOG(1) << "Tracker works alone? " << (tracker_standalone_ ? "true" : "false");
    VLOG(1) << "Apply Gradient Threshold? " << (use_grad_thres_ ? "true" : "false;") << " Threshold: " << grad_thres_;
    VLOG(1) << "Apply Polygon Threshold? " << (use_polygon_thres_ ? "true" : "false;") << " Tracking area (percent): " << tracking_area_percent_;
    VLOG(1) << "Apply Brightness Threshold? " << (use_bright_thres_ ? "true" : "false;") << " Threshold: " << bright_thres_;
    VLOG(1) << "Apply Gaussian Blur to reconsturcted map? " << (use_gaussian_blur_ ? "true" : "false;") << " sigma: " << gaussian_blur_sigma_;

    if (tracker_standalone_)
    {
      // Load mosaic image from the result of the mapping part
      // FILE *pFile;
      // pFile = fopen("/home/yunfan/work_spaces/master_thesis/bmvc2014/src/dvs_mosaic/data/mosaic_image.bin", "rb");
      // // read image size from file
      // int sizeImg[2];
      // size_t res;
      // res = fread(sizeImg, 2, sizeof(int), pFile);
      // CHECK_EQ(mosaic_size_.width, sizeImg[0]) << "Mosaic sizes differ";
      // CHECK_EQ(mosaic_size_.height, sizeImg[1]) << "Mosaic sizes differ";
      // // read image data from file
      // mosaic_img_ = cv::Mat::zeros(mosaic_size_, CV_32FC1);
      // res = fread(mosaic_img_.data, sizeImg[0] * sizeImg[1], sizeof(float), pFile);
      // fclose(pFile);
      cv::FileStorage fr1("/home/yunfan/work_spaces/master_thesis/bmvc2014/src/dvs_mosaic/data/mosaic.yml", cv::FileStorage::READ);
      fr1["mosaic map"] >> mosaic_img_;

      // Compute derivate of the map
      cv::Mat grad_map_x, grad_map_y;
      cv::Mat kernel = 0.5 * (cv::Mat_<double>(1, 3) << -1, 0, 1);
      cv::filter2D(mosaic_img_, grad_map_x, -1, kernel);
      cv::filter2D(mosaic_img_, grad_map_y, -1, kernel.t());

      VLOG(1) << "gradient kernel: " << kernel;
      std::vector<cv::Mat> channels;
      channels.emplace_back(grad_map_x);
      channels.emplace_back(grad_map_y);
      cv::merge(channels, grad_map_);

      // Load reconstructed image for visualization
      cv::FileStorage fr2("/home/yunfan/work_spaces/master_thesis/bmvc2014/src/dvs_mosaic/data/mosaic_recons.yml", cv::FileStorage::READ);
      fr2["mosaic recons map"] >> mosaic_img_recons_;
     
    }
  }

  Mosaic::~Mosaic()
  {
    time_map_pub_.shutdown();
    mosaic_pub_.shutdown();
    mosaic_gradx_pub_.shutdown();
    mosaic_grady_pub_.shutdown();
    mosaic_tracecov_pub_.shutdown();
    pose_pub_.shutdown();
    pose_cop_pub_.shutdown();

    // std::vector<double> times_gt;
    // std::vector<double> a1_gt;
    // std::vector<double> a2_gt;
    // std::vector<double> a3_gt;
    // for (auto &p : poses_)
    // {
    //   times_gt.push_back(p.first.toSec());
    //   Eigen::AngleAxisd rot_vec_gt(p.second.getEigenQuaternion());
    //   a1_gt.push_back(rot_vec_gt.angle() * rot_vec_gt.axis()[0] * 180 / M_PI);
    //   a2_gt.push_back(rot_vec_gt.angle() * rot_vec_gt.axis()[1] * 180 / M_PI);
    //   a3_gt.push_back(rot_vec_gt.angle() * rot_vec_gt.axis()[2] * 180 / M_PI);
    // }

    // matplotlibcpp::figure();
    // matplotlibcpp::named_plot("true theta_1", times_gt, a1_gt, "r--");
    // matplotlibcpp::named_plot("true theta_2", times_gt, a2_gt, "b--");
    // matplotlibcpp::named_plot("true theta_3", times_gt, a3_gt, "y--");

    // std::vector<double> times_est;
    // std::vector<double> a1_est;
    // std::vector<double> a2_est;
    // std::vector<double> a3_est;
    // for (auto &p : poses_est_)
    // {
    //   times_est.push_back(p.first.toSec());
    //   Eigen::AngleAxisd rot_vec_gt(p.second.getEigenQuaternion());
    //   a1_est.push_back(rot_vec_gt.angle() * rot_vec_gt.axis()[0] * 180 / M_PI);
    //   a2_est.push_back(rot_vec_gt.angle() * rot_vec_gt.axis()[1] * 180 / M_PI);
    //   a3_est.push_back(rot_vec_gt.angle() * rot_vec_gt.axis()[2] * 180 / M_PI);
    // }

    // matplotlibcpp::named_plot("esti theta_1", times_est, a1_est, "r");
    // matplotlibcpp::named_plot("esti theta_2", times_est, a2_est, "b");
    // matplotlibcpp::named_plot("esti theta_3", times_est, a3_est, "y");

    // matplotlibcpp::xlabel("time");
    // matplotlibcpp::ylabel("angle [deg]");
    // matplotlibcpp::title("wrapped angles vs time");
    // matplotlibcpp::legend();
    // matplotlibcpp::save("/home/yunfan/Pictures/tracker_4_14.png");
    // matplotlibcpp::show();

    // matplotlibcpp::figure();
    // matplotlibcpp::plot(times_est, pose_covar_est_, "b");
    // matplotlibcpp::xlabel("time");
    // matplotlibcpp::ylabel("[deg]");
    // matplotlibcpp::title("sqrt(Trace of the state covariance)");
    // matplotlibcpp::save("/home/yunfan/Pictures/tracker_covar_4_14.png");
    // matplotlibcpp::show();

    // // save binary image
    // std::string filename1 = "/home/yunfan/work_spaces/master_thesis/bmvc2014/mosaic.yml";
    // cv::FileStorage fs1(filename1, cv::FileStorage::WRITE);
    // fs1 << "mosaic map" << mosaic_img_;
    // std::string filename2 = "/home/yunfan/work_spaces/master_thesis/bmvc2014/mosaic_recons.yml";
    // cv::FileStorage fs2(filename2, cv::FileStorage::WRITE);
    // fs2 << "mosaic recons map" << pano_ev;
  }

  /**
  * \brief Function to process event messages received by the ROS node
  */
  void Mosaic::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
  {

    // Append events of current message to the queue
    for (const dvs_msgs::Event &ev : msg->events)
      events_.push_back(ev);

    static unsigned long total_event_count = 0;
    total_event_count += msg->events.size();

    if (packet_number == 0)
    {
      // Initialize time map and last rotation map
      time_map_ = cv::Mat(sensor_height_, sensor_width_, CV_64FC1, cv::Scalar(-0.01));
      map_of_last_rotations_ = std::vector<cv::Matx33d>(sensor_width_ * sensor_height_, cv::Matx33d(dNaN));
    }

    // Multiple calls to the tracker to consume the events in one message
    while (num_events_update_ <= events_.size())
    {
      VLOG(1) << "Packet # " << packet_number << "  event# " << total_event_count;
      packet_number++;

      // Get subset of events
      events_subset_ = std::vector<dvs_msgs::Event>(events_.begin(),
                                                    events_.begin() + num_events_update_);

      // Compute time span of the events
      const ros::Time time_first = events_subset_.front().ts;
      const ros::Time time_last = events_subset_.back().ts;
      const ros::Duration time_dt = time_last - time_first;
      time_packet_ = time_first + time_dt * 0.5;
      VLOG(2) << "TRACK: t = " << time_packet_.toSec() << ". Duration [s] = " << time_dt.toSec();

      // visualization
      if (visualize)
      {
        if(tracker_standalone_)
          pano_ev = mosaic_img_recons_.clone();
        else
        {
          pano_ev = mosaic_img_.clone();
          image_util::normalize(pano_ev, pano_ev, 1.);
          cv::cvtColor(pano_ev, pano_ev, cv::COLOR_GRAY2BGR);
        }
      }

      // Compute ground truth rotation matrix (shared by all events in the batch)
      rotationAt(time_packet_, Rot_gt); // Ground truth pose
      cv::Matx33d Rot_interp;
      cv::Rodrigues(rot_vec_, Rot_interp);
      Eigen::Matrix3d R_eigen;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
          R_eigen(i, j) = Rot_interp(i, j);
        }
      poses_est_.insert({time_packet_, Transformation(Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond(R_eigen))});
      pose_covar_est_.push_back(sqrt(cv::sum(covar_rot_ * cv::Mat::eye(3, 3, CV_64FC1))[0]) * 180 / M_PI);

      // initilize rotation vector with ground truth
      if (packet_number < init_packet_num_ )
        cv::Rodrigues(Rot_gt, rot_vec_);

      calculatePacketPoly();

      // packet
      cv::Mat inno_packet = cv::Mat(num_events_update_, 1, CV_64FC1);
      cv::Mat jacb_packet = cv::Mat(num_events_update_, 3, CV_64FC1);
      std::vector<int> idxs;

      // EKF propagation equations for state and covariance
      const double dt = time_packet_.toSec() - t_prev;
      t_prev = time_packet_.toSec();
      int packet_events_count = 0;

      cv::Mat rot_pred = rot_vec_.clone();
      cv::Mat covar_pred = covar_rot_ + cv::Mat::eye(3, 3, CV_64FC1) * (var_process_noise_ * dt);
      cv::Matx33d Rot_pred;
      cv::Rodrigues(rot_pred, Rot_pred);

      // EKF propagation equations for state and covariance
      skip_count_polygon_ = 0;
      skip_count_grad_ = 0;
      skip_count_bright_ = 0;
      // Loop through the events
      for (const dvs_msgs::Event &ev : events_subset_)
      {

        // update rotation map
        const int idx = ev.y * sensor_width_ + ev.x;
        idxs.push_back(idx);
        cv::Matx33d Rot_prev = map_of_last_rotations_.at(idx);


        if (std::isnan(Rot_prev(0, 0)))
        {
          VLOG(3) << "Uninitialized event. Continue";
          map_of_last_rotations_[idx] = Rot_pred;

          continue;
        }

        // Visualization
        if (visualize)
        {
          // Visualization
          // Get map point corresponding to current event and ground truth rotation
          cv::Point3d rotated_bvec_gt = Rot_gt * precomputed_bearing_vectors_.at(idx);
          cv::Point3d rotated_bvec_est = Rot_pred * precomputed_bearing_vectors_.at(idx);

          cv::Point2f pm_gt;
          cv::Point2f pm_est;

          project_EquirectangularProjection(rotated_bvec_gt, pm_gt);
          project_EquirectangularProjection(rotated_bvec_est, pm_est);
          const int icg = pm_gt.x, irg = pm_gt.y; // integer position

          // draw ROI polygon
          cv::polylines(pano_ev, tracking_polygon_, true, cv::Scalar{255, 0, 0}, 2);

          if (0 <= irg && irg < mosaic_height_ && 0 <= icg && icg < mosaic_width_)
          {
            cv::circle(pano_ev, cv::Point(icg, irg), 10, cv::Scalar(0, 255, 0));
          }
          const int ice = pm_est.x, ire = pm_est.y; // integer position
          if (0 <= ire && ire < mosaic_height_ && 0 <= ice && ice < mosaic_width_)
          {
            cv::circle(pano_ev, cv::Point(ice, ire), 5, cv::Scalar(0, 0, 255));
          }
        }

        cv::Point3d rotated_bvec = Rot_pred * precomputed_bearing_vectors_.at(idx);
        cv::Point2f pm;
        cv::Mat dpm_d3d = project_EquirectangularProjection(rotated_bvec, pm, true);

        cv::Point3d rotated_bvec_prev = Rot_prev * precomputed_bearing_vectors_.at(idx);
        cv::Point2f pm_prev;
        project_EquirectangularProjection(rotated_bvec_prev, pm_prev);


        cv::Vec2f grad_vec = grad_map_.at<cv::Vec2f>(pm);

        if (use_grad_thres_ && abs(grad_vec[0] + grad_vec[1]) < grad_thres_)
        {
          VLOG(2) << "!!!!!!!!!!!SKIP POINTS!!!!!!!!!!!!!!!!!!!!";
          skip_count_grad_++;
          continue;
        }


        if (use_polygon_thres_ && !tracking_polygon_.empty() && cv::pointPolygonTest(tracking_polygon_, pm, false) < 0)
        {
          VLOG(2) << "!!!!!!!!!!!SKIP POINTS!!!!!!!!!!!!!!!!!!!!";
          skip_count_polygon_++;
          continue;
        }

        double predicted_contrast = computePredictedConstrastOfEvent(pm, pm_prev);

        if (std::isnan(predicted_contrast))
        {
          VLOG(2) << "!!!!!!!!!!!SKIP POINTS!!!!!!!!!!!!!!!!!!!!";
          skip_count_bright_++;
          continue;
        }

        cv::Mat deriv_pred_contrast;
        computeDeriv(pm, dpm_d3d, rotated_bvec, deriv_pred_contrast);

        //double innovation = C_th_ - (ev.polarity ? 1 : -1) * predicted_contrast;
        double innovation = C_th_ - predicted_contrast;
        deriv_pred_contrast = (ev.polarity ? 1 : -1) * deriv_pred_contrast;

        inno_packet.row(packet_events_count) = innovation;
        jacb_packet.row(packet_events_count) = deriv_pred_contrast + 0;

       
        if(!tracker_standalone_)
          processEventForMap(ev, Rot_prev);

        ++packet_events_count;
      }

      if(packet_events_count>0)
      {
        // Implement EKF traker correction-step equations. Matrix form (by stacking measurement equations)
        jacb_packet = jacb_packet(cv::Rect(0, 0, 3, packet_events_count)).clone();
        inno_packet = inno_packet(cv::Rect(0, 0, 1, packet_events_count)).clone();
        const double ivar_meas_noise = 1 / var_R_tracking_;
        cv::Mat S_inv_packet = -(ivar_meas_noise * ivar_meas_noise) * jacb_packet * (covar_pred.inv() + ivar_meas_noise * (jacb_packet.t() * jacb_packet)).inv() * jacb_packet.t();
        S_inv_packet += (cv::Mat::eye(S_inv_packet.rows, S_inv_packet.cols, CV_64FC1) * ivar_meas_noise);
        cv::Mat kalman_gain_packet = covar_pred * jacb_packet.t() * S_inv_packet;
        rot_vec_ = rot_pred + kalman_gain_packet * inno_packet;
        covar_rot_ = covar_pred - kalman_gain_packet * jacb_packet * covar_pred;
      }
     

      cv::Matx33d Rot_cur;
      cv::Rodrigues(rot_vec_, Rot_cur);
      for (int n : idxs)
        map_of_last_rotations_[n] = Rot_cur;


      if (packet_number % num_packet_reconstrct_mosaic_ == 0 && !tracker_standalone_)
      {
        VLOG(1) << "---- Reconstruct Mosaic ----";
        poisson::reconstructBrightnessFromGradientMap(grad_map_, mosaic_img_);
        if(use_gaussian_blur_)
          cv::GaussianBlur(mosaic_img_, mosaic_img_, cv::Size(0, 0), gaussian_blur_sigma_);
      }


      if(!tracker_standalone_)
        publishMap();

      //calculatePacketPoly();
      VLOG(1) << "total events processed: " << packet_events_count;
      VLOG(1) << "skip count gradient: " << skip_count_grad_;
      VLOG(1) << "skip count polygon: " << skip_count_polygon_;
      VLOG(1) << "skip count brightness: " << skip_count_bright_;

      // Debugging
      if (extra_log_debugging)
      {
        cv::Matx31d rot_vec_gt;
        cv::Rodrigues(Rot_gt, rot_vec_gt);
        static std::ofstream ofs("/home/yunfan/work_spaces/master_thesis/bmvc2014/log_rot", std::ofstream::trunc);
        static int count2 = 0;
        ofs << "###########################################" << std::endl;
        ofs << "packet number: " << packet_number << std::endl;
        ofs << "packet count: " << packet_events_count << std::endl;
        // ofs << "kalman gain packet: " << std::endl;
        // ofs << kalman_gain_packet << std::endl;
        // ofs << "inno packet: " << inno_packet.rows << ", " << inno_packet.cols << std::endl;
        // ofs << inno_packet << std::endl;
        // ofs << "jacb packet: " << jacb_packet.rows << ", " << jacb_packet.cols << std::endl;
        // ofs << jacb_packet << std::endl;
        ofs << "GT rotation vec: [" << rot_vec_gt(0, 0) << ", " << rot_vec_gt(1, 0) << ", " << rot_vec_gt(2, 0) << std::endl;
        ofs << "rotation vec: " << std::endl;
        ofs << rot_vec_ << std::endl;
        count2++;
        if (count2 == 100)
          ofs.close();
      }

      //Publish comparison image
      if(visualize && tracker_standalone_)
      {
        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = ros::Time::now();
        cv_image.encoding = "bgr8";
        cv_image.image = pano_ev;
        if (pose_cop_pub_.getNumSubscribers() > 0)
          pose_cop_pub_.publish(cv_image.toImageMsg());
      }

      // Slide
      events_.erase(events_.begin(), events_.begin() + num_events_update_);
    }
  }

}
