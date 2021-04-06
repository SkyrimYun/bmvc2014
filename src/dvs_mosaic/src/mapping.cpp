#include <dvs_mosaic/mosaic.h>
#include <glog/logging.h>
#include <dvs_mosaic/reconstruction.h>
#include <fstream>


namespace dvs_mosaic
{

/**
* \brief Process each event to refine the mosaic variables (mean and covariance)
*/
void Mosaic::processEventForMap(const dvs_msgs::Event &ev, const cv::Matx33d Rot_prev)
{
  const int idx = ev.y * sensor_width_ + ev.x;

  // Get time of current and last event at the pixel
  const double t_ev = ev.ts.toSec();
  const double t_prev = time_map_.at<double>(ev.y, ev.x);
  time_map_.at<double>(ev.y, ev.x) = t_ev;

  
  if (t_prev < 0)
  {
    VLOG(3) << "Uninitialized pixel. Continue";
    return;
  }

  const double dt_ev = t_ev - t_prev;
  CHECK_GT(dt_ev,0) << "Non-positive dt_ev"; // Two events at same pixel with same timestamp

  const double thres = ev.polarity ? 1.0 : -1.0;

  const cv::Matx33d Rot;
  cv::Rodrigues(rot_vec_, Rot); // convert parameter vector to Rotation

  // Get map point corresponding to current event
  // hint: call project_EquirectangularProjection
  cv::Point3d rotated_bvec = Rot * precomputed_bearing_vectors_.at(idx);
  cv::Point2f pm;
  project_EquirectangularProjection(rotated_bvec, pm);

  //VLOG(0) << "pm: [" << pm.x << ", " << pm.y<<"]";

  // Get map point corresponding to previous event at same pixel
  cv::Point3d rotated_bvec_prev = Rot_prev * precomputed_bearing_vectors_.at(idx);
  cv::Point2f pm_prev;
  project_EquirectangularProjection(rotated_bvec_prev, pm_prev);

  // Get approx optical flow vector (vector v in the paper)
  cv::Point2f flow_vec = (pm - pm_prev) / dt_ev;


  // Extended Kalman Filter (EKF) for the intensity gradient map.
  // Get gradient and covariance at current map point pm
  cv::Vec2f grad_vec = grad_map_.at<cv::Vec2f>(pm);
  cv::Matx21f gm(grad_vec[0], grad_vec[1]);
  
  cv::Vec3f Pg_vec = grad_map_covar_.at<cv::Vec3f>(pm);
  cv::Matx22f Pg(Pg_vec[0], Pg_vec[1], Pg_vec[1], Pg_vec[2]);


    // Compute innovation, measurement matrix and Kalman gain
  float nu_innovation = 1 / dt_ev - (gm(0,0) * flow_vec.x + gm(1, 0) * flow_vec.y) / thres;
  cv::Matx12f dh_dg(flow_vec.x / thres, flow_vec.y / thres);
  float s = (dh_dg * Pg * dh_dg.t())(0,0) + var_R_mapping;
  cv::Matx21f Kalman_gain = (Pg * dh_dg.t());
  Kalman_gain(0, 0) = Kalman_gain(0, 0) / s;
  Kalman_gain(1, 0) = Kalman_gain(1, 0) / s;


  // Update gradient (state) and covariance
  gm += Kalman_gain * nu_innovation;
  Pg -= Kalman_gain * s * Kalman_gain.t();

  // debuging
  if(extra_log_debugging)
  {
    static std::ofstream ofs("/home/yunfan/work_spaces/master_thesis/bmvc2014/log_mapping", std::ofstream::trunc);
    static int count = 0;
    ofs << "###########################################" << std::endl;
    ofs << count++ << std::endl;
    ofs << "points: [" << pm_prev.x << ", " << pm_prev.y << "] -> "
        << "[" << pm.x << ", " << pm.y << "]" << std::endl;
    ofs << "flow_vec: [" << flow_vec.x << ", " << flow_vec.y << "]" << std::endl;
    ofs << "grad: [" << grad_vec[0] << ", " << grad_vec[1] << "] -> "
        << "[" << gm(0, 0) << ", " << gm(1, 0) << "]" << std::endl;
    ofs << "covar: [" << Pg_vec[0] << ", " << Pg_vec[1] << ", " << Pg_vec[2] << "] -> "
        << Pg << std::endl;
    ofs << "nu_innovation: " << nu_innovation << std::endl;
    ofs << "dh/dg: [" << dh_dg(0, 0) << ", " << dh_dg(0, 1) << "]" << std::endl;
    ofs << "s: " << s << std::endl;
    ofs << "Kalman_gain: " << Kalman_gain << std::endl;
    ofs << "gm+: " << Kalman_gain * nu_innovation << std::endl;
    ofs << "pg-: " << Kalman_gain * s * Kalman_gain.t() << std::endl;
    if (count == 200)
      ofs.close();
  }
  


  // Store updated values of grad_map_ and grad_map_covar_ at corresponding pixel
  grad_map_.at<cv::Vec2f>(pm) = cv::Vec2f(gm(0, 0), gm(1, 0));
  grad_map_covar_.at<cv::Vec3f>(pm) = cv::Vec3f(Pg(0, 0), Pg(0, 1), Pg(1, 1));

  // Visualization
  if (visualize)
  {
    // Visualization
    // Get map point corresponding to current event and ground truth rotation
    cv::Point3d rotated_bvec_gt = Rot_gt * precomputed_bearing_vectors_.at(idx);
    cv::Point3d rotated_bvec_est = Rot * precomputed_bearing_vectors_.at(idx);

    cv::Point2f pm_gt;
    cv::Point2f pm_est;

    project_EquirectangularProjection(rotated_bvec_gt, pm_gt);
    project_EquirectangularProjection(rotated_bvec_est, pm_est);
    const int icg = pm_gt.x, irg = pm_gt.y; // integer position
    if (0 <= irg && irg < mosaic_height_ && 0 <= icg && icg < mosaic_width_)
    {
      cv::circle(mosaic_img_vis_, cv::Point(icg, irg), 10, cv::Scalar(0, 255, 0));
    }
    const int ice = pm_est.x, ire = pm_est.y; // integer position
    if (0 <= ire && ire < mosaic_height_ && 0 <= ice && ice < mosaic_width_)
    {
      cv::circle(mosaic_img_vis_, cv::Point(ice, ire), 5, cv::Scalar(0, 0, 255));
    }
  }
}


void Mosaic::reconstuctMosaic()
{
  while(true)
  {
    std::unique_lock<std::mutex> lock(data_lock_);
    reconstruct_.wait(lock);
    VLOG(1) << "---- Reconstruct Mosaic ----";
    poisson::reconstructBrightnessFromGradientMap(grad_map_, mosaic_img_);
    cv::GaussianBlur(mosaic_img_, mosaic_img_, cv::Size(0, 0), 1);
    reconstruct_.notify_one();
  }
}

}
