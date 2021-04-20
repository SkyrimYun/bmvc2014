#include <dvs_mosaic/mosaic.h>
#include <glog/logging.h>
#include <fstream>


namespace dvs_mosaic
{

  /**
  * \brief Mapper. Process each event to refine the mosaic variables (mean and covariance)
  * \param ev the current input event
  * \param Rot_prev previous rotation when the previous event reached the same pixel
  */
  void Mosaic::processEventForMap(const dvs_msgs::Event &ev, const cv::Matx33d Rot_prev)
  {
    const int idx = ev.y * sensor_width_ + ev.x;

    // Get time of current and last event at the pixel
    const double t_ev = ev.ts.toSec();
    const double t_prev = time_map_.at<double>(ev.y, ev.x);
    time_map_.at<double>(ev.y, ev.x) = t_ev;

    const double dt_ev = t_ev - t_prev;
    CHECK_GT(dt_ev, 0) << "Non-positive dt_ev"; // Two events at same pixel with same timestamp


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
    cv::Point2f flow_vec = (pm - pm_prev) / (dt_ev + 1e-9);

    // Extended Kalman Filter (EKF) for the intensity gradient map.
    // Get gradient and covariance at current map point pm
    cv::Vec2f grad_vec = grad_map_.at<cv::Vec2f>(pm);
    cv::Matx21f gm(grad_vec[0], grad_vec[1]);

    cv::Vec3f Pg_vec = grad_map_covar_.at<cv::Vec3f>(pm);
    cv::Matx22f Pg(Pg_vec[0], Pg_vec[1], Pg_vec[1], Pg_vec[2]);


    // Compute innovation, measurement matrix and Kalman gain
    cv::Matx21f dh_dg;
    float nu_innovation;
    if(measure_contrast_)
    {
      // Use contrast as measurement function (Gallego, arXiv 2015)
      dh_dg = cv::Matx21f(flow_vec * dt_ev * (ev.polarity ? 1 : -1));
      nu_innovation = C_th_ - (dh_dg.t() * gm)(0, 0);
    }
    else
    {
      // Use the event rate as measurement function (Kim, BMCV 2014)
      dh_dg = cv::Matx21f(flow_vec / (C_th_ * (ev.polarity ? 1 : -1)));
      nu_innovation = 1.0 / dt_ev - (dh_dg.t() * gm)(0, 0);
    }

    cv::Matx21f Pg_dhdg = Pg * dh_dg;
    const float S_covar_innovation = var_R_mapping_ + (dh_dg.t() * Pg_dhdg)(0, 0);
    cv::Matx21f Kalman_gain = (1.f / S_covar_innovation) * Pg_dhdg;

    // Update gradient (state) and covariance
    gm += Kalman_gain * nu_innovation;
    Pg -= Pg_dhdg * Kalman_gain.t();

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
      ofs << "s: " << S_covar_innovation << std::endl;
      ofs << "Kalman_gain: " << Kalman_gain << std::endl;
      ofs << "gm+: " << Kalman_gain * nu_innovation << std::endl;
      ofs << "pg-: " << Kalman_gain * S_covar_innovation * Kalman_gain.t() << std::endl;
      if (count == 200)
        ofs.close();
    }
  

    // Store updated values of grad_map_ and grad_map_covar_ at corresponding pixel
    grad_map_.at<cv::Vec2f>(pm) = cv::Vec2f(gm(0, 0), gm(1, 0));
    grad_map_covar_.at<cv::Vec3f>(pm) = cv::Vec3f(Pg(0, 0), 0.5f*(Pg(0, 1)+Pg(1, 0)), Pg(1, 1));
  }
}
