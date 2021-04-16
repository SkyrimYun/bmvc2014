#include <dvs_mosaic/mosaic.h>
#include <glog/logging.h>
#include <fstream>


namespace dvs_mosaic
{

/**
* \brief Compute the map's brightness corresponding to an event and rotation
*/
double Mosaic::getMapBrightnessAt(const cv::Point2f &pm, int mode)
{
  // Get map intensity at that map point
  double val = 0;                 // default
  const int ic = pm.x, ir = pm.y; // integer position
  if (1 <= ir && ir < mosaic_height_ - 2 && 1 <= ic && ic < mosaic_width_ - 2)
  {
    // "Nearest neighbor" interpolation
    //val = mosaic_img_.at<float>(ir, ic);

    // Bilinear interpolation
    double dx = pm.x - ic;
    double dy = pm.y - ir;
    switch (mode)
    {
    case 0:
    {
      val += mosaic_img_.at<float>(ir, ic) * (1 - dx) * (1 - dy);
      val += mosaic_img_.at<float>(ir + 1, ic) * (1 - dx) * dy;
      val += mosaic_img_.at<float>(ir, ic + 1) * dx * (1 - dy);
      val += mosaic_img_.at<float>(ir + 1, ic + 1) * dx * dy;
      break;
    }
    case 1:
    {
      val += grad_map_.at<cv::Vec2f>(ir, ic)[0] * (1 - dx) * (1 - dy);
      val += grad_map_.at<cv::Vec2f>(ir + 1, ic)[0] * (1 - dx) * dy;
      val += grad_map_.at<cv::Vec2f>(ir, ic + 1)[0] * dx * (1 - dy);
      val += grad_map_.at<cv::Vec2f>(ir + 1, ic + 1)[0] * dx * dy;
      break;
    }
    case 2:
    {
      val += grad_map_.at<cv::Vec2f>(ir, ic)[1] * (1 - dx) * (1 - dy);
      val += grad_map_.at<cv::Vec2f>(ir + 1, ic)[1] * (1 - dx) * dy;
      val += grad_map_.at<cv::Vec2f>(ir, ic + 1)[1] * dx * (1 - dy);
      val += grad_map_.at<cv::Vec2f>(ir + 1, ic + 1)[1] * dx * dy;
      break;
    }
    default:
      LOG(FATAL) << "Undefined map type";
    }
  }
  else
  {
    //LOG(FATAL) << mode << "boundary out range [" << ic << ", " << ir << "]";
  }
  return val;
}

/**
* \brief Compute the map's brightness increment (contrast) corresponding to an event
* and two rotations (this event's rotation and the previous one)
*/
double Mosaic::computePredictedConstrastOfEvent(
    const cv::Point2f &pm,
    const cv::Point2f &pm_prev)
{
  // Get map intensity at point pm
  const double brightnessM_pm = getMapBrightnessAt(pm, get_mosaic_map);
  // Get map intensity at point pm_prev
  const double brightnessM_pm_prev = getMapBrightnessAt(pm_prev, get_mosaic_map);

  // Compute the prediction of C_th
  const double predicted_contrast = (brightnessM_pm - brightnessM_pm_prev);

  // if (abs(brightnessM_pm)<0.1||abs(brightnessM_pm_prev)<0.1)
  // {
  //   static std::ofstream ofs("/home/yunfan/work_spaces/master_thesis/bmvc2014/bright_val_log", std::ofstream::trunc);
  //   static int count4 = 0;
  //   ofs << "###########################################" << std::endl;
  //   ofs << "packet number: " << packet_number << std::endl;
  //   ofs << count4++ << std::endl;
  //   ofs << "pm: " << std::endl;
  //   ofs << pm << std::endl;
  //   ofs << "pm previous:" << std::endl;
  //   ofs << pm_prev << std::endl;
  //   ofs << "brightness pm: " << brightnessM_pm << std::endl;
  //   ofs << "brightness pm_prev: " << brightnessM_pm_prev << std::endl;
  //   if (count4 == 50000)
  //     ofs.close();
  // }

  VLOG(2) << "predicted_contrast = " << predicted_contrast;

  if (use_bright_thres_ && (abs(brightnessM_pm) < bright_thres_ || abs(brightnessM_pm_prev) < bright_thres_))
    return dNaN;
  else
    return predicted_contrast;
}

/**
 * \brief Compute the predicted contrast (on the mosaic map) corresponding to an event
 * and two given rotations.
 * Compute also its numerical derivative (using forward differences)
 */
void Mosaic::computeDeriv(
    const cv::Point2f pm,
    const cv::Mat dpm_d3d,
    const cv::Point3d rotated_bvec,
    cv::Mat &Jac)
{
  VLOG(2) << "f(x)";
  
  const cv::Matx33d Rot;
  cv::Rodrigues(rot_vec_, Rot); // convert parameter vector to Rotation

  Jac = cv::Mat::zeros(1, 3, CV_64FC1);
 
  // Analytic gradient
  double M_deriv_x = getMapBrightnessAt(pm, get_grad_x);
  double M_deriv_y = getMapBrightnessAt(pm, get_grad_y);
  Jac = M_deriv_x * (dpm_d3d.row(0)).t() + M_deriv_y * (dpm_d3d.row(1)).t();
  cv::Mat v = cv::Mat(rot_vec_);
  double v_norm = cv::norm(v);
  cv::Mat v2skew = (cv::Mat_<double>(3, 3) << 0, -v.at<double>(2, 0), v.at<double>(1, 0), v.at<double>(2, 0), 0, -v.at<double>(0, 0), -v.at<double>(1, 0), v.at<double>(0, 0), 0);
  cv::Mat matrix_factor = (v * v.t() + (cv::Mat(Rot).t() - cv::Mat::eye(3, 3, CV_64FC1)) * v2skew) / pow(v_norm, 2);
  cv::Vec3d rotated_bvec_v(rotated_bvec);
  cv::Vec3d temp1 = rotated_bvec_v.cross(cv::Vec3d(Jac));
  Jac = cv::Mat(temp1).t() * matrix_factor.t();
  
  
}
}
