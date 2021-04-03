#include <dvs_mosaic/mosaic.h>
#include <math.h>
#include <glog/logging.h>


namespace dvs_mosaic
{

/**
* \brief To speed up, compute 3D point (direction) corresponding to each camera pixel
*/
void Mosaic::precomputeBearingVectors()
{
  VLOG(2) << "Start precomputeBearingVectors";
  for(int y=0; y < sensor_height_; y++)
  {
    for(int x=0; x < sensor_width_; x++)
    {
      cv::Point2d rectified_point = dvs_cam_.rectifyPoint(cv::Point2d(x,y));
      cv::Point3d bearing_vec = dvs_cam_.projectPixelTo3dRay(rectified_point);
      precomputed_bearing_vectors_.push_back(bearing_vec);
    }
  }
  VLOG(2) << "End precomputeBearingVectors " << precomputed_bearing_vectors_.size();
}


/**
* \brief Projects a 3D point according to equirectangular model
* \note Used for 360 degrees panoramic cameras that outputs a full panorama frame
*/
cv::Mat Mosaic::project_EquirectangularProjection(const cv::Point3d &pt_3d, cv::Point2f &pt_on_mosaic, bool calculate_d2d3)
{
  const double x = pt_3d.x;
  const double y = pt_3d.y;
  const double z = pt_3d.z;

  double rho = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
  pt_on_mosaic.x = mosaic_width_ / 2.0 + fx_ * atan2(x, z);
  pt_on_mosaic.y = mosaic_height_ / 2.0 + fy_ * asin(y / rho);

  cv::Mat d2d3 = cv::Mat::zeros(2, 3, CV_64FC1);
  if (calculate_d2d3)
  {
    double temp1 = fx_ / ((1 + pow(x / z, 2)) * z);
    double temp2 = -fy_ / sqrt(1 - pow(y / rho, 2));
    double temp3 = y / pow(rho, 3);
    d2d3.at<double>(0, 0) = temp1;
    d2d3.at<double>(0, 2) = -temp1 * (x / z);
    d2d3.at<double>(1, 0) = temp2 * temp3 * x;
    d2d3.at<double>(1, 1) = temp2 * (temp3 * y - 1 / rho);
    d2d3.at<double>(1, 2) = temp2 * temp3 * z;
  }
  return d2d3;
}

/** \brief Compute linearly interpolated rotation matrix at a specified query time
% \note Additionally, it needs trajectory-like data (timestamps & control poses)
*/
bool Mosaic::rotationAt(const ros::Time& t_query, cv::Matx33d& Rot_interp)
{
  ros::Time t0_, t1_;
  Transformation T0_, T1_;

  // Check if it is between two known poses
  auto it1 = poses_.upper_bound(t_query);
  if(it1 ==  poses_.begin())
  {
    LOG_FIRST_N(WARNING, 5) << "Cannot extrapolate in the past. Requested pose: "
                            << t_query << " but the earliest pose available is at time: "
                            << poses_.begin()->first;
    return false;
  }
  else if(it1 == poses_.end())
  {
    LOG_FIRST_N(WARNING, 5) << "Cannot extrapolate in the future. Requested pose: "
                            << t_query << " but the latest pose available is at time: "
                            << (poses_.rbegin())->first;
    return false;
  }
  else
  {
    auto it0 = std::prev(it1);
    t0_ = (it0)->first;
    T0_ = (it0)->second;
    t1_ = (it1)->first;
    T1_ = (it1)->second;
  }

  // Linear interpolation in SE(3)
  auto T_relative = T0_.inverse() * T1_;
  // Interpolation parameter in [0,1]
  auto delta_t = (t_query - t0_).toSec() / (t1_ - t0_).toSec();
  // Linear interpolation, Lie group formulation
  Transformation T = T0_ * Transformation::exp(delta_t * T_relative.log());

  // Extract rotational part
  auto R = T.getRotationMatrix();

  VLOG(3) << R;

  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
    {
      Rot_interp(i,j) = R(i,j);
    }

  return true;
}

}
