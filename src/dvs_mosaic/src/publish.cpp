#include <dvs_mosaic/mosaic.h>
#include <dvs_mosaic/image_util.h>
#include <geometry_msgs/PoseStamped.h>
#include <glog/logging.h>

namespace dvs_mosaic
{

/**
* \brief Publish several variables related to the mapping (mosaicing) part
*/
void Mosaic::publishMap()
{
  // Publish the current map state
  VLOG(2) << "publishMap()";

  if ( time_map_pub_.getNumSubscribers() > 0 )
  {
    cv_bridge::CvImage cv_image_time;
    cv_image_time.header.stamp = ros::Time::now();
    cv_image_time.encoding = "mono8";
    image_util::normalize(time_map_, cv_image_time.image, 15.);
    time_map_pub_.publish(cv_image_time.toImageMsg());
  }

  // Various mosaic-related topics
  cv_bridge::CvImage cv_image;
  cv_image.header.stamp = ros::Time::now();
  cv_image.encoding = "bgr8";
  if ( mosaic_pub_.getNumSubscribers() > 0 )
  {
    cv_image.image = pano_ev;
    mosaic_pub_.publish(cv_image.toImageMsg());
  }

  if ( mosaic_gradx_pub_.getNumSubscribers() > 0 ||
       mosaic_grady_pub_.getNumSubscribers() > 0 )
  {
    cv_bridge::CvImage cv_gradx;
    cv_gradx.header.stamp = ros::Time::now();
    cv_gradx.encoding = "mono8";
    cv_bridge::CvImage cv_grady;
    cv_grady.header.stamp = ros::Time::now();
    cv_grady.encoding = "mono8";
    cv::Mat gradxy[2];
    cv::split(grad_map_, gradxy);
    image_util::normalize(gradxy[0], cv_gradx.image, 1);
    image_util::normalize(gradxy[1], cv_grady.image, 1);
    mosaic_gradx_pub_.publish(cv_gradx.toImageMsg());
    mosaic_grady_pub_.publish(cv_grady.toImageMsg());
    }

  if ( mosaic_tracecov_pub_.getNumSubscribers() > 0 )
  {
    // Visualize confidence: trace of the covariance of the gradient map
    // Hints: use cv::split to split a multi-channel array into its channels (images)
    //        call image_util::normalize
    cv_bridge::CvImage cv_cov;
    cv_cov.header.stamp = ros::Time::now();
    cv_cov.encoding = "mono8";
    cv::Mat covabc[3];
    cv::split(grad_map_covar_, covabc);
    image_util::normalize(covabc[0] + covabc[2], cv_cov.image, 1);
    mosaic_tracecov_pub_.publish(cv_cov.toImageMsg());
  }
}


/**
* \brief Publish pose once the tracker has estimated it
*/
void Mosaic::publishPose()
{
  // if (pose_pub_.getNumSubscribers() <= 0)
  //   return;

  // VLOG(1) << "publishPose()";
  // geometry_msgs::PoseStamped pose_msg;
  // // FILL IN ... when tracking part is implemented


  // pose_pub_.publish(pose_msg);
}

}
