#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace poisson
{
  void reconstructBrightnessFromGradientMap(const cv::Mat& grad_map,
                                                  cv::Mat& map_reconstructed);
}
