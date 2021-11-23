# bmvc2014

This project aims to reconstruct Kim's method in BMVC2014 "**Simultaneous Mosaicing and Tracking with an Event Camera**" since the code is not publicly released by the author.

This method presents a 3DOF reconstruction using two decoupled probabilistic filters from a
single hand-held event camera. One Extended Kalman filter (EKF) is responsible for gradient
estimation and another particle filter is for rotation estimation using a random walk model.
The intensity image is reconstructed from the gradient map by Poisson editing. EKF assumes
the rotation from the particle filter is correct, while the particle filter also assumes the intensity
image reconstructed from the gradient map is correct.

Although the original paper uses a partical filter for tracking, we uses an EKF for tracking based on Kim's PhD thesis.
The final preformance of this project does not reach expectation. We believe there are some implementation details in this method that have not been mentioned in the paper. 
We hope we can improve this project in the future.

## Branch
**feature_average_pose_quaternion_average**
  - this is the latest (or final) branch we use. We reconmannd you to check this branch first if you intend to play with the code
  - this branch uses quaternion average for the pose average
  
## Dependencies
This project builds on ROS melodic.
First, check the readme in bmvc2014/src/dvs_mosaic/
Except, the latest code also needs OpenCV and **matplotlib_cpp** for plotting.

## Data
check the ROS bags in bmvc2014/src/dvs_mosaic/data/

## Parameters
parameters in launch file **synth.launch**
- display_accuracy_ : display the real-time accuracy in terminator
- tracker_standalone_ : whether to run tracker alone with GT mosaic image
- use_partial_mosaic_ : whether to run tracker alone with partial GT mosaic image
- average_method_ : 
  - true : smooth by average quaternion
  - false : smooth by average relative quaternion
- average_level_ : take how many adjacent pose for average
- use_grad_thres_ : do not reconstruct intensity on pixel with gradient uncertainty below grad_thres_
- use_polygon_thres_ : only use the pixels inside a polygon which is smaller than current camera frame for tracking
- use_bright_thres_ : do not use pixels on mosaic for tracking if their intensity is below bright_thres_
