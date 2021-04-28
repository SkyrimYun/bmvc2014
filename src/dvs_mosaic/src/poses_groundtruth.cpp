#include <dvs_mosaic/mosaic.h>
#include <fstream>
#include <glog/logging.h>


namespace dvs_mosaic
{

/**
* \brief Load poses from file (ground truth for prototyping the mapping part)
*/
void Mosaic::loadPoses()
{
  std::ifstream input_file;
  // set the appropriate path to the file
  if(new_dataset_)
  {
    input_file.open(ros::package::getPath("dvs_mosaic") + "/data/kim/poses.txt");
    // Open file to read data
    if (input_file.is_open())
    {
      VLOG(2) << "Control poses file opened";

      int count = 0;
      std::string line;
      while (getline(input_file, line))
      {
        std::istringstream stm(line);

        double sec;
        double x, y, z, qx, qy, qz, qw;
        Transformation T0;
        if (stm >> sec >> x >> y >> z >> qx >> qy >> qz >> qw)
        {
          ros::Time t(sec);
          const Eigen::Vector3d position(x, y, z);
          const Eigen::Quaterniond quat(qw, qx, qy, qz);
          Transformation T(position, quat);
          VLOG(2) << t;
          VLOG(2) << T;
          poses_.insert(std::pair<ros::Time, Transformation>(t, T));
        }
        count++;
      }
      VLOG(2) << "count poses = " << count;

      input_file.close();
    }
  }
  else
  {
    input_file.open(ros::package::getPath("dvs_mosaic") + "/data/synth1/poses.txt");
    // Open file to read data
    if (input_file.is_open())
    {
      VLOG(2) << "Control poses file opened";

      int count = 0;
      std::string line;
      while (getline(input_file, line))
      {
        std::istringstream stm(line);

        long sec, nsec;
        double x, y, z, qx, qy, qz, qw;
        Transformation T0;
        if (stm >> sec >> nsec >> x >> y >> z >> qx >> qy >> qz >> qw)
        {
          ros::Time(sec, nsec);
          const Eigen::Vector3d position(x, y, z);
          const Eigen::Quaterniond quat(qw, qx, qy, qz);
          Transformation T(position, quat);
          poses_.insert(std::pair<ros::Time, Transformation>(ros::Time(sec, nsec), T));
        }
        count++;
      }
      VLOG(2) << "count poses = " << count;

      input_file.close();
    }
  }

  

  // Remove offset: pre-multiply by the inverse of the first pose so that
  // the first rotation becomes the identity (and events project in the middle of the mosaic)

  // get the first control pose
  Transformation T0 = (*poses_.begin()).second;

  size_t control_pose_idx = 0u;
  for(auto it : poses_)
  {
    VLOG(3) << "--Control pose #" << control_pose_idx++ << ". time = " << it.first;
    VLOG(3) << "--T = ";
    VLOG(3) << it.second;
    poses_[it.first] = (T0.inverse()) * it.second;
  }

  control_pose_idx = 0u;
  for(auto it : poses_)
  {
    VLOG(3) << "--Control pose #" << control_pose_idx++ << ". time = " << it.first;
    VLOG(3) << "---------------------T normalized = ";
    VLOG(3) << it.second;
  }
}

}
