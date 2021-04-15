#include <dvs_mosaic/mosaic.h>
#include <glog/logging.h>
#include <fstream>

namespace dvs_mosaic
{
    void Mosaic::processEventForTrack(const dvs_msgs::Event &ev, const cv::Matx33d Rot_prev)
    {
        static double t_prev = -0.1;
        const double dt = ev.ts.toSec() - t_prev;
        t_prev = ev.ts.toSec();

        cv::Mat covar_pred = covar_rot_ + cv::Mat::eye(3, 3, CV_64FC1) * (var_process_noise_ * dt);

        cv::Matx33d Rot_pred;
        cv::Rodrigues(rot_vec_, Rot_pred);


        // Get map point corresponding to current event and given rotation
        const int idx = ev.y * sensor_width_ + ev.x;

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
        if (use_grad_thres_ && abs(grad_vec[0] + grad_vec[1]) < grad_thres_ )
        {
            VLOG(2) << "!!!!!!!!!!!SKIP POINTS!!!!!!!!!!!!!!!!!!!!";
            skip_count_grad_++;
            return;
        }

        if(use_polygon_thres_ && cv::pointPolygonTest(tracking_polygon_, pm, false)<0)
        {
            VLOG(2) << "!!!!!!!!!!!SKIP POINTS!!!!!!!!!!!!!!!!!!!!";
            skip_count_polygon_++;
            return;
        }

        double predicted_contrast = computePredictedConstrastOfEvent(pm, pm_prev);

        if(std::isnan(predicted_contrast))
        {
            VLOG(2) << "!!!!!!!!!!!SKIP POINTS!!!!!!!!!!!!!!!!!!!!";
            skip_count_bright_++;
            return;
        }

        cv::Mat deriv_pred_contrast;
        computeDeriv(pm, dpm_d3d, rotated_bvec, deriv_pred_contrast);

                
        //double innovation = C_th_ - (ev.polarity ? 1 : -1) * predicted_contrast;
        double innovation = C_th_ - predicted_contrast;
        deriv_pred_contrast = (ev.polarity ? 1 : -1) * deriv_pred_contrast;

        cv::Mat s = deriv_pred_contrast * covar_pred * deriv_pred_contrast.t() + var_R_tracking_;
        cv::Mat kalman_gain = (covar_pred * deriv_pred_contrast.t()) / s;
        rot_vec_ += kalman_gain * innovation;
        covar_rot_ = covar_pred - kalman_gain * s * kalman_gain.t();

        // Debugging
        if (extra_log_debugging)
        {
            static std::ofstream ofs("/home/yunfan/work_spaces/master_thesis/bmvc2014/log", std::ofstream::trunc);
            static int count1 = 0;
            ofs << "###########################################" << std::endl;
            ofs << count1++ << std::endl;
            ofs << "dt*noise: " << dt * var_process_noise_ << std::endl;
            ofs << "predicted contrast: " << predicted_contrast << std::endl;
            ofs << "innovation: " << innovation << std::endl;
            ofs << "gradient: " << deriv_pred_contrast << std::endl;
            if (count1 == 100)
                ofs.close();
        }

       
    }

    void Mosaic::calculatePacketPoly()
    {
        cv::Matx33d Rot_packet;
        cv::Point2f pm_packet_bl, pm_packet_ur, pm_packet_br, pm_packet_ul;
        cv::Rodrigues(rot_vec_, Rot_packet);
        cv::Point3d bl_bvec = Rot_packet * precomputed_bearing_vectors_.front();
        cv::Point3d br_bvec = Rot_packet * precomputed_bearing_vectors_[sensor_bottom_right];
        cv::Point3d ul_bvec = Rot_packet * precomputed_bearing_vectors_[sensor_upper_left];
        cv::Point3d ur_bvec = Rot_packet * precomputed_bearing_vectors_.back();
        project_EquirectangularProjection(bl_bvec, pm_packet_bl);
        project_EquirectangularProjection(br_bvec, pm_packet_br);
        project_EquirectangularProjection(ul_bvec, pm_packet_ul);
        project_EquirectangularProjection(ur_bvec, pm_packet_ur);
        if (pm_packet_bl.x > pm_packet_ur.x)
        {
            pm_packet_ur = cv::Point2f(mosaic_width_, mosaic_height_);
            pm_packet_ul = cv::Point2f(0, mosaic_height_);
            pm_packet_br = cv::Point2f(mosaic_width_, 0);
            pm_packet_bl = cv::Point2f(0, 0);
        }        
        else
        {
            cv::Point2f cent = (pm_packet_ur + pm_packet_bl) / 2;
            pm_packet_ur -= ((pm_packet_ur - cent) * (1 - tracking_area_percent_));
            pm_packet_ul -= ((pm_packet_ul - cent) * (1 - tracking_area_percent_));
            pm_packet_br -= ((pm_packet_br - cent) * (1 - tracking_area_percent_));
            pm_packet_bl -= ((pm_packet_bl - cent) * (1 - tracking_area_percent_));
        }
        // ROI polygon
        tracking_polygon_.clear();
        tracking_polygon_.push_back(cv::Point2i(pm_packet_bl));
        tracking_polygon_.push_back(cv::Point2i(pm_packet_br));
        tracking_polygon_.push_back(cv::Point2i(pm_packet_ur));
        tracking_polygon_.push_back(cv::Point2i(pm_packet_ul));
        //VLOG(1) << "packet point: [" << pm_packet_min.x << ", " << pm_packet_min.y << "] -> [" << pm_packet_max.x << ", " << pm_packet_max.y << "]";
    }
}