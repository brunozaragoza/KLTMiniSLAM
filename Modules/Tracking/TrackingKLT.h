/**
* This file is part of Mini-SLAM
*
* Copyright (C) 2021 Juan J. Gómez Rodríguez and Juan D. Tardós, University of Zaragoza.
*
* Mini-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Mini-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with Mini-SLAM.
* If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Author: Juan J. Gómez Rodríguez (jjgomez@unizar.es)
 *
 * Implementation of the MaTrackingpping functionalities of Mini-SLAM. It takes every new image, extracts features
 * and itrs descriptors and computes the camera pose by matching MapPoints and running an only pose optimization.
 *
 * It also takes the decision of whenever we need to insert a new KeyFrame into the map
 */

 #ifndef MINI_SLAM_TRACKING_H
 #define MINI_SLAM_TRACKING_H
 
 #include "Features/Feature.h"
 #include "Features/Descriptor.h"
 
 #include "System/Settings.h"
 
 #include "Tracking/Frame.h"
 #include "Tracking/MonocularMapInitializer.h"
 #include "Matching/lucas_kanade_tracker.h"
 #include "Matching/landmarkstatus.h"
 #include <Visualization/FrameVisualizer.h>
 #include <Visualization/MapVisualizer.h>
 
 #include <sophus/se3.hpp>
 
 #include <opencv2/opencv.hpp>
 
 #include <memory>
 
 class TrackingKLT {
 public:
    TrackingKLT();
    struct Options {
        int klt_window_size = 7;
        int klt_max_level = 4;
        int klt_max_iters = 100;
        float klt_epsilon = 0.001;
        float klt_min_eig_th = 1e-4;
        float klt_min_SSIM = 0.55;

        int images_to_insert_keyframe = 5;

        float radians_per_pixel;
    };

     /*
      * Constructor with the visualizers, the settings and the map
      */
     TrackingKLT(Settings& settings, std::shared_ptr<FrameVisualizer>& visualizer,
              std::shared_ptr<MapVisualizer>& mapVisualizer, std::shared_ptr<Map> map);
 
     /*
      * Performs the tracking for an image. Returns true on success
      */
     bool doTracking(const cv::Mat& im, Sophus::SE3f& Tcw);
     void ExtractFeaturesInFrame(const cv::Mat& im);
     /*
      * Gets the last KeyFrame inserted into the map
      */
     std::shared_ptr<KeyFrame> getLastKeyFrame();
 private:
     LucasKanadeTracker klt_tracker_;
     //Extracts features and descriptors in the current image
     void extractFeatures(const cv::Mat& im);
 
     //Updates MapPoints in the reference frame as the local mapping may have changed them
     void updateLastPose();
 
     //Initializes a new map from monocuar 2 views
     bool MonocularMapInitialization();
 
     //Performs the camera tracking with a constant velocity model
     bool cameraTracking();
 
     //Updates the constan velocity model
     void updateMotionModel();
     void promoteCurrentFrameToKeyFrame();

     int nframesext=0;
 
     //Feature and descriptor extractors
     std::shared_ptr<Feature> featExtractor_;
     std::shared_ptr<Descriptor> descExtractor_;
 
     //Reference and current frame
     Frame currFrame_, prevFrame_;
 
     //Last location a KeyPoint was seen. Only used in the monocular initialization
     std::vector<cv::Point2f> vPrevMatched_;
 
     //Matches between the reference and the current frame
     std::vector<int> vMatches_;
 
     //Tracking status
     enum TrackStatus{
         NOT_INITIALIZED = 0,    //No map is initialized -> perform monocular initialization
         GOOD = 1,               //Map is initialized and track was good -> perform camera tracking
         LOST = 2                //Track was lost -> perform relocalization (to be implemented)
     };
 
     TrackStatus status_;
     bool bFirstIm_=false;         //Flag to check if we have already received an image
     bool bMotionModel_;     //Flag to check if the velocity model is valid
 
     //Monocular map initializer
     std::shared_ptr<MonocularMapInitializer> monoInitializer_;
 
     //SLAM map
     std::shared_ptr<Map> pMap_;
 
     //Visualizers
     std::shared_ptr<FrameVisualizer> visualizer_;
     std::shared_ptr<MapVisualizer> mapVisualizer_;
     cv::Mat currIm_;
     cv::Mat firstim;
     cv::Mat prevIm_;
     Options options_;
     //Constant velocity model
     Sophus::SE3f motionModel_;
 
     //Number of features correctly tracked
     int nFeatTracked_;
 
     //Last KeyFrame inserted
     std::shared_ptr<KeyFrame> pLastKeyFrame_;
     ID nLastKeyFrameId;
 
     //Number of images processed since the last KeyFrame insertion
     int nFramesFromLastKF_;
     bool bInserted;
     int nLastFeatureExtract=0;
     //Settings of the system
     Settings settings_;
 };
 
 
 #endif //MINI_SLAM_TRACKING_H
 