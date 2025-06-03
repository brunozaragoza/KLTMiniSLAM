/*
 * This file is part of NR-SLAM
 *
 * Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * NR-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 #ifndef MINISLAM_MONOCULAR_MAP_INITIALIZER_H
 #define MINISLAM_MONOCULAR_MAP_INITIALIZER_H
 
 #include "Features/Feature.h"
 #include "Matching/lucas_kanade_tracker.h"
 #include "MonocularMapInitializer.h"
 #include "Visualization/FrameVisualizer.h"
 #include "Visualization/image_viz.h"
 #include <opencv2/opencv.hpp>
 #include <unordered_map>
 class MonocularMapInitializerKLT {
 public:
     struct Options {
         int klt_window_size = 7;
         int klt_max_level = 4;
         int klt_max_iters = 10;
         float klt_epsilon = 0.001;
         float klt_min_eig_th = 1e-4;
         float klt_min_SSIM = 0.5;
 
         int rigid_initializer_max_features;
         int rigid_initializer_min_sample_set_size;
         float rigid_initializer_min_parallax;
         float rigid_initializer_radians_per_pixel;
         float rigid_initializer_epipolar_threshold;
     };
 
     struct InitializationResults {
         Sophus::SE3f camera_transform_world;
 
         std::vector<cv::KeyPoint> reference_keypoints;
         std::vector<cv::KeyPoint> current_keypoints;
 
         std::vector<Eigen::Vector3f> reference_landmark_positions;
         std::vector<Eigen::Vector3f> current_landmark_positions;
     };
 
     MonocularMapInitializerKLT() = delete;
 
     MonocularMapInitializerKLT(Options& options, std::shared_ptr<Feature> feature_extractor,
                             std::shared_ptr<CameraModel> calibration);
 
    std::tuple<InitializationResults,bool> ProcessNewImage(const cv::Mat& im, const cv::Mat& im_clahe,
                          const cv::Mat& mask);
                          void DataAssociation(const cv::Mat& im, const cv::Mat& im_clahe,
                            const cv::Mat& mask);
 private:
    
 
     void ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                          std::vector<cv::KeyPoint>& keypoints);
 
     void AddFeatureTracks(const std::vector<cv::KeyPoint>& keypoints,
                           const std::vector<LandmarkStatus>& keypoint_statuses);
 
     void UpdateTrackingReference(const cv::Mat& im);
 
     std::vector<int> FeatureTracksClustering(const cv::Mat& image_to_display);
 
     void ResetInitialization(const cv::Mat& im, const cv::Mat& im_clahe,
                              const cv::Mat& mask);
 
     typedef std::tuple<Sophus::SE3f, std::vector<Eigen::Vector3f>> RigidInitializationResults;
 
     std::tuple<RigidInitializationResults,bool> RigidInitialization();
 
     std::tuple<InitializationResults,bool> InitializationRefinement(std::vector<cv::KeyPoint>& current_keypoints,
                                   std::vector<Eigen::Vector3f>& landmarks_position,
                                   std::vector<int>& feature_labels,
                                   Sophus::SE3f& camera_transform_world);
 
     InitializationResults BuildInitializationResults(std::vector<std::vector<cv::KeyPoint>>& feature_tracks,
                                                 std::vector<std::vector<Eigen::Vector3f>>& landmark_tracks,
                                                 std::vector<int>& track_labels,
                                                 std::vector<Sophus::SE3f>& camera_trajectory);
 
     enum InternalStatus {
         NO_DATA,
         OK,
         RECENTLY_RESET
     };
 
     InternalStatus internal_status_;
 
     struct FeatureTrack {
         std::vector<cv::KeyPoint> track_;
     };
     std::shared_ptr<ImageVisualizer> image_visualizer_;
 
     struct FeatureTracks {
         int max_feature_track_lenght = 0;
        //TODO: Fix his to use a vector instead of a map
         // This map is used to store the feature tracks indexed by their feature ID.
         // This allows for quick access to the feature track corresponding to a specific feature ID.
         // It is assumed that feature IDs are unique and correspond to the class_id of cv::KeyPoint.
         // This is useful for tracking features across multiple frames in a video sequence.
         // The map is implemented using absl::flat_hash_map for efficient lookups.
         // The key is the feature ID (int), and the value is the FeatureTrack object.
         std::unordered_map<int, FeatureTrack> feature_id_to_feature_track;
     };
 
     Options options_;
 
     std::shared_ptr<Feature> feature_extractor_;
 
     LucasKanadeTracker klt_tracker_;
 
     FeatureTracks feature_tracks_;
 
     std::vector<cv::KeyPoint> current_keypoints_;
 
     std::vector<LandmarkStatus> current_keypoint_statuses_;
 
     int images_from_last_reference_ = 0;
 
 
     int n_tracks_in_image_;
 
     std::shared_ptr<CameraModel> calibration_;
     MonocularMapInitializer monoInitializer_;
 };
 
 
 #endif //NRSLAM_MONOCULAR_MAP_INITIALIZER_H
 