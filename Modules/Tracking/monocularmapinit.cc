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

#include "monocularmapinit.h"

 #include "Utils/dbscan.h"
 
 
 #include <Eigen/Core>
 
 using namespace std;
 
 MonocularMapInitializerKLT::MonocularMapInitializerKLT(Options& options,
                                                  std::shared_ptr<Feature> feature_extractor,
                                                  std::shared_ptr<CameraModel> calibration) :
     options_(options), feature_extractor_(feature_extractor){
     klt_tracker_ = LucasKanadeTracker(cv::Size(options_.klt_window_size, options_.klt_window_size),
                                       options_.klt_max_level, options_.klt_max_iters,
                                       options_.klt_epsilon, options_.klt_min_eig_th);
 
    calibration_ = calibration;
     
    monoInitializer_ = MonocularMapInitializer(options_.rigid_initializer_max_features, calibration, options_.rigid_initializer_epipolar_threshold, options_.rigid_initializer_min_parallax);
    ImageVisualizer::Options optionss;
    optionss.image_save_path = "images/";
    optionss.wait_for_user_button = true;
    
        // Initialize the image visualizer.
        // Note: This is a shared pointer to allow for easy visualization of images.
    image_visualizer_ = std::make_shared<ImageVisualizer>(optionss);
 
     internal_status_ = NO_DATA;
 }
 
 std::tuple<MonocularMapInitializerKLT::InitializationResults,bool>
 MonocularMapInitializerKLT::ProcessNewImage(const cv::Mat& im, const cv::Mat& im_clahe,
                                               const cv::Mat& mask) {
     // Track features and update the feature tracks.
     DataAssociation(im, im_clahe, mask);
//     image_visualizer_->DrawFeatures(current_keypoints_, im, "Current Keypoints");
  //   image_visualizer_->UpdateWindows();
    //image_visualizer_->SetCurrentImage(im, im_clahe);
               if (internal_status_ == RECENTLY_RESET) {
        return std::tuple<InitializationResults,bool>(InitializationResults{Sophus::SE3f(), {}, {}, {}, {}},false);
    }
                                                
     // Perform optical flow clustering.
     auto feature_labels = FeatureTracksClustering(im);
 
     //Try to perform a rigid initialization.
    auto [initialization_results, success] = RigidInitialization();
    std::cout << "MonocularMapInitializerKLT: Trying to perform rigid initialization." << std::endl;
    
    if(!success){
        std::cout << "MonocularMapInitializerKLT: Rigid initialization failed." << std::endl;
        return std::tuple<InitializationResults,bool>(InitializationResults{Sophus::SE3f(), {}, {}, {}, {}},false);
    }
    std::cout << "MonocularMapInitializerKLT: Rigid initialization succeeded.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
     auto [camera_transform_world, landmarks_positions] = initialization_results;
     // Perform a deformable Bundle Adjustment to refine the results.
     return InitializationRefinement(current_keypoints_, landmarks_positions, feature_labels, camera_transform_world);
 
 }
 
 void MonocularMapInitializerKLT::ResetInitialization(const cv::Mat& im, const cv::Mat& im_clahe,
                                                   const cv::Mat& mask) {
     // Clear previous data.
     current_keypoints_.clear();
 
     feature_tracks_.max_feature_track_lenght = 0;
     feature_tracks_.feature_id_to_feature_track.clear();
 
     ExtractFeatures(im, mask, current_keypoints_);
 
     // Initialize KLT.
     klt_tracker_.SetReferenceImage(im, current_keypoints_);
 
     current_keypoint_statuses_.resize(current_keypoints_.size());
     fill(current_keypoint_statuses_.begin(), current_keypoint_statuses_.end(),
          TRACKED);
 
     images_from_last_reference_ = 0;
 
     //Set reference data in the rigid initializer.
     monoInitializer_.changeReference(current_keypoints_);
 
     internal_status_ = RECENTLY_RESET;
 }
 
 void MonocularMapInitializerKLT::DataAssociation(const cv::Mat& im, const cv::Mat& im_clahe,
                                               const cv::Mat& mask) {
     if (internal_status_ == NO_DATA) {
         ResetInitialization(im, im_clahe, mask);
     } else {
         // Track features.

         
         n_tracks_in_image_ = klt_tracker_.Track(im, current_keypoints_, current_keypoint_statuses_,
                            true, options_.klt_min_SSIM, mask);
 
         //std::cout << "Number of matches: " << n_tracks_in_image_;
 
         if (n_tracks_in_image_ < 100) {
             ResetInitialization(im, im_clahe, mask);
         } else {
             images_from_last_reference_++;
             internal_status_ = OK;
            // std::cout << "MonocularMapInitializerKLT: Tracked " << n_tracks_in_image_
              //         << " features." << std::endl;
 
             // Update KLT reference image if needed.
             if (images_from_last_reference_ > 30) {
                 ResetInitialization(im, im_clahe, mask);
 
                 images_from_last_reference_ = 0;
             }
         }
     }
 
     // Add features to the track history.
     AddFeatureTracks(current_keypoints_, current_keypoint_statuses_);
 }
 
 void MonocularMapInitializerKLT::ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                                std::vector<cv::KeyPoint>& keypoints) {
     // Extract features.
     feature_extractor_->extract(im, keypoints);
 
     // Mask out points.
     vector<cv::KeyPoint> masked_keypoints;
     for(size_t i = 0; i < keypoints.size(); i++){
 
         if(!mask.at<uchar>(keypoints[i].pt)){
             continue;
         }
         else{
             masked_keypoints.push_back(keypoints[i]);
         }
     }
 
     keypoints = masked_keypoints;
 }
 
 void MonocularMapInitializerKLT::AddFeatureTracks(const std::vector<cv::KeyPoint> &keypoints,
                                                const std::vector<LandmarkStatus>& keypoint_statuses) {
     for (int idx = 0; idx < keypoints.size(); idx++) {
         if (keypoint_statuses[idx] == TRACKED) {
             const cv::KeyPoint keypoint = keypoints[idx];
             feature_tracks_.feature_id_to_feature_track[keypoint.class_id]
                 .track_.push_back(keypoint);
         }
     }
 
     feature_tracks_.max_feature_track_lenght++;
 }
 
 void MonocularMapInitializerKLT::UpdateTrackingReference(const cv::Mat& im) {
     vector<cv::KeyPoint> tracked_keypoints;
     for (int idx = 0; idx < current_keypoints_.size(); idx++) {
         if (current_keypoint_statuses_[idx] == TRACKED) {
             tracked_keypoints.push_back(current_keypoints_[idx]);
         }
     }
 
     current_keypoints_ = tracked_keypoints;
 
     current_keypoint_statuses_.resize(current_keypoints_.size());
     fill(current_keypoint_statuses_.begin(), current_keypoint_statuses_.end(),
          TRACKED);
//     std::cout << "MonocularMapInitializerKLT: Updating reference image with "
  //             << current_keypoints_.size() << " keypoints." << std::endl;
     // Update the KLT tracker with the new reference image.
     if (current_keypoints_.empty()) {
         std::cerr << "MonocularMapInitializerKLT: No keypoints to update reference image." << std::endl;
         return;
     }
     // Set the reference image to the KLT tracker.
     // This is necessary to ensure that the KLT tracker has the correct reference image
     // for the next tracking iteration.
     // This is necessary to ensure that the KLT tracker has the correct reference image
     klt_tracker_.SetReferenceImage(im, current_keypoints_);
 }
 
 std::vector<int> MonocularMapInitializerKLT::FeatureTracksClustering(const cv::Mat& image_to_display) {
     // Get only feature tracks with maximum length.
     const int max_track_length = feature_tracks_.max_feature_track_lenght;
     std::vector<Eigen::VectorXf> plain_feature_tracks;
     std::vector<std::vector<cv::Point2f>> feature_tracks;
     std::unordered_map<int, int> idx_to_feature_id;
     for (const auto& [id, feature_track] : feature_tracks_.feature_id_to_feature_track) {
         if (feature_track.track_.size() == max_track_length) {
             idx_to_feature_id[feature_tracks.size()] = id;
 
             Eigen::VectorXf plain_track((max_track_length - 1) * 2);
             std::vector<cv::Point2f> track(max_track_length);
 
             track[0] = feature_track.track_[0].pt;
 
             for (int idx = 1; idx < feature_track.track_.size(); idx++) {
                 cv::Point2f flow = feature_track.track_[idx].pt - feature_track.track_[idx - 1].pt;
                 plain_track((idx - 1) * 2) = flow.x;
                 plain_track((idx - 1) * 2 + 1) = flow.y;
 
                 track[idx] = feature_track.track_[idx].pt;
             }
 
             plain_feature_tracks.push_back(plain_track);
             feature_tracks.push_back(track);
         }
     }
 
     vector<int> point_labels = DbscanND(plain_feature_tracks);
 
     // TODO:Draw clustered tracks.
   // image_visualizer_->DrawClusteredOpticalFlow(feature_tracks, point_labels,image_to_display);
    //image_visualizer_->UpdateWindows(); 
     return point_labels;
 }
 
std::tuple<MonocularMapInitializerKLT::RigidInitializationResults,bool> MonocularMapInitializerKLT::RigidInitialization() {
     Sophus::SE3f camera_transform_world;
     std::vector<Eigen::Vector3f> landmarks_position;
    //convert status if tracked to -1 or 1
    std::cout << "RIGIDDDDDDDDDDDD"<< std::endl;
     if (current_keypoints_.size() < 10) {
         std::cerr << "MonocularMapInitializerKLT: Not enough keypoints for rigid initialization." << std::endl;
         return make_tuple(make_tuple(camera_transform_world, landmarks_position),false);
     }
 
     // Convert current keypoint statuses to the format expected by the initializer.
     // 1 for tracked, -1 for not tracked.
    std::vector<int> current_keypoint_statuses_format;
    for(const auto& status_ : current_keypoint_statuses_) {
         if (status_ == TRACKED ) {
             current_keypoint_statuses_format.push_back(1);
         } else {
             current_keypoint_statuses_format.push_back(-1);
         }
     }
 
     // Initialize the monocular map initializer.
     std::vector<bool> vTriangulated;
     std::cout << "MonocularMapInitializerKLT: Rigid initialization with "
               << current_keypoints_.size() << " keypoints." << std::endl;
    std::cout << "NTRACKS"<< n_tracks_in_image_ << std::endl; 
    vTriangulated.resize(n_tracks_in_image_, false);
     landmarks_position.resize(n_tracks_in_image_);
    // std::cout << "VTriangulated size: " << vTriangulated.size() << std::endl;
    //std::cout << "MonocularMapInitializerKLT: Rigid initialization with " << current_keypoints_.size() << " keypoints." <<  std::endl;
     //std::cout << "MonocularMapInitializerKLT: Rigid initialization with " << current_keypoint_statuses_format.size() << " keypoints." <<  std::endl;
    //std::cout << "MonocularMapInitializerKLT: Rigid initialization with " << n_tracks_in_image_ << " tracks in image." <<  std::endl;
     // Perform the rigid initialization.
     auto status =monoInitializer_.initialize(current_keypoints_, 
                                                current_keypoint_statuses_format,
                                                 n_tracks_in_image_, 
                                                 camera_transform_world,
                                                 landmarks_position,
                                                vTriangulated);
    //update point tracked with the triangulated points
        for (int idx = 0; idx < current_keypoint_statuses_format.size(); idx++) {
            if (vTriangulated[idx]) {
                current_keypoint_statuses_[idx] = TRACKED_WITH_3D;
            } else {
                current_keypoint_statuses_[idx] = TRACKED;
            }
        }
        std::cout << "Number of Landmarks N"<< landmarks_position.size() << std::endl;
 
     if (landmarks_position.size() == 0) {
        std::cerr << "MonocularMapInitializerKLT: Rigid initialization failed: " << std::endl;
        return make_tuple(make_tuple(camera_transform_world, landmarks_position),false);
    } else {
           std::cout << "MonocularMapInitializerKLT: Rigid initialization succeeded." << std::endl;
         return make_tuple(make_tuple(camera_transform_world, landmarks_position),true);
     }
 }
 
 std::tuple<MonocularMapInitializerKLT::InitializationResults,bool>
 MonocularMapInitializerKLT::InitializationRefinement(std::vector<cv::KeyPoint>& current_keypoints,
                               std::vector<Eigen::Vector3f>& landmarks_position,
                               std::vector<int>& feature_labels,
                               Sophus::SE3f& camera_transform_world) {
     vector<vector<cv::KeyPoint>> feature_tracks;
     vector<vector<Eigen::Vector3f>> landmark_tracks;
     vector<int> track_labels;
     vector<Sophus::SE3f> camera_trajectory;
     std::cout << "MonocularMapInitializerKLT: Refining initialization with "
       << landmarks_position.size() << " landmarks." << std::endl;
     const int track_leghth = feature_tracks_.max_feature_track_lenght;
    
    std::cout <<"TRACK LENGTH: " << track_leghth << std::endl;
                                                       
     for (int idx = 0; idx < landmarks_position.size(); idx++) {
        //TODO: check this
        
        int feature_id = current_keypoints[idx].class_id;

       // if (feature_tracks_.feature_id_to_feature_track[feature_id].track_.size() !=
         //   feature_tracks_.max_feature_track_lenght) {
           // continue;
        //}

        feature_tracks.push_back(feature_tracks_.feature_id_to_feature_track[feature_id].track_);
        std::cout << "MonocularMapInitializerKLT: Adding feature track with ID: " << feature_id << std::endl;
        track_labels.push_back(feature_labels[idx]);
        vector<Eigen::Vector3f> landmark_track(track_leghth, landmarks_position[idx]);
        landmark_tracks.push_back(landmark_track);
     }
     
     std::cout << "MonocularMapInitializerKLT: Building initialization results with "
    << feature_tracks.size() << " feature tracks." << std::endl;
     std::cout << "MonocularMapInitializerKLT: Building initialization results with "
     << landmark_tracks.size() << " landmakr tracks." << std::endl;

     // Interpolate camera trajectory.
     camera_trajectory.resize(track_leghth);
     Eigen::Quaternionf origin_rotation = Eigen::Quaternionf::Identity();
     for (int idx = 0; idx < track_leghth; idx++) {
         const float weight = idx / (track_leghth - 1);
         camera_trajectory[idx].translation() = camera_transform_world.translation() * weight;
         camera_trajectory[idx].setQuaternion(
                 origin_rotation.slerp(weight, camera_transform_world.unit_quaternion()));
     }
     std::cout << "CAMERA TRAJ"<< camera_trajectory.size() << std::endl;
     // Build reference and current frames from the estimated geometry.
 
    auto results = BuildInitializationResults(feature_tracks, landmark_tracks, track_labels, camera_trajectory);
     results.camera_transform_world = camera_transform_world;
     return std::tuple<MonocularMapInitializerKLT::InitializationResults,bool>(results, true);
 }
 
 MonocularMapInitializerKLT::InitializationResults
 MonocularMapInitializerKLT::BuildInitializationResults(std::vector<std::vector<cv::KeyPoint>> &feature_tracks,
                                             std::vector<std::vector<Eigen::Vector3f>> &landmark_tracks,
                                             std::vector<int> &track_labels,
                                             std::vector<Sophus::SE3f> &camera_trajectory) {
     MonocularMapInitializerKLT::InitializationResults initialization_results;
     initialization_results.camera_transform_world = camera_trajectory.back();
 
     vector<float> initial_depths;
 
     for (int idx = 0; idx < feature_tracks.size(); idx++) {
         cv::KeyPoint reference_keypoint(feature_tracks[idx].front());
         cv::KeyPoint current_keypoint(feature_tracks[idx].back());
 
         Eigen::Vector3f reference_landmark_position = landmark_tracks[idx].front();
         Eigen::Vector3f current_landmark_position = landmark_tracks[idx].back();
 
         initial_depths.push_back(reference_landmark_position.z());
 
         initialization_results.reference_keypoints.push_back(reference_keypoint);
         initialization_results.current_keypoints.push_back(current_keypoint);
         initialization_results.reference_landmark_positions.push_back(reference_landmark_position);
         initialization_results.current_landmark_positions.push_back(current_landmark_position);
     }
 
     return initialization_results;
 }
 