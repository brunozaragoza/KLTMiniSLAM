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

#include "TrackingKLT.h"

#include "Features/FAST.h"
#include "Features/ORB.h"
#include "Features/ShiTomasi.h"
#include "Map/KeyFrame.h"
#include "Map/MapPoint.h"
#include "Matching/DescriptorMatching.h"

#include "Optimization/g2oBundleAdjustment.h"

using namespace std;

TrackingKLT::TrackingKLT() {}

TrackingKLT::TrackingKLT(Settings &settings, std::shared_ptr<FrameVisualizer> &visualizer,
                         std::shared_ptr<MapVisualizer> &mapVisualizer, std::shared_ptr<Map> map)
{
    currFrame_ = Frame(settings.getFeaturesPerImage(), settings.getGridCols(), settings.getGridRows(),
                       settings.getImCols(), settings.getImRows(), settings.getNumberOfScales(), settings.getScaleFactor(),
                       settings.getCalibration(), settings.getDistortionParameters());
    prevFrame_ = Frame(settings.getFeaturesPerImage(), settings.getGridCols(), settings.getGridRows(),
                       settings.getImCols(), settings.getImRows(), settings.getNumberOfScales(), settings.getScaleFactor(),
                       settings.getCalibration(), settings.getDistortionParameters());
    ShiTomasi::Options par;
    par.non_max_suprresion_window_size = 7;
    featExtractor_ = make_shared<ShiTomasi>(par);
    descExtractor_ = shared_ptr<Descriptor>(new ORB(settings.getNumberOfScales(), settings.getScaleFactor()));

    vMatches_ = vector<int>(settings.getFeaturesPerImage());

    vPrevMatched_ = vector<cv::Point2f>(settings.getFeaturesPerImage());

    status_ = NOT_INITIALIZED;
    bFirstIm_ = true;
    bMotionModel_ = false;
    klt_tracker_ = LucasKanadeTracker(cv::Size(options_.klt_window_size, options_.klt_window_size),
                                      options_.klt_max_level, options_.klt_max_iters,
                                      options_.klt_epsilon, options_.klt_min_eig_th);
    MonocularMapInitializerKLT::Options options;
    monoInitializer_ = std::make_shared<MonocularMapInitializerKLT>(options, featExtractor_, settings.getCalibration());

    visualizer_ = visualizer;
    mapVisualizer_ = mapVisualizer;
    visualizer_ = visualizer;
    pMap_ = map;

    nLastKeyFrameId = 0;
    nFramesFromLastKF_ = 0;

    bInserted = false;

    settings_ = settings;
}

bool TrackingKLT::doTracking(const cv::Mat &im, Sophus::SE3f &Tcw)
{
    currIm_ = im.clone();
    // Update previous frame
    if (status_ != NOT_INITIALIZED)
        prevFrame_.assign(currFrame_);
    // Update previous frame

    currFrame_.setIm(currIm_);

    nframesext++;
    //visualizer_->drawCurrentFeatures(prevFrame_.getKeyPoints(), currIm_);
    // If no map is initialized, perform monocular initialization
    if (status_ == NOT_INITIALIZED)
    {
        cv::Mat global_mask(currIm_.rows, currIm_.cols, CV_8U, cv::Scalar(255));
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::erode(global_mask, global_mask, kernel);

            // Extract features in the current image
        
            if (MonocularMapInitialization(currIm_, global_mask, prevIm_))
        {

            std::cout << "MonocularMapInitialization" << std::endl;  

            status_ = GOOD;
            Tcw = currFrame_.getPose();
            // Promote current frame to KeyFrame
            pLastKeyFrame_ = shared_ptr<KeyFrame>(new KeyFrame(currFrame_));
            visualizer_->drawCurrentFeatures(currFrame_.getKeyPoints(), currIm_);

            // Insert KeyFrame into the map
            pMap_->insertKeyFrame(pLastKeyFrame_);

            return true;
        }
    
        else
        {
            return false;
        }
    
    }
    // Track the following camera poses
    if (status_ == GOOD)
    {
      

        if (cameraTracking())
        {

            Tcw = currFrame_.getPose();
            //updateMotionModel();
            // Promote current frame to KeyFrame
            pLastKeyFrame_ = shared_ptr<KeyFrame>(new KeyFrame(currFrame_));
            // Update motion model

            // Insert KeyFrame into the map
            pMap_->insertKeyFrame(pLastKeyFrame_);
            visualizer_->drawCurrentFrame(currFrame_);
            
            return true;
        }
        else
        {
            status_ = LOST;
            return false;
        }
    }
}
std::shared_ptr<KeyFrame> TrackingKLT::getLastKeyFrame()
{
    shared_ptr<KeyFrame> toReturn = pLastKeyFrame_;
    pLastKeyFrame_ = nullptr;

    return toReturn;
}
void TrackingKLT::updateLastPose()
{
    Sophus::SE3f Tcw = currFrame_.getPose();
    prevFrame_.setPose(Tcw);
}

void TrackingKLT::extractFeatures(const cv::Mat &im)
{
    // Extracf image features
    featExtractor_->extract(im, currFrame_.getKeyPointsDistorted());
    // Compute descriptors to extracted features
    descExtractor_->describe(im, currFrame_.getKeyPointsDistorted(), currFrame_.getDescriptors());

    // Distribute keys and undistort them
    currFrame_.distributeFeatures();
}

void TrackingKLT::ExtractFeaturesInFrame(const cv::Mat& im) {
    extractFeatures(im);  // this should populate currFrame_

    const auto& currKPs = currFrame_.getKeyPoints();

    // Update vMatches_ based on frame's current statuses
    std::vector<LandmarkStatus> statuses;

    for (int idx = 0; idx < currKPs.size(); idx++) {
            statuses.push_back(LandmarkStatus::TRACKED);
    }

    currFrame_.setLandmarkStatuses(statuses);
    currFrame_.setKeyPoints(currFrame_.getKeyPoints());
}

bool TrackingKLT::MonocularMapInitialization(const cv::Mat& im_left,
        const cv::Mat& mask, const cv::Mat& im_clahe) {

bFirstIm_ = false;
    // Perform the initialization
auto initialization_status = monoInitializer_->ProcessNewImage(im_left, im_clahe, mask);

if (initialization_status.current_keypoints.empty()) {
    std::cout << "MonocularMapInitializerKLT: No keypoints found for initialization." << std::endl;
    return false;
}
auto initialization_results = initialization_status;

vector<float> depths;
for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx];
depths.push_back(current_landmark_position.z());
}

const int median_idx = depths.size() / 2;
nth_element(depths.begin(), depths.begin() + median_idx, depths.end());
const float median_depth = depths[median_idx];
const float scale = 3.f / median_depth;
float accumulate = 0;
for (float element : depths){
    accumulate += element;
}

float mu=accumulate / depths.size();
float accumulated = 0;

for(float element : depths){
    accumulated += (element - mu) * (element - mu);
}

float sigma=sqrtf(accumulated / depths.size());
float sigma_scaled = sigma * scale;


Frame reference_frame;
for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
cv::KeyPoint reference_keypoint = initialization_results.reference_keypoints[idx];
cv::KeyPoint current_keypoint = initialization_results.current_keypoints[idx];

Eigen::Vector3f reference_landmark_position = initialization_results.reference_landmark_positions[idx] * scale;
Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx] * scale;
//Create a new MapPoint
std::shared_ptr<MapPoint> reference_landmark_position_ptr = make_shared<MapPoint>(reference_landmark_position);
pMap_->insertMapPoint( reference_landmark_position_ptr);

pMap_->addObservation(0,reference_landmark_position_ptr->getId(),idx);
pMap_->addObservation(1,reference_landmark_position_ptr->getId(),idx);
/*reference_frame.InsertObservation(reference_keypoint,
           reference_landmark_position,
           mappoint_id,
           TRACKED_WITH_3D);

current_frame_->InsertObservation(current_keypoint,
           current_landmark_position,
           mappoint_id,
           TRACKED_WITH_3D);
*/

           
pMap_->addObservation(0,reference_landmark_position_ptr->getId(),idx);

//reference_frame.MutableCameraTransformationWorld() = Sophus::SE3f();
//initialization_results.camera_transform_world.translation() = initialization_results.camera_transform_world.translation() * scale;
//current_frame_->MutableCameraTransformationWorld() = initialization_results.camera_transform_world;
}
// Create Keyframes from the frames.
auto first_keyframe = make_shared<KeyFrame>(reference_frame);
auto current_keyframe = make_shared<KeyFrame>(currFrame_);

// Insert KeyFrame in the map.
pMap_->insertKeyFrame(first_keyframe);
pMap_->insertKeyFrame(current_keyframe);

// Set reference image to the KLT tracker.
klt_tracker_.SetReferenceImage(im_left, currFrame_.getKeyPoints(), mask);
return true;
    
}

bool TrackingKLT::cameraTracking()
{
    // === [0] Prepare mask
    cv::Mat global_mask(currIm_.rows, currIm_.cols, CV_8U, cv::Scalar(255));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(global_mask, global_mask, kernel);
    for( const auto &pt: prevFrame_.getMapPoints()){
        currFrame_.setMapPoint(pt->getId(),pt);
    }
    std::cout<<"[DEBUG] MapPoints in the current frame: "<<currFrame_.getMapPoints().size()<<std::endl;

    // === [1] KLT Tracking ===
    std::vector<cv::KeyPoint> trackedPts = prevFrame_.getKeyPoints();
    std::vector<LandmarkStatus> statuses(prevFrame_.getKeyPoints().size(), LandmarkStatus::TRACKED);
    int nMatches = klt_tracker_.Track(currIm_, trackedPts, statuses, true, options_.klt_min_SSIM, global_mask);
    std::cout << "[KLT] Tracked features: " << nMatches << std::endl;
    // print the status of the tracked points
    auto vTriangulated = prevFrame_.getMapPoints();
    
     //associate the 3 points to the vmatches
    for (int t = 0; t < currFrame_.LandmarkStatuses().size(); t++)
    {
        if (currFrame_.LandmarkStatuses()[t] == LandmarkStatus::TRACKED)
            vMatches_[t] = 1;
        else
        {
            vMatches_[t] = -1;
        }
    }
    currFrame_.setLandmarkStatuses(statuses);
    currFrame_.setKeyPoints(trackedPts);

    Sophus::SE3f currPose= prevFrame_.getPose();
    currFrame_.setPose(currPose);

    poseOnlyOptimization(currFrame_);
    
    //Tcw.translation() /= scale;
    Sophus::SE3f Twc=currFrame_.getPose();
    mapVisualizer_->updateCurrentPose(Twc);

    klt_tracker_.SetReferenceImage(currIm_, currFrame_.getKeyPoints(), global_mask);
    prevFrame_.assign(currFrame_);
    prevIm_ = currIm_;
    // std::cout << "[DEBUG] currFrame_ Pose:\n" << currFrame_.getPose().matrix() << std::endl;
    status_ == GOOD;
    // === [9] Success if enough tracked
    return nMatches >= 20;
}



void TrackingKLT::updateMotionModel()
{
    motionModel_ = currFrame_.getPose() * prevFrame_.getPose().inverse();
}
