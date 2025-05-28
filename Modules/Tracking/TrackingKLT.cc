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
    
    monoInitializer_ = std::make_shared<MonocularMapInitializer>(settings.getFeaturesPerImage(),settings.getCalibration(),settings.getEpipolarTh(),settings.getMinCos());

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
   

    visualizer_->drawCurrentFeatures(currFrame_.getKeyPointsDistorted(), currIm_);
    // If no map is initialized, perform monocular initialization
    if (status_ == NOT_INITIALIZED)
    {
        extractFeatures(im);
      
            if (MonocularMapInitialization())
        {


            //nFramesFromLastKF_ = 0;

            std::cout << "MonocularMapInitialization" << std::endl;  
            
            status_ = GOOD;
            Tcw = currFrame_.getPose();
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

bool TrackingKLT::MonocularMapInitialization() {


    //Set first frame received as the reference frame
    if(bFirstIm_){
        monoInitializer_->changeReference(currFrame_.getKeyPoints());
        prevFrame_.assign(currFrame_);

        bFirstIm_ = false;

        visualizer_->setReferenceFrame(prevFrame_.getKeyPointsDistorted(),currIm_);

        for(size_t i = 0; i < vPrevMatched_.size(); i++){
            vPrevMatched_[i] = prevFrame_.getKeyPoint(i).pt;
        }

        return false;
    }

    //Find matches between previous and current frame
    int nMatches = searchForInitializaion(prevFrame_,currFrame_,settings_.getMatchingInitTh(),vMatches_,vPrevMatched_);

    //visualizer_->drawFrameMatches(currFrame_.getKeyPointsDistorted(),currIm_,vMatches_);

    //If not enough matches found, updtate reference frame
    if(nMatches < 70){
        monoInitializer_->changeReference(currFrame_.getKeyPoints());
        prevFrame_.assign(currFrame_);

        visualizer_->setReferenceFrame(prevFrame_.getKeyPointsDistorted(),currIm_);

        for(size_t i = 0; i < vPrevMatched_.size(); i++){
            vPrevMatched_[i] = prevFrame_.getKeyPoint(i).pt;
        }

        return false;
    }

    //Try to initialize by finding an Essential matrix
    Sophus::SE3f Tcw;
    vector<Eigen::Vector3f> v3DPoints;
    v3DPoints.reserve(vMatches_.capacity());
    vector<bool> vTriangulated(vMatches_.capacity(),false);
    if(!monoInitializer_->initialize(currFrame_.getKeyPoints(), vMatches_, nMatches, Tcw, v3DPoints, vTriangulated)){
        return false;
    }

    //Get map scale
    vector<float> vDepths;
    for(int i = 0; i < vTriangulated.size(); i++){
        if(vTriangulated[i])
            vDepths.push_back(v3DPoints[i](2));
    }

    nth_element(vDepths.begin(),vDepths.begin()+vDepths.size()/2,vDepths.end());
    const float scale = vDepths[vDepths.size()/2];

    //Create map
    Tcw.translation() = Tcw.translation() / scale;

    currFrame_.setPose(Tcw);

    int nTriangulated = 0;

    for(size_t i = 0; i < vTriangulated.size(); i++){
        if(vTriangulated[i]){
            Eigen::Vector3f v = v3DPoints[i] / scale;
            shared_ptr<MapPoint> pMP(new MapPoint(v));

            prevFrame_.setMapPoint(i,pMP);
            currFrame_.setMapPoint(vMatches_[i],pMP);

            pMap_->insertMapPoint(pMP);

            nTriangulated++;
        }
    }

    cout << "Map initialized with " << nTriangulated << " MapPoints" << endl;

    shared_ptr<KeyFrame> kf0(new KeyFrame(prevFrame_));
    shared_ptr<KeyFrame> kf1(new KeyFrame(currFrame_));

    pMap_->insertKeyFrame(kf0);
    pMap_->insertKeyFrame(kf1);

    //Set observations into the map
    vector<shared_ptr<MapPoint>>& vMapPoints = kf0->getMapPoints();
    for(size_t i = 0; i < vMapPoints.size(); i++){
        auto pMP = vMapPoints[i];
        if(pMP){
            //Add observation
            pMap_->addObservation(0,pMP->getId(),i);
            pMap_->addObservation(1,pMP->getId(),vMatches_[i]);
        }
    }

    //Run a Bundle Adjustment to refine the solution
    bundleAdjustment(pMap_.get());

    Tcw = kf1->getPose();
    currFrame_.setPose(Tcw);

    updateMotionModel();

    pLastKeyFrame_ = kf1;
    nLastKeyFrameId = kf1->getId();

    mapVisualizer_->updateCurrentPose(Tcw);

    bInserted = true;

    return true;
}

bool TrackingKLT::cameraTracking()
{
    std::cout << "TRACKING"<< std::endl;

    // === [0] Prepare mask
    cv::Mat global_mask(currIm_.rows, currIm_.cols, CV_8U, cv::Scalar(255));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(global_mask, global_mask, kernel);
    // === [1] KLT Tracking
        // Copy previous keypoints (CORRECTION 1)
        std::vector<cv::KeyPoint> pts ;
    
        // Prepare landmark statuses (CORRECTION 2)
        std::vector<LandmarkStatus> statuses(prevFrame_.LandmarkStatuses().size(), LandmarkStatus::TRACKED);
    
        // Run KLT tracking
        currFrame_.resize(prevFrame_.getKeyPoints().size());
        currFrame_.setKeyPoints(prevFrame_.getKeyPoints());
        currFrame_.setLandmarkStatuses(statuses);
        int nMatches = klt_tracker_.Track(currIm_, currFrame_.getKeyPoints(), statuses,
                                          true, options_.klt_min_SSIM, global_mask);
        //convert status to matches using a loop 


           // Store tracked keypoints into current frame (CORRECTION 3)
            currFrame_.setLandmarkStatuses(statuses);
            currFrame_.resize(pts.size());
            currFrame_.setKeyPoints(pts);
            currFrame_.setIm(currIm_);
            visualizer_->drawCurrentFrame(currFrame_);

            for (int t = 0; t < currFrame_.LandmarkStatuses().size(); t++)
            {
                if (currFrame_.LandmarkStatuses()[t] == LandmarkStatus::TRACKED){
                    vMatches_[t] = 1;
                }
                else
                {
                    vMatches_[t] = -1;
                }
            }                              //draw matches 
        visualizer_->drawFrameMatches(currFrame_.getKeyPointsDistorted(), currIm_,vMatches_ );
        visualizer_->updateWindows();

        // === [2] Check if tracking was successful                                          
    //Sophus::SE3f currPose= prevFrame_.getPose();
    //currFrame_.setPose(currPose);

    /*poseOnlyOptimization(currFrame_);
    
    //Tcw.translation() /= scale;
    Sophus::SE3f Twc=currFrame_.getPose();
    mapVisualizer_->updateCurrentPose(Twc);

    klt_tracker_.SetReferenceImage(currIm_, currFrame_.getKeyPoints(), global_mask);
    prevFrame_.assign(currFrame_);
    prevIm_ = currIm_;
   */
    // std::cout << "[DEBUG] currFrame_ Pose:\n" << currFrame_.getPose().matrix() << std::endl;
    status_ == GOOD;
    // === [9] Success if enough tracked
    return true;
}



void TrackingKLT::updateMotionModel()
{
    motionModel_ = currFrame_.getPose() * prevFrame_.getPose().inverse();
}
