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
    currFrame_ = make_shared<Frame>(settings.getFeaturesPerImage(), settings.getGridCols(), settings.getGridRows(),
                                    settings.getImCols(), settings.getImRows(), settings.getNumberOfScales(), settings.getScaleFactor(),
                                    settings.getCalibration(), settings.getDistortionParameters());
    prevFrame_ = make_shared<Frame>(settings.getFeaturesPerImage(), settings.getGridCols(), settings.getGridRows(),
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

    monoInitializer_ = std::make_shared<MonocularMapInitializer>(settings.getFeaturesPerImage(), settings.getCalibration(), settings.getEpipolarTh(), settings.getMinCos());

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

    currFrame_->setIm(currIm_);

    nframesext++;
    auto pts= currFrame_->GetKeypointsWithStatus({TRACKED,TRACKED_WITH_3D});
    visualizer_->drawCurrentFeatures(pts, currIm_);
    visualizer_->updateWindows();
    // If no map is initialized, perform monocular initialization
    if (status_ == NOT_INITIALIZED)
    {
        extractFeatures(im);

        if (MonocularMapInitialization())
        {

            status_ = GOOD;
            Tcw = currFrame_->getPose();
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

            Tcw = currFrame_->getPose();

            mapVisualizer_->updateCurrentPose(Tcw);
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

void TrackingKLT::extractFeatures(const cv::Mat &im)
{
    // Extracf image features
    featExtractor_->extract(im, currFrame_->getKeyPointsDistorted());
    // Compute descriptors to extracted features
    descExtractor_->describe(im, currFrame_->getKeyPointsDistorted(), currFrame_->getDescriptors());

    // Distribute keys and undistort them
    currFrame_->distributeFeatures();
}

bool TrackingKLT::MonocularMapInitialization()
{

    // Set first frame received as the reference frame
    if (bFirstIm_)
    {
        monoInitializer_->changeReference(currFrame_->getKeyPoints());
        prevFrame_->assign(*currFrame_);

        bFirstIm_ = false;

        visualizer_->setReferenceFrame(prevFrame_->getKeyPointsDistorted(), currIm_);

        for (size_t i = 0; i < vPrevMatched_.size(); i++)
        {
            vPrevMatched_[i] = prevFrame_->getKeyPoint(i).pt;
        }

        return false;
    }

    // Find matches between previous and current frame
    int nMatches = searchForInitializaion(*prevFrame_, *currFrame_, settings_.getMatchingInitTh(), vMatches_, vPrevMatched_);

    // If not enough matches found, updtate reference frame
    if (nMatches < 70)
    {
        monoInitializer_->changeReference(currFrame_->getKeyPoints());
        prevFrame_->assign(*currFrame_);

        visualizer_->setReferenceFrame(prevFrame_->getKeyPointsDistorted(), currIm_);

        for (size_t i = 0; i < vPrevMatched_.size(); i++)
        {
            vPrevMatched_[i] = prevFrame_->getKeyPoint(i).pt;
        }

        return false;
    }

    // Try to initialize by finding an Essential matrix
    Sophus::SE3f Tcw;
    vector<Eigen::Vector3f> v3DPoints;
    v3DPoints.reserve(vMatches_.capacity());
    vector<bool> vTriangulated(vMatches_.capacity(), false);
    if (!monoInitializer_->initialize(currFrame_->getKeyPoints(), vMatches_, nMatches, Tcw, v3DPoints, vTriangulated))
    {
        return false;
    }

    
    // find the points corresponding to the matches
    for(int i=0;i<vMatches_.size();i++){
        if(vMatches_[i]==-1) continue;
        currFrame_->LandmarkStatuses()[i]=TRACKED; 
    }

    // Get map scale
    vector<float> vDepths;
    for (int i = 0; i < vTriangulated.size(); i++)
    {
        if (vTriangulated[i])
            vDepths.push_back(v3DPoints[i](2));
    }

    nth_element(vDepths.begin(), vDepths.begin() + vDepths.size() / 2, vDepths.end());
    const float scale = vDepths[vDepths.size() / 2];

    // Create map
    std::vector<cv::KeyPoint> currpts;
    Tcw.translation() = Tcw.translation() / scale;
    cv::Mat prevIM = currFrame_->getIm();
    int nTriangulated = 0;

    for (size_t i = 0; i < vTriangulated.size(); i++)
    {
        if (vTriangulated[i])
        {
            Eigen::Vector3f v = v3DPoints[i] / scale;
            shared_ptr<MapPoint> pMP(new MapPoint(v));
            prevFrame_->setMapPoint(i, pMP);
            currFrame_->setMapPoint(vMatches_[i], pMP);
            pMap_->insertMapPoint(pMP);
            currFrame_->LandmarkStatuses()[vMatches_[i]] = TRACKED_WITH_3D;
            nTriangulated++;
        }
    }
    //    visualizer_->drawFrameMatches(currFrame_->getKeyPoints(), currIm_, vMatches_);
    std::cout << "Map initialized with: " << currFrame_->getMapPoints().size() << endl;
    std::cout << "Tracked Points:"<<currFrame_->GetKeypointsWithStatus({TRACKED_WITH_3D,TRACKED}).size()<<std::endl;
    
    shared_ptr<KeyFrame> kf0(new KeyFrame(*prevFrame_));
    shared_ptr<KeyFrame> kf1(new KeyFrame(*currFrame_));

    pMap_->insertKeyFrame(kf0);
    pMap_->insertKeyFrame(kf1);

    // Set observations into the map
    vector<shared_ptr<MapPoint>> &vMapPoints = kf0->getMapPoints();
    for (size_t i = 0; i < vMapPoints.size(); i++)
    {
        auto pMP = vMapPoints[i];
        if (pMP)
        {
            // Add observation
            pMap_->addObservation(0, pMP->getId(), i);
        }
    }
    vector<shared_ptr<MapPoint>> &vMapPoints2 = kf1->getMapPoints();
    for (size_t i = 0; i < vMapPoints2.size(); i++)
    {
        auto pMP = vMapPoints2[i];
        if (pMP)
        {
            // Add observation
            pMap_->addObservation(1, pMP->getId(), i);
        }
    }

    // Run a Bundle Adjustment to refine the solution
    bundleAdjustment(pMap_.get());

    Tcw = kf1->getPose();

    updateMotionModel();

    klt_tracker_.SetReferenceImage(currIm_, currFrame_->getKeyPoints());

    mapVisualizer_->updateCurrentPose(Tcw);
    // prevFrame_->assign(*currFrame_);
    bInserted = true;

    return true;
}

bool TrackingKLT::cameraTracking()
{
    std::cout << "TRACKING" << std::endl;
    // === [0] Prepare mask
    // convert currIm    to grayscale if it is not already
    cv::Mat global_mask(currIm_.rows, currIm_.cols, CV_8U, cv::Scalar(255));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(global_mask, global_mask, kernel);
    // === [1] KLT Tracking
    int nMatches = klt_tracker_.Track(currIm_, currFrame_->getKeyPoints(), currFrame_->LandmarkStatuses(),
                                      true, options_.klt_min_SSIM, global_mask);

    std::cout << "NMATCHES: " << nMatches << std::endl;
    // === [3] Check if enough points were tracked
    // pose optimization
    std::cout << "Pose optimization done." << std::endl;
    std::cout << "NINLIERS=" << poseOnlyOptimization(*currFrame_) << std::endl;
    Sophus::SE3f Tcwc = currFrame_->getPose();
    mapVisualizer_->updateCurrentPose(Tcwc);
    klt_tracker_.SetReferenceImage(currIm_, currFrame_->getKeyPoints(), global_mask);
    // assign frame
    // prevFrame_->assign(*currFrame_);
    return true;
}

void TrackingKLT::updateMotionModel()
{
    motionModel_ = currFrame_->getPose() * prevFrame_->getPose().inverse();
}
