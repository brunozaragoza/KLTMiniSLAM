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
    featExtractor_ = shared_ptr<Feature>(new FAST(settings.getNumberOfScales(), settings.getScaleFactor(), settings.getFeaturesPerImage() * 2, 20, 7));
    featExtractor_2 = make_shared<ShiTomasi>(par);
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
            if (nFramesFromLastKF_ % 3)
                promoteCurrentFrameToKeyFrame();
            mapVisualizer_->updateCurrentPose(Tcw);
            nFramesFromLastKF_++;
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
void TrackingKLT::promoteCurrentFrameToKeyFrame()
{
    // Promote current frame to KeyFrame
    pLastKeyFrame_ = shared_ptr<KeyFrame>(new KeyFrame(*currFrame_));

    // Insert KeyFrame into the map
    pMap_->insertKeyFrame(pLastKeyFrame_);

    // Add all obsevations into the map
    nLastKeyFrameId = pLastKeyFrame_->getId();
    vector<shared_ptr<MapPoint>> &vMapPoints = pLastKeyFrame_->getMapPoints();
    for (int i = 0; i < vMapPoints.size(); i++)
    {
        MapPoint *pMP = vMapPoints[i].get();
        if (pMP)
            pMap_->addObservation(nLastKeyFrameId, pMP->getId(), i);
    }
    pMap_->checkKeyFrame(pLastKeyFrame_->getId());
    nFramesFromLastKF_ = 0;
    bInserted = true;
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
        std::cout << "FIRMS IMG" << std::endl;
        visualizer_->setReferenceFrame(prevFrame_->getKeyPointsDistorted(), currIm_);
        visualizer_->drawCurrentFeatures(prevFrame_->getKeyPoints(), currIm_);
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
    cv::Mat prevIm = prevFrame_->getIm();
    visualizer_->drawMatches(prevFrame_->getKeyPoints(), prevIm, currFrame_->getKeyPoints(), currIm_, vMatches_);

    // find the points corresponding to the matches
    for (int i = 0; i < vMatches_.size(); i++)
    {
        if (vMatches_[i] == -1)
            continue;
        currFrame_->LandmarkStatuses()[i] = TRACKED;
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
            currFrame_->AddGeometryToKeypoint(vMatches_[i],
                                              pMP->getId());
            currFrame_->LandmarkStatuses()[vMatches_[i]] = TRACKED_WITH_3D;
            nTriangulated++;
        }
    }
    //    visualizer_->drawFrameMatches(currFrame_->getKeyPoints(), currIm_, vMatches_);
    mapPts = currFrame_->GetLandmarkPointersWithStatus({TRACKED_WITH_3D});

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
    // bundleAdjustment(pMap_.get());

    Tcw = kf1->getPose();
    pLastKeyFrame_ = kf1;
    updateMotionModel();

    klt_tracker_.SetReferenceImage(currIm_, currFrame_->getKeyPoints());

    mapVisualizer_->updateCurrentPose(Tcw);
    // prevFrame_->assign(*currFrame_);
    bInserted = true;

    return true;
}
void TrackingKLT::PointReuse(const cv::Mat &im, const cv::Mat &mask,
                             std::vector<long unsigned int> &lost_mappoint_ids)
{
    auto all_mappoints = pMap_->getMapPoints();
    for (int i = 0; i < all_mappoints.size(); i++)
    {
        if (!currFrame_->getMapPoint(all_mappoints[i]->getId()))
        {
            auto mappoint = all_mappoints[i];
            // Project mappoint into the camera and check if it lies inside the image.
            Eigen::Vector3f landmark_position_seed = mappoint->getWorldPosition();
            Eigen::Vector3f landmark_camera_position = currFrame_->getPose() * landmark_position_seed;

            if (landmark_camera_position.z() < 0)
            {
                continue;
            }

            Eigen::Vector2f projected_landmark;
            currFrame_->getCalibration()->project(landmark_camera_position, projected_landmark);

            if (projected_landmark.x() >= 0 && projected_landmark.x() < im.cols &&
                projected_landmark.y() >= 0 && projected_landmark.y() < im.rows)
            {
                auto idd = all_mappoints[i]->getId();
                lost_mappoint_ids.push_back(idd);
            }
        }
    }

    if (lost_mappoint_ids.empty())
    {
        return;
    }
    // Project candidates into the image.
    Frame frame_with_only_candidates;

    LucasKanadeTracker klt(cv::Size(options_.klt_window_size, options_.klt_window_size),
                           1, options_.klt_max_iters,
                           options_.klt_epsilon, options_.klt_min_eig_th);

    int candidates_in_image = 0;
    vector<cv::KeyPoint> keypoint_seeds;
    for (const auto &mappoint_id : lost_mappoint_ids)
    {
        auto mappoint = pMap_->getMapPoint(mappoint_id);
        Eigen::Vector3f landmark_position_seed = mappoint->getWorldPosition();
        Eigen::Vector3f landmark_camera_position = currFrame_->getPose() * landmark_position_seed;
        Eigen::Vector2f projected_landmark;
        currFrame_->getCalibration()->project(landmark_camera_position, projected_landmark);

        if (isnan(projected_landmark.x()) || isnan(projected_landmark.y()))
        {
            std::cout << "NaN found!";
        }

        if (projected_landmark.x() >= 0 && projected_landmark.x() < im.cols &&
            projected_landmark.y() >= 0 && projected_landmark.y() < im.rows)
        {
            cv::KeyPoint kp(cv::Point2f(projected_landmark.x(), projected_landmark.y()), 1.0f);

            frame_with_only_candidates.InsertObservation(kp, mappoint, mappoint_id, TRACKED_WITH_3D);

            keypoint_seeds.push_back(kp);

            candidates_in_image++;
        }
    }

    if (candidates_in_image == 0)
    {
        return;
    }
    std::cout << "CANDIDATES" << candidates_in_image << std::endl;

    klt.Track(im, frame_with_only_candidates.getKeyPoints(), frame_with_only_candidates.LandmarkStatuses(),
              true, 0.75, mask);

    // Insert tracked candidates into the current frame
    vector<cv::KeyPoint> tracked_candidate_keypoints =
        frame_with_only_candidates.GetKeypointsWithStatus({TRACKED_WITH_3D});
    auto tracked_candidate_landmarks =
        frame_with_only_candidates.GetLandmarkPointersWithStatus({TRACKED_WITH_3D});
    vector<int> tracked_candidate_mappoint_ids =
        frame_with_only_candidates.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

    int reused_landmarks = 0;

    for (int idx = 0; idx < tracked_candidate_keypoints.size(); idx++)
    {
        cv::KeyPoint keypoint = tracked_candidate_keypoints[idx];
        auto mappoint = tracked_candidate_landmarks[idx];
        int mappoint_id = tracked_candidate_mappoint_ids[idx];

        keypoint.class_id = pMap_->getMapPoint(mappoint_id)->getId();

        Eigen::Vector3f landmark_position_seed = mappoint->getWorldPosition();
        Eigen::Vector3f landmark_camera_position = currFrame_->getPose() * landmark_position_seed;
        Eigen::Vector2f projected_landmark;
        currFrame_->getCalibration()->project(landmark_camera_position, projected_landmark);

        float errx = projected_landmark.x() - keypoint.pt.x;
        float erry = projected_landmark.y() - keypoint.pt.y;

        float errtotal = errx * errx + erry * erry;
        if (errtotal > 5.99)
        {
            continue;
        }
        if (currFrame_->MapPointIdToIndex().find(mappoint_id) != currFrame_->MapPointIdToIndex().end())
        {
            const int idx_in_frame = currFrame_->MapPointIdToIndex().at(mappoint_id);

            currFrame_->getKeyPoints()[idx_in_frame] = keypoint;
            currFrame_->getMapPoints()[idx_in_frame] = mappoint;
            currFrame_->LandmarkStatuses()[idx_in_frame] = TRACKED_WITH_3D;
        }
        else
        {
            currFrame_->InsertObservation(keypoint, mappoint, mappoint_id, TRACKED_WITH_3D);
        }

        reused_landmarks++;
    }

    std::cout << "Reused landmarks: " << reused_landmarks;
}
bool TrackingKLT::cameraTracking()
{
    std::cout << "TRACKING" << std::endl;
    // === [0] Prepare mask
    cv::Mat global_mask(currIm_.rows, currIm_.cols, CV_8U, cv::Scalar(255));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(global_mask, global_mask, kernel);
    // === [1] KLT Tracking
    std::vector<cv::KeyPoint> pts = currFrame_->getKeyPoints();
    cv::Mat previm = prevFrame_->getIm();
    std::vector<long unsigned int> lost_mappoint_ids;

    // PointReuse(currIm_, global_mask, lost_mappoint_ids);
    // std::cout << "LOST MAP IDS" << lost_mappoint_ids.size() << std::endl;
    int nMatches = klt_tracker_.Track(currIm_, currFrame_->getKeyPoints(), currFrame_->LandmarkStatuses(),
                                      true, options_.klt_min_SSIM, global_mask);

    std::cout << "NMATCHES: " << nMatches << std::endl;

    std::vector<int> matches;
    matches.reserve(currFrame_->LandmarkStatuses().size());
    for (int i = 0; i < currFrame_->LandmarkStatuses().size(); i++)
    {
        auto status = currFrame_->LandmarkStatuses()[i];
        if (status == TRACKED)
        {
            matches.push_back(i);
            continue;
        }
        if (status == TRACKED_WITH_3D)
        {
            matches.push_back(i);
            continue;
        }
        matches.push_back(-1);
    }
    visualizer_->drawMatches(pts, previm, currFrame_->getKeyPoints(), currIm_, matches);
    klt_tracker_.SetReferenceImage(currIm_, currFrame_->getKeyPoints(), global_mask);
    prevFrame_->setIm(currIm_);
    // pose optimization
    int n_inliers = poseOnlyOptimization(*currFrame_);
    if (nMatches > 40)
    {
        float  avg=0;
        float nb=0;
        for (int i = 0; i < currFrame_->getMapPoints().size(); i++)
        {
            auto mappoint = currFrame_->getMapPoints()[i];
            if (pMap_->getNumberOfObservations(i) > 0){
                    avg+= pMap_->getNumberOfObservations(i);
                    nb++;
            }
        }
        avg= avg/nb;
        std::cout << "Number of frames sequence:"<< avg << std::endl; 
    }

    if (n_inliers < 10)
    {

        std::cout << "Not enough iniliers" << std::endl;
        return false;
    }

    std::cout << "Pose optimization done." << std::endl;

    Sophus::SE3f Tcwc = currFrame_->getPose();
    mapVisualizer_->updateCurrentPose(Tcwc);

    return true;
}

void TrackingKLT::updateMotionModel()
{
    motionModel_ = currFrame_->getPose() * prevFrame_->getPose().inverse();
}
