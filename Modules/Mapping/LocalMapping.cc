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

#include "Mapping/LocalMapping.h"
#include "Optimization/g2oBundleAdjustment.h"
#include "Matching/DescriptorMatching.h"
#include "Utils/Geometry.h"

using namespace std;

LocalMapping::LocalMapping() {

}

LocalMapping::LocalMapping(Settings& settings, std::shared_ptr<Map> pMap) {
    settings_ = settings;
    pMap_ = pMap;
}

void LocalMapping::doMapping(std::shared_ptr<KeyFrame> &pCurrKeyFrame) {
    //Keep input keyframe
    if (!pCurrKeyFrame) return;

    if (currKeyFrame_ && pCurrKeyFrame->getId() != currKeyFrame_->getId()) {
        prevKeyFrame_ = currKeyFrame_;
    }
    currKeyFrame_ = pCurrKeyFrame;

    //Remove redundant MapPoints
    //mapPointCulling();

    //Triangulate new MapPoints
    triangulateNewMapPoints();

    checkDuplicatedMapPoints();

    //Run a local Bundle Adjustment
    localBundleAdjustment(pMap_.get(),currKeyFrame_->getId());
}

void LocalMapping::mapPointCulling() {
    /*
     * Your code for Lab 4 - Task 4 here!
     */
    std::vector<ID> removeIdx;
    for (auto entry : pMap_->getMapPoints()) {
        ID idx = std::get<0>(entry);
        std::shared_ptr<MapPoint> pMP = std::get<1>(entry);
        int nKFObs = pMap_->getNumberOfObservations(idx);  // seen in keyframes
        int nFObs = pMP->getNumFramesHaveSeen();           // seen in frames

        // Condition 1: A pct of the frames that should see the point do see it
        float shouldSeePct = 0.2f;
        int nFShouldSee = pMP->getNumFramesShouldSee();
        bool c1 = nFShouldSee >= 8 && nFObs <= int(nFShouldSee * shouldSeePct);

        // Condition 2: It has not been seen in the 2 following KeyFrames
        // after its triangulation (equivalent to nobs == 2 and not seen in
        // current or previous)
        bool c2 =
            nKFObs <= 2 && currKeyFrame_ && prevKeyFrame_ &&
            pMap_->isMapPointInKeyFrame(idx, currKeyFrame_->getId()) == -1 &&
            pMap_->isMapPointInKeyFrame(idx, prevKeyFrame_->getId()) == -1;

        if (c1 || c2) {
            removeIdx.push_back(idx);
        }
    }
    for (ID idx : removeIdx) {
        pMap_->removeMapPoint(idx);
    }
}

void LocalMapping::triangulateNewMapPoints() {
    //Get a list of the best covisible KeyFrames with the current one
    vector<pair<ID,int>> vKeyFrameCovisible = pMap_->getCovisibleKeyFrames(currKeyFrame_->getId());

    vector<int> vMatches(currKeyFrame_->getMapPoints().size());

    //Get data from the current KeyFrame
    shared_ptr<CameraModel> calibration1 = currKeyFrame_->getCalibration();
    Sophus::SE3f T1w = currKeyFrame_->getPose();

    int nTriangulated = 0;

    for(pair<ID,int> pairKeyFrame_Obs : vKeyFrameCovisible){
        int commonObservations = pairKeyFrame_Obs.second;
        if(commonObservations < 20)
            continue;

        shared_ptr<KeyFrame> pKF = pMap_->getKeyFrame(pairKeyFrame_Obs.first);
        if(pKF->getId() == currKeyFrame_->getId())
            continue;

        /*** Lab 4 - Task 2: read calibration of 2nd KF                   ***/
        /*** It should be the same as calibration1 for the monocular case ***/
        shared_ptr<CameraModel> calibration2 = pKF->getCalibration();

        //Check that baseline between KeyFrames is not too short
        Eigen::Vector3f vBaseLine = currKeyFrame_->getPose().inverse().translation() - pKF->getPose().inverse().translation();
        float medianDepth = pKF->computeSceneMedianDepth();
        float ratioBaseLineDepth = vBaseLine.norm() / medianDepth;

        if(ratioBaseLineDepth < 0.001){
            continue;
        }

        Sophus::SE3f T2w = pKF->getPose();

        Sophus::SE3f T21 = T2w*T1w.inverse();
        Eigen::Matrix<float,3,3> E = computeEssentialMatrixFromPose(T21);

        //Match features between the current and the covisible KeyFrame
        //TODO: this can be further improved using the orb vocabulary
        int nMatches = searchForTriangulation(currKeyFrame_.get(),pKF.get(),settings_.getMatchingForTriangulationTh(),
                settings_.getEpipolarTh(),E,vMatches);

        vector<cv::KeyPoint> vTriangulated1, vTriangulated2;
        vector<int> vMatches_;
        //Try to triangulate a new MapPoint with each match
        for(size_t i = 0; i < vMatches.size(); i++){
            if(vMatches[i] != -1){
                /*
                 * Your code for Lab 4 - Task 2 here!
                 * Note that the last KeyFrame inserted is stored at this->currKeyFrame_
                 */
                cv::Point2f p1 = currKeyFrame_->getKeyPoint(i).pt;
                Eigen::Vector3f ray1 = calibration1->unproject(p1).normalized();

                cv::Point2f p2 = pKF->getKeyPoint(vMatches[i]).pt;
                Eigen::Vector3f ray2 = calibration2->unproject(p2).normalized();

                // Condition 1: has been observed with enough parallax
                float cos = cosRayParallax(T21 * ray1, ray2);
                if (cos > settings_.getMinCos()) {
                    continue;
                }

                Eigen::Vector3f x3D;
                triangulate(ray1, ray2, T1w, T2w, x3D);

                // Condition 2: triangulated in front of both cameras
                Eigen::Vector3f x3D1 = T1w * x3D;
                Eigen::Vector3f x3D2 = T2w * x3D;
                if (x3D1.z() <= 0.0f || x3D2.z() <= 0.0f) {
                    continue;
                }

                // Condition 3: point reprojection error is low
                int maxReproj = settings_.getMaxReprojError();
                cv::Point2f p2D1 = calibration1->project(x3D1);
                if (squaredReprojectionError(p1, p2D1) > maxReproj) {
                    continue;
                }
                cv::Point2f p2D2 = calibration2->project(x3D2);
                if (squaredReprojectionError(p2, p2D2) > maxReproj) {
                    continue;
                }

                // Add to map and keyframe observations
                std::shared_ptr<MapPoint> pMP = std::make_shared<MapPoint>(x3D);
                currKeyFrame_->setMapPoint(i, pMP);
                pKF->setMapPoint(vMatches[i], pMP);
                pMap_->insertMapPoint(pMP);
                pMap_->addObservation(currKeyFrame_->getId(), pMP->getId(), i);
                pMap_->addObservation(pKF->getId(), pMP->getId(), vMatches[i]);
            }
        }
    }
}

void LocalMapping::checkDuplicatedMapPoints() {
    vector<pair<ID,int>> vKFcovisible = pMap_->getCovisibleKeyFrames(currKeyFrame_->getId());
    vector<shared_ptr<MapPoint>> vCurrMapPoints = currKeyFrame_->getMapPoints();

    for(int i = 0; i < vKFcovisible.size(); i++){
        if(vKFcovisible[i].first == currKeyFrame_->getId())
            continue;
        int nFused = fuse(pMap_->getKeyFrame(vKFcovisible[i].first),settings_.getMatchingFuseTh(),vCurrMapPoints,pMap_.get(), settings_);
        pMap_->checkKeyFrame(vKFcovisible[i].first);
        pMap_->checkKeyFrame(currKeyFrame_->getId());
    }
}
