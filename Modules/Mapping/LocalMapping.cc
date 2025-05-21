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
    currKeyFrame_ = pCurrKeyFrame;

    if(!currKeyFrame_)
        return;

    //Remove redundant MapPoints
    mapPointCulling();

    //Triangulate new MapPoints
    triangulateNewMapPoints();

    checkDuplicatedMapPoints();

    //Run a local Bundle Adjustment
    localBundleAdjustment(pMap_.get(),currKeyFrame_->getId());
}

void LocalMapping::mapPointCulling() {
    vector<shared_ptr<MapPoint>> vMapPoints = currKeyFrame_->getMapPoints();
    std::vector<std::pair<ID,int>> covisiblekeyfs=   pMap_->getCovisibleKeyFrames(currKeyFrame_->getId());
    for(shared_ptr<MapPoint> pMP : vMapPoints){
        if(!pMP)
            continue;

        int numObservations = pMap_->getNumberOfObservations(pMP->getId());

        int nkfs = 0; // Initialize to an invalid keyframe ID
 
         for (auto kf : covisiblekeyfs) {
             int idx = pMap_->isMapPointInKeyFrame(pMP->getId(), kf.first);
             if (idx != -1) {
                 nkfs++;;  // Keep track of the latest keyframe ID
             }
         }
         
        //less than 3 observations and avoid deleting the beggining of the map
         // Check if it is more than 25% of covisible keyframes 
         float ratio= (float)nkfs/(float)covisiblekeyfs.size();
         if ((ratio - 0.05<0.0001 & covisiblekeyfs.size()>5)) {
            // std::cout << "MapPoint ID: " << pMP->getId() << " has not been seen in the last frames." << std::endl;
            pMap_->removeMapPoint(pMP->getId());
            continue;
         }
        if((numObservations <3  && currKeyFrame_->getId()>5) ){
            std::cout << "MapPoint ID: " << pMP->getId() << " has not been 3 observations." << std::endl;
            
            pMap_->removeMapPoint(pMP->getId());
        }

        
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

        //Check that baseline between KeyFrames is not too short
        Eigen::Vector3f vBaseLine = currKeyFrame_->getPose().inverse().translation() - pKF->getPose().inverse().translation();
        float medianDepth = pKF->computeSceneMedianDepth();
        float ratioBaseLineDepth = vBaseLine.norm() / medianDepth;

        if(ratioBaseLineDepth < 0.01){
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

                cv::KeyPoint kp1 = currKeyFrame_->getKeyPoint(i);
                cv::KeyPoint kp2 = pKF->getKeyPoint(vMatches[i]);

                //Store the points in the vectors
                vTriangulated1.push_back(kp1);
                vTriangulated2.push_back(kp2);

                //get the rays in each camera frame NORMALIZED
                Eigen::Vector2f uv1(kp1.pt.x, kp1.pt.y);
                Eigen::Vector2f uv2(kp2.pt.x, kp2.pt.y);

                Eigen::Vector3f ray1, ray2;
                calibration1->unproject(uv1, ray1);
                pKF->getCalibration()->unproject(uv2, ray2);

                //1st check: Parallax between rays (with respect one camera frame)
                Eigen::Vector3f ray21 = T21 * ray2;

                ray1 = ray1.normalized();
                ray2 = ray2.normalized();
                ray21 = ray21.normalized();


                if (cosRayParallax(ray1, ray21) > settings_.getMinCos()) {
                    std::cout << "Parallax check failed" << std::endl;
                    continue;
                }

                //Triangulation
                Eigen::Vector3f p3D;        //Point in world coordinates
                triangulate(ray1, ray2, T1w, T2w, p3D);

                //2nd check: Depth consistency
                Eigen::Vector3f p3D_c1 = T1w * p3D;
                Eigen::Vector3f p3D_c2 = T2w * p3D;
                // std::cout << p3D_c1.z() << ", " << p3D_c2.z() << std::endl;
                if (p3D_c1.z() < 0 || p3D_c2.z() < 0) {
                    std::cout << "Depth consistency check failed" << std::endl;
                    continue;
                }

                //3rd check: Reprojection error
                cv::Point2f uv1_reproj = calibration1->project(p3D_c1);
                cv::Point2f uv2_reproj = pKF->getCalibration()->project(p3D_c2);

                float reproj_error_1 = squaredReprojectionError(kp1.pt, uv1_reproj);
                float reproj_error_2 = squaredReprojectionError(kp2.pt, uv2_reproj);

                // std::cout << "Reprojection error 1: " << reproj_error_1 << std::endl;
                if (reproj_error_1 >= 5 || reproj_error_2 >= 5) {
                    std::cout << "Reprojection error check failed" << std::endl;
                    continue;
                }

                //Create a new MapPoint
                std::shared_ptr<MapPoint> pMP = std::make_shared<MapPoint>(p3D);

            

                pMap_->insertMapPoint(pMP);

                //Assign the new MapPoint to the KeyFrames and add the observation
                currKeyFrame_->setMapPoint(i,pMP);
                pKF->setMapPoint(vMatches[i],pMP);
                
                pMap_->addObservation(currKeyFrame_->getId(),pMP->getId(),i);
                pMap_->addObservation(pKF->getId(),pMP->getId(),vMatches[i]);
                
                nTriangulated++;

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
        int nFused = fuse(pMap_->getKeyFrame(vKFcovisible[i].first),settings_.getMatchingFuseTh(),vCurrMapPoints,pMap_.get());
        pMap_->checkKeyFrame(vKFcovisible[i].first);
        pMap_->checkKeyFrame(currKeyFrame_->getId());
    }
}
