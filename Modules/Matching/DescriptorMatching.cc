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

#include "DescriptorMatching.h"

using namespace std;

int HammingDistance(const cv::Mat &a, const cv::Mat &b){
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++){
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

int kltMatching(Frame& refFrame, Frame& currFrame, int th, std::vector<int>& vMatches) {
    const std::vector<cv::KeyPoint>& vRefKeys = refFrame.getKeyPoints();
    const std::vector<cv::KeyPoint>& vCurrKeys = currFrame.getKeyPoints();

    cv::Mat refDesc = refFrame.getDescriptors();
    cv::Mat currDesc = currFrame.getDescriptors();

    // Sanity checks
    if (refDesc.empty() || currDesc.empty()) {
        std::cerr << "ERROR: Descriptor matrix is empty!\n";
        return 0;
    }

    if (refDesc.rows != vRefKeys.size() || currDesc.rows != vCurrKeys.size()) {
        std::cerr << "ERROR: Descriptor size mismatch with keypoints!\n";
        return 0;
    }

    vMatches.clear();
    vMatches.resize(vRefKeys.size(), -1);

    int nMatches = 0;

    for (size_t i = 0; i < vRefKeys.size(); i++) {
        const cv::Mat desc = refDesc.row(i);

        int bestDist = 256, secondBestDist = 256;
        int bestIdx = -1;

        for (size_t j = 0; j < vCurrKeys.size(); j++) {
            int dist = HammingDistance(desc, currDesc.row(j));
            if (dist < bestDist) {
                secondBestDist = bestDist;
                bestDist = dist;
                bestIdx = j;
            } else if (dist < secondBestDist) {
                secondBestDist = dist;
            }
        }

        if (bestDist <= th && bestDist < 0.9f * secondBestDist) {
            vMatches[i] = bestIdx;
            ++nMatches;
        }
    }

    std::cout << "KLT Matching: Found " << nMatches << " matches." << std::endl;
    return nMatches;
}

int searchForInitializaion(Frame& refFrame, Frame& currFrame, int th, vector<int>& vMatches, std::vector<cv::Point2f>& vPrevMatched){
    fill(vMatches.begin(),vMatches.end(),-1);

    vector<size_t> vIndicesToCheck(100);

    vector<cv::KeyPoint>& vRefKeys = refFrame.getKeyPoints();

    vector<int> vMatchedDistance(vRefKeys.size(),INT_MAX);
    vector<int> vnMatches21(vRefKeys.size(),-1);

    cv::Mat refDesc = refFrame.getDescriptors();
    cv::Mat currDesc = currFrame.getDescriptors();

    int nMatches = 0;
    const int minOctave = 0, maxOctave = 0;
    for(size_t i = 0; i < vRefKeys.size(); i++){
        //Only search matches with KeyPoints in the finest scale
        if(vRefKeys[i].octave > maxOctave){
            continue;
        }

        //Clear previous matches
        vIndicesToCheck.clear();

        //Get last octave in which the point was seen
        int nLastOctave = vRefKeys[i].octave;
        //Search radius depends on the size of the point in the image
        float radius = 15 * 3 * currFrame.getScaleFactor(nLastOctave);
        //Get candidates whose coordinates are close to the current point
        currFrame.getFeaturesInArea(vRefKeys[i].pt.x, vRefKeys[i].pt.y, radius, nLastOctave-1, nLastOctave+1, vIndicesToCheck);

        cv::Mat desc = refDesc.row(i);

        //Match with the one with the smallest Hamming distance
        int bestDist = 255, secondBestDist = 255;
        size_t bestIdx;        
        for(auto j : vIndicesToCheck){

            int dist = HammingDistance(desc,currDesc.row(j));

            if(dist < bestDist){
                secondBestDist = bestDist;
                bestDist = dist;
                bestIdx = j;
            }
            else if(dist < secondBestDist){
                secondBestDist = dist;
            }
        }
        if(bestDist <= th && (float)bestDist < (float(secondBestDist)*0.9)){
            vMatches[i] = bestIdx;
            vPrevMatched[i]=currFrame.getKeyPoint(bestIdx).pt;

            nMatches++;
        }


    }

    std::vector<cv::KeyPoint> vPrevMatched_= currFrame.getKeyPoints();

    for(size_t i = 0; i < vMatches.size(); i++){
        if(vMatches[i] != -1){
            vPrevMatched[i]=currFrame.getKeyPoint(vMatches[i]).pt;
        }
    }


        // Draw matches
        std::vector<cv::DMatch> matches;
        for(size_t i = 0; i < vMatches.size(); i++){
            if(vMatches[i] != -1){
                matches.push_back(cv::DMatch(i,vMatches[i],0));
            }
        }
        cv::Mat img_matches;
        cv::drawMatches(refFrame.getIm(), vRefKeys,currFrame.getIm(), vPrevMatched_, matches, img_matches);
    
        // Show result
        cv::imshow("Matches", img_matches);
        bool success = cv::imwrite("output.jpg", img_matches);
    return nMatches;
}

int guidedMatching(Frame& refFrame, Frame& currFrame, int th, std::vector<int>& vMatches, int windowSizeFactor){
    currFrame.clearMapPoints();
    fill(vMatches.begin(),vMatches.end(),-1);
    vector<size_t> vIndicesToCheck(100);

    vector<shared_ptr<MapPoint>> vRefMapPoints = refFrame.getMapPoints();

    shared_ptr<CameraModel> currCamera = currFrame.getCalibration();
    Sophus::SE3f Tcw = currFrame.getPose();

    int nMatches = 0;
    for(size_t i = 0; i < vRefMapPoints.size(); i++){
        if(!vRefMapPoints[i]){
            continue;
        }

        //Project map point into the current frame
        Eigen::Vector3f p3Dcam = Tcw * vRefMapPoints[i]->getWorldPosition();
        cv::Point2f uv = currCamera->project(p3Dcam);

        //Clear previous matches
        vIndicesToCheck.clear();

        //Get last octave in which the point was seen
        int nLastOctave = refFrame.getKeyPoint(i).octave;

        //Search radius depends on the size of the point in the image
        float radius = 15 * windowSizeFactor * currFrame.getScaleFactor(nLastOctave);

        //Get candidates whose coordinates are close to the current point
        currFrame.getFeaturesInArea(uv.x,uv.y,radius,nLastOctave-1,nLastOctave+1,vIndicesToCheck);

        cv::Mat desc = vRefMapPoints[i]->getDescriptor();

        //Match with the one with the smallest Hamming distance
        int bestDist = 255, secondBestDist = 255;
        size_t bestIdx;
        for(auto j : vIndicesToCheck){
            if(currFrame.getMapPoint(j)){
                continue;
            }

            int dist = HammingDistance(desc.row(0),currFrame.getDescriptors().row(j));

            if(dist < bestDist){
                secondBestDist = bestDist;
                bestDist = dist;
                bestIdx = j;
            }
            else if(dist < secondBestDist){
                secondBestDist = dist;
            }
        }
        if(bestDist <= th && (float)bestDist < (float(secondBestDist)*0.9)){
            vMatches[i] = bestIdx;
            currFrame.setMapPoint(bestIdx,vRefMapPoints[i]);
            nMatches++;
        }
    }

    return nMatches;
}

int searchWithProjection(Frame& currFrame, int th, std::vector<std::shared_ptr<MapPoint>>& vMapPoints){
    CameraModel* currCalibration = currFrame.getCalibration().get();
    Sophus::SE3f Tcw = currFrame.getPose();

    vector<size_t> vIndicesToCheck(100);

    int nMatches = 0, vCos = 0, invDist = 0, noClose = 0;
    for(shared_ptr<MapPoint> pMP : vMapPoints){
        //Clear previous matches
        vIndicesToCheck.clear();

        //Check normal orientation
        assert(pMP);
        Eigen::Vector3f rayWorld = pMP->getWorldPosition() - currFrame.getPose().inverse().translation();
        float viewCos = rayWorld.normalized().dot(pMP->getNormalOrientation());

        if(viewCos < 0.5){
            vCos++;
            continue;
        }

        //Check distance is in the scale invariance region of the MapPoint
        float dist = rayWorld.norm();
        float maxDistance = pMP->getMaxDistanceInvariance();
        float minDistance = pMP->getMinDistanceInvariance();

        if(dist < minDistance || dist > maxDistance){
            invDist++;
            continue;
        }

        //Predict scale
        int predictedOctave = ceil(log(maxDistance/dist)/log(currFrame.getScaleFactor(1)));
        if(predictedOctave < 0)
            predictedOctave = 0;
        else if(predictedOctave > currFrame.getNumberOfScales())
            predictedOctave = currFrame.getNumberOfScales();

        //Project MapPoint into the Frame
        Eigen::Vector3f p3Dc = Tcw*pMP->getWorldPosition();
        cv::Point2f uv = currCalibration->project(p3Dc);

        float radius = currFrame.getScaleFactor(predictedOctave);
        if(viewCos>0.998)
            radius *= 2.5;
        else
            radius *= 4.0;


        currFrame.getFeaturesInArea(uv.x,uv.y,radius,predictedOctave-1,predictedOctave+1,vIndicesToCheck);

        cv::Mat desc = pMP->getDescriptor();

        //Match with the one with the smallest Hamming distance
        int bestDist = 255, secondBestDist = 255;
        size_t bestIdx;
        int hd = 0;
        auto all_descr = currFrame.getDescriptors();

        for(auto j : vIndicesToCheck){
            if(currFrame.getMapPoint(j)){
                continue;
            }

            hd = HammingDistance(desc.row(0),all_descr.row(j));

            if(hd < bestDist){
                secondBestDist = bestDist;
                bestDist = hd;
                bestIdx = j;
            }
            else if(hd < secondBestDist){
                secondBestDist = hd;
            }
        }

        if(bestDist <= th && (float)bestDist < (float(secondBestDist)*0.9)){
            currFrame.setMapPoint(bestIdx,pMP);

            nMatches++;
        }
        else{
            noClose++;
        }

    }

    return nMatches;
}

int searchForTriangulation(KeyFrame* kf1, KeyFrame* kf2, int th, float fEpipolarTh, Eigen::Matrix<float,3,3>& E, std::vector<int>& vMatches){
    //Clear matches
    fill(vMatches.begin(),vMatches.end(),-1);

    //Retrieve KeyFrame descriptors
    cv::Mat& desc1 = kf1->getDescriptors();
    cv::Mat& desc2 = kf2->getDescriptors();

    //Get MaPoints
    vector<shared_ptr<MapPoint>>& vMapPoints1 = kf1->getMapPoints();
    vector<shared_ptr<MapPoint>>& vMapPoints2 = kf2->getMapPoints();

    shared_ptr<CameraModel> calibration = kf1->getCalibration();

    vector<int> vMatchedDistance(vMapPoints1.size(),INT_MAX);
    vector<int> vnMatches21(vMapPoints1.size(),-1);

    vector<bool> vbMatched2(vMapPoints2.size(),false);

    int nMatches = 0;
    for(size_t i = 0; i < vMapPoints1.size(); i++){
        //Check that there is no MapPoint already triangulated
        if(vMapPoints1[i]){
            continue;
        }

        //Unrpoject KeyPoint
        Eigen::Vector3f ray1 = (E*calibration->unproject(kf1->getKeyPoint(i).pt).transpose()).normalized();

        //Search a match in the other KeyFrame
        int bestDist = 255, secondBestDist = 255;
        size_t bestIdx;
        for(size_t j = 0; j < vMapPoints2.size(); j++){
            if(vMapPoints2[j] || vbMatched2[2])
                continue;

            int dist = HammingDistance(desc1.row(i),desc2.row(j));

            if(dist > 50 || dist > bestDist)
                continue;

            if(vMatchedDistance[j] <= dist)
                continue;

            //Unproject KeyPoint
            Eigen::Vector3f ray2 = calibration->unproject(kf2->getKeyPoint(j).pt).normalized();

            //Check epipolar constrain
            float err = fabs(M_PI/2 - acos(ray2.dot(ray1)));
            if(err < fEpipolarTh){
                bestDist = dist;
                bestIdx = j;
            }
        }

        if(bestDist < th){
            if(vMapPoints2[bestIdx])
                continue;

            if(vnMatches21[bestIdx]>=0){
                vMatches[vnMatches21[bestIdx]]=-1;
                nMatches--;
            }

            vMatches[i] = bestIdx;
            vnMatches21[bestIdx]= i;
            vbMatched2[bestDist] = true;
            vMatchedDistance[bestIdx]=bestDist;
            nMatches++;
        }
    }

    return nMatches;
}

int fuse(std::shared_ptr<KeyFrame> pKF, int th, std::vector<std::shared_ptr<MapPoint>>& vMapPoints, Map* pMap){
    Sophus::SE3f Tcw = pKF->getPose();
    shared_ptr<CameraModel> calibration = pKF->getCalibration();

    vector<size_t> vIndicesToCheck(100);

    vector<shared_ptr<MapPoint>>& vKFMps = pKF->getMapPoints();

    cv::Mat descMat = pKF->getDescriptors();
    int nFused = 0;

    for(size_t i = 0; i < vMapPoints.size(); i++){
        //Clear previous matches
        vIndicesToCheck.clear();

        shared_ptr<MapPoint> pMP = vMapPoints[i];

        if(!pMP)
            continue;

        if(pMap->getNumberOfObservations(pMP->getId()) == 0)
            continue;

        if(pMap->isMapPointInKeyFrame(pMP->getId(),pKF->getId()) != -1){
            continue;
        }


        // From here: task3

        //Project MapPoint into the Frame
        Eigen::Vector3f p3Dc = Tcw*pMP->getWorldPosition();
        //check if its in front of the camera (Sanity check)
        if(p3Dc.z() < 0)
        {
            continue;
        }

        cv::Point2f uv = calibration->project(p3Dc);
        //Check if projection is inside the image (Sanity check)

        //Search radius depends on the size of the point in the image
        //get the octave of the keypoint
        int octave = pKF->getKeyPoint(i).octave;
        float radius = 15 * pKF->getScaleFactor(octave);
        //Get candidates whose coordinates are close to the current point
        pKF->getFeaturesInArea(uv.x,uv.y,radius,octave-1,octave+1,vIndicesToCheck);

        //Compare descriptors using hamming distance
        cv::Mat desc = pMP->getDescriptor();
        //Match with the one with the smallest Hamming distance
        int bestDist = 255, secondBestDist = 255;
        size_t bestIdx;
        for(auto j : vIndicesToCheck){
            //Check if the point is already associated to a map point
            if(pKF->getMapPoints()[j]){
                continue;
            }

            int dist = HammingDistance(desc.row(0),descMat.row(j));

            if(dist < bestDist){
                secondBestDist = bestDist;
                bestDist = dist;
                bestIdx = j;
            }
            else if(dist < secondBestDist){
                secondBestDist = dist;
            }
        }

        if (bestDist > th || (float)bestDist >= (float)secondBestDist * 0.9)
        {
            continue;
        }

        //Fuse the points or add an observation
        // If the matched KeyPoint has no MapPoint associated, just add the observation.
        // Otherwise fuse both MapPoints into a single one, and add the observation.
        // if (!pKF->getMapPoints()[bestIdx]) {
        //     //void Map::addObservation(ID kfId, ID mpId, size_t idx)
        //     pMap->addObservation(pKF->getId(), pMP->getId(), i);
        // } else {
        //     shared_ptr<MapPoint> pMP2 = pKF->getMapPoints()[bestIdx];
        //     pMap->fuseMapPoints(vpMPS[i]->getId(), pMP2->getId());
        //     pMap->addObservation(bestIdx, pMP2);
        // }

        //what Victor said in the email:
        if (pKF->getMapPoints()[bestIdx]) {
            pMap->fuseMapPoints(pMP->getId(), vKFMps[bestIdx]->getId());
        } else {
            pKF->setMapPoint(bestIdx, pMP);
            pMap->addObservation(pKF->getId(), pMP->getId(), bestIdx);
        }

    }

    return nFused;
}

