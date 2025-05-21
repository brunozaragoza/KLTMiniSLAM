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

#include "VideoLoader.h"

#include <iostream>
#include <fstream>

#include <sys/stat.h>

using namespace std;

VideoLoader::VideoLoader(std::string folderPath, std::string timesPath) {
    ifstream fTimes;
    fTimes.open(timesPath.c_str());
    vTimeStamps_.reserve(5000);
    vImgsPairs_.reserve(5000);

    string sPath = "Images";

    if(!fTimes.is_open()){
        cerr << "[VideoLoader]: Could not load dataset at " << folderPath << endl;
        return;
    }

    while(!fTimes.eof()){
        string s;
        getline(fTimes,s);
        if(!s.empty()){
            stringstream ss;
            ss << s;
            // std::cout << s << std::endl;
            string timestamp = s + ".jpg";
            // std::cout << timestamp << std::endl;
            string imLeft = folderPath + "/" + sPath + "/" + timestamp;
            string imRight = folderPath + "/" + sPath + "/" + timestamp;
        
            vImgsPairs_.push_back(make_pair(imLeft, imRight));
            double t;
            ss >> t;
            vTimeStamps_.push_back(t/1e9);
        }
    }

    cv::Mat im;
    getLeftImage(0,im);
    imSize_ = im.size();

    fTimes.close();

}

bool VideoLoader::getLeftImage(size_t idx, cv::Mat& im) {
    if(idx >= vTimeStamps_.size()) return false;

    // cout << "[VideoLoader]: loading image at " << vImgsPairs_[idx].first << endl;
    im = cv::imread(vImgsPairs_[idx].first, cv::IMREAD_UNCHANGED);

    return true;
}

bool VideoLoader::getRightImage(size_t idx, cv::Mat& im) {
    if(idx >= vTimeStamps_.size()) return false;

    im = cv::imread(vImgsPairs_[idx].first, cv::IMREAD_UNCHANGED);

    return true;
}

bool VideoLoader::getTimeStamp(size_t idx, double &timestamp) {
    if(idx >= vTimeStamps_.size()) return false;

    timestamp = vTimeStamps_[idx];

    return true;
}

int VideoLoader::getLenght() {
    return (int)vTimeStamps_.size();
}

cv::Size VideoLoader::getImageSize() {
    return imSize_;
}

