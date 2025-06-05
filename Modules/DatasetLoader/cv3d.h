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

 #ifndef NRSLAM_CV3D_H
 #define NRSLAM_CV3D_H
 
 #include <string>
 
 #include <opencv2/opencv.hpp>

 #include <map>
 
 class CV3D {
 public:
     // Loads the dataset stored at path. If the video has not been previously split, it splits it. Otherwise just loads
     // the images.
     CV3D(const std::string& dataset_path);
 
     // Retrieves the ith image in the sequence.
     cv::Mat GetImage(const int idx);
     std::string GetImageFile(const int idx);
     int getLenght();
     private:
        std::map<int, std::string>  images_names_; //Vector with the image paths
        int size_map_=0;
        bool GetFrames(std::string dataset_path);
 
 };
 
 
 #endif //NRSLAM_ENDOMAPPER_H
 