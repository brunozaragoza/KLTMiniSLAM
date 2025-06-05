#include "cv3d.h"


#include <boost/filesystem.hpp>


using namespace std;
CV3D::CV3D(const std::string &dataset_path) {
        if(GetFrames(dataset_path)){
            std::cout<<"Dataset  Loaded"<<std::endl;
        }
}

cv::Mat CV3D::GetImage(const int idx)
{
    if (idx >= size_map_) {
        std::cerr<< "Image index out boundaries.";
    }
    return cv::imread(images_names_[idx], cv::IMREAD_UNCHANGED);
}
std::string CV3D::GetImageFile(const int idx){
    if (idx >= size_map_) {
        return "";
    }
    return images_names_[idx];
}

bool CV3D::GetFrames(std::string dataset_path){

    if (!boost::filesystem::exists(dataset_path) || !boost::filesystem::is_directory(dataset_path))
    {
        std::cerr << "Invalid directory: " << dataset_path << std::endl;
        return 1;
    }

    // Iterate over the directory contents
    for (const auto &entry : boost::filesystem::directory_iterator(dataset_path))
    {
        if (boost::filesystem::is_regular_file(entry.path()) &&
            (entry.path().extension() == ".png" ||
             entry.path().extension() == ".jpg"))
        {
            std::string file= entry.path().filename().string();
            int idx_current = std::stoi(file);
            images_names_[idx_current]= entry.path().string();
            size_map_++;
        }
    }
    return true;

}

int CV3D::getLenght() {
    return size_map_;
}