
#ifndef JJSLAM_SHITOMASI_H
#define JJSLAM_SHITOMASI_H

#include "Feature.h"

#include <opencv2/opencv.hpp>

class ShiTomasi : public Feature
{
public:
    void extract(const cv::Mat &im, std::vector<cv::KeyPoint> &vKeys) override;
    struct Options
    {
        int non_max_suprresion_window_size = 5;
    };

    ShiTomasi() = delete;

    ShiTomasi(Options &options);

    ~ShiTomasi();

    void SetnonMaxSuppresionWindow(int window_Size);

    private:
        void ResizeBuffers(cv::Size new_size);
    
        void ComputeScores(const cv::Mat& im);
    
        int GetKeyPoints(std::vector<cv::KeyPoint>& keypoints);
        int GetKeyPoints(std::vector<cv::KeyPoint>& keypoints,
                         std::vector<cv::KeyPoint>& keypoints_already_extracted);
        int GetPoints(std::vector<cv::Point2f>& points);
        int GetPoints(std::vector<cv::Point2f>& points,
                      std::vector<cv::KeyPoint>& keypoints_already_extracted);
    
        void ComputePyramidInfo();
        void ComputePyramid();
    
        //Compute image gradients and, at the same time, compute
        //Sho-Tomasi scores for each pixel. On this way, we are only
        //iterating over the image once.
        void FastSobelXYandScore(const cv::Mat& im);
    
        //Computes Shi-Tomasi score for a given pixel
        void DetectCorner(int col);
    
        //Computes the lower eigen value for the spatial tensor
        inline void ComputeMinEigenValue();
    
        bool IsLocalMaximum(int r, int c);
    
        unsigned int next_feature_id_ = 0;
    
        //----------------------------------------
        //              BASIC FIELDS
        //----------------------------------------
        cv::Mat Xgrad, Ygrad;
        cv::Mat scores;
        int non_max_suppresion_window_;
    
        int nrows, ncols;
    
        //----------------------------------------
        //      RUN-TIME OPTIMIZATION FIELDS
        //----------------------------------------
    
        //########################################
        //Pointers for fast matrix iteration
        //########################################
        //Grayscale image pointer
        const uchar* pIm[3];
    
        //Gradient pointers
        short* pXgrad[3], *pYgrad[3];
    
        //Shi-Tomasi pixel score
        float* pScore;
        //########################################
    
        //########################################
        //Speed up gradient computation variables
        //########################################
        short c1, c2, c3;
        std::vector<short> r1, r2, r3;
        //########################################
    
        //########################################
        //Other variables
        //########################################
    
        //Value to store eigen values of the current pixel's tensor.
        //Avoid to allocate memory at each iteration
        float eigen_value;
    
        //Vector to store actual tensor -> [G11 G12 G22]
        //Avoid to allocate memory at each iteration
        float tensor[3];
    
        //Short term column data [c11 + c12 + c13]-> used only when iterating over a row
        float _G11_c1, _G12_c1, _G22_c1;
        float _G11_c2, _G12_c2, _G22_c2;
    
        //Max score found
        float max_score, R;
    
        //Avoid a division
        float inv_size =  1.f / 9.f;
};

#endif