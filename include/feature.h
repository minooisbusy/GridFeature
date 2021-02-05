#ifndef FEATURE_H
#define FEATURE_H
#include <opencv2/opencv.hpp>
#include<vector>

class ORBextractor
{
public:
ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         int _iniThFAST, int _minThFAST);
    ~ORBextractor() {}

    void operator()(cv::InputArray _image, cv::InputArray _mask,
        std::vector<cv::KeyPoint>& _keypoints, cv::OutputArray _descriptors);


        // Get methods
        int inline GetLevels(){ return nlevels; }

        float inline GetScaleFactor(){ return scaleFactor; }

        std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

        std::vector<float> inline GetInverseScaleFactors() { return mvInvScaleFactor; }

        std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

        std::vector<float> inline GetInverseScaleSigmaSquares() { return mvInvLevelSigma2; }

        std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(const cv::Mat &image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, 
                                                const int &minX, const int &maxX, 
                                                const int& minY, const int &maxY,
                                                const int &nFeatures, const int &level);
    std::vector<cv::Point> pattern;

    int nfeatures;
    float scaleFactor;
    float nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

#endif