#ifndef FEATURE_H
#define FEATURE_H
#include<opencv2/opencv.hpp>

class ORBextractor
{
public:
    ORBextractor(){};
    operator(cv::Mat image);
};

#endif