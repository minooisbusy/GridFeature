#include "feature.h"
#include<iostream>

using namespace GRID;

int main(int argc, char* argv[])
{
    std::cout << "-- Run FEATURE Extractor" << std::endl;
    if(argc<2){
        std::cout<<"argument error"<<std::endl;
        return 0;
    } 
    cv::Mat image = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
    if(image.empty())
    {
        std::cout <<"empty error"<<std::endl;
        return 0;
    }
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float scaleFactor = fSettings["ORBextractor.scaleFactor"];
    int iniThFAST = fSettings["ORBextractor.iniThFAST"];
    int minThFAST = fSettings["ORBextractor.minThFAST"];
    int nLevels = fSettings["ORBextractor.nLevels"];

    std::cout << "-- Feature Extractor Initialization... ";
    ORBextractor extractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    extractor(image, cv::Mat(), kpts, desc);
    std::cout << "Complete!"<<std::endl;
}