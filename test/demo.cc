#include "feature.h"
#include<iostream>

int main(int argc, char* argv[])
{
    if(argc<2){
        std::cout<<"argument error"<<std::endl;
        return 0;
    } 
    cv::Mat image = cv::imread(argv[1]);//cv::Mat::zeros(256,256,CV_8UC1);
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
    
    ORBextractor extractor(nFeatures, scaleFactor, nLevels,
                            iniThFAST, minThFAST);


}