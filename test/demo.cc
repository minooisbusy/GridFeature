#include<iostream>
#include<memory>
#include "feature.h"
#include "frame.h"
#include "matcher.h"

int main(int argc, char* argv[])
{
    for(int i=1; i<argc; i++)
        std::cout << "Input argument["<<i<<"]:"<<argv[i]<<std::endl;
    std::cout << "-- Run FEATURE Extractor" << std::endl;
    if(argc<2){
        std::cout<<"argument error"<<std::endl;
        return 0;
    } 
    cv::Mat im1 = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
    cv::Mat im2 = cv::imread(argv[2],cv::IMREAD_GRAYSCALE);
    if(im1.empty() || im2.empty())
    {
        std::cout <<"empty error"<<std::endl;
        return 0;
    }
    cv::FileStorage fSettings(argv[3], cv::FileStorage::READ);
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float scaleFactor = fSettings["ORBextractor.scaleFactor"];
    int iniThFAST = fSettings["ORBextractor.iniThFAST"];
    int minThFAST = fSettings["ORBextractor.minThFAST"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float k1 = fSettings["Camera.k1"];
    float k2 = fSettings["Camera.k2"];
    //float k3 = fSettings["Camera.k3"];
    float p1 = fSettings["Camera.p1"];
    float p2 = fSettings["Camera.k2"];
    float bf = fSettings["Camera.bf"];
    cv::Mat K = cv::Mat::ones(3,3,CV_32FC1);
    cv::Mat DistCoef(4,1,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    DistCoef.at<float>(0) = k1;
    DistCoef.at<float>(1) = k2;
    DistCoef.at<float>(2) = p1;
    DistCoef.at<float>(3) = p2;


    std::cout << "-- Feature Extractor Initialization... ";
    std::shared_ptr<GRID::ORBextractor> pExtractor = std::make_shared<GRID::ORBextractor>(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
    std::cout << "Done!"<<std::endl;

    std::cout << "-- Frame1 Initialization... ";
    GRID::Frame F1(im1, static_cast<double>(0), pExtractor, K, DistCoef,bf);
    std::cout << "Done!"<<std::endl;
    std::cout << "-- Frame2 Initialization... ";
    GRID::Frame F2(im2, static_cast<double>(1), pExtractor, K, DistCoef,bf);
    std::cout << "Done!"<<std::endl;

    std::cout << "-- Small allocation... ";
    std::vector<cv::Point2f> vbPrevMatched(F1.mvKeysUn.size());
    std::vector<int> vIniMatches;
    for(size_t i = 0; i <F1.mvKeysUn.size(); i++)
        vbPrevMatched[i] = F1.mvKeysUn[i].pt;
    std::fill(vIniMatches.begin(), vIniMatches.end(), -1);
    std::cout<<"Done!"<<std::endl;

    std::cout << "-- Matcher Initialization... ";
    GRID::ORBmatcher matcher(0.9, true);
    std::cout << "Done!"<<std::endl;;

    std::cout << "-- Matching... ";
    int nmatch = matcher.SearchForInitialization(F1,F2,vbPrevMatched, vIniMatches, 100);
    std::cout<<"Matched points = "<<nmatch<<std::endl;

    std::vector<cv::DMatch> vDm;
    std::vector<cv::KeyPoint> out;
    for(int i=0; i<nmatch;i++)
    {
       cv::DMatch dm;
       cv::KeyPoint kp;
       dm.imgIdx=0;
       dm.queryIdx = i;
       dm.trainIdx = i; 
       dm.distance = 0;
       kp.pt = vbPrevMatched[i];
       vDm.push_back(dm);
       out.push_back(kp);
    }
    cv::drawKeypoints(im1,F1.mvKeys,im1,cv::Scalar(0,255,0));
    cv::drawKeypoints(im2,F1.mvKeys,im2,cv::Scalar(0,255,0));
    std::cout << F1.mvKeys.size()<<std::endl;
    std::cout << F2.mvKeys.size()<<std::endl;
    std::cout << vIniMatches.size()<<std::endl;
    std::cout << vbPrevMatched.size()<<std::endl;
    cv::Mat I, J;
    cv::drawMatches(im1,F1.mvKeys,im2,out,vDm,I,cv::Scalar(0,255,0),cv::Scalar(0,0,255));
    cv::hconcat(im1,im2,J);
    
    imshow("d",I);
    imshow("ed",J);
    cv::waitKey(0);


}