#ifndef FRAME_H
#define FRAME_H
#include "feature.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
namespace GRID
{

    class Frame
    {
    public:
        Frame();

        //Copy Constructor
        Frame(const Frame &frame);

        // Comstructor for Monocular cameras
        Frame(const cv::Mat &imGray, const double &timeStamp, std::shared_ptr<ORBextractor> &extractor, cv::Mat &K, cv::Mat &distCoef, const float &bf);

        void ExtractORB(int flag, const cv::Mat &im);

        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

        std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = -1, const int maxLevel = -1) const;

    public:
        std::shared_ptr<ORBextractor> mpORBextractorLeft;

        double mTimeStamp;
        static unsigned long int nNextId;

        //Calibration matrix and OpenCV distortion parameters
        cv::Mat mK;
        static float fx; // static member shared at all instances
        static float fy;
        static float cx;
        static float cy;
        static float invfx;
        static float invfy;
        cv::Mat mDistCoef;

        float mbf;

        float mb;

        // Number of KeyPoints
        int N;

        std::vector<cv::KeyPoint> mvKeys;
        std::vector<cv::KeyPoint> mvKeysUn;

        cv::Mat mDescriptors;

        static float mfGridElementWidthInv;
        static float mfGridElementHeightInv;

        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

        int mnScaleLevels;
        float mfScaleFactor;
        float mfLogScaleFactor;
        std::vector<float> mvScaleFactors;
        std::vector<float> mvInvScaleFactors;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        // Undistored Image Bounds (computed once)
        static float mnMinX;
        static float mnMaxX;
        static float mnMinY;
        static float mnMaxY;

        static bool mbInitialComputations;

    private:
        void UndistortKeyPoints();
        void ComputeImageBounds(const cv::Mat &imLeft);

        void AssignFeaturesToGrid();
    };
} // namespace GRID
#endif