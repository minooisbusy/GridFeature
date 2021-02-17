#include "frame.h"

using namespace cv;
using namespace std;
namespace GRID
{
long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBextractorLeft(frame.mpORBextractorLeft),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb),N(frame.N), mvKeys(frame.mvKeys),
     mvKeysUn(frame.mvKeysUn),
     mDescriptors(frame.mDescriptors.clone()),
     mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];
}
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, std::shared_ptr<ORBextractor>& extractor, cv::Mat &K, cv::Mat &distCoef, const float &bf)
    :mpORBextractorLeft(extractor), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf)
{
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    ExtractORB(0, imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;
    
    UndistortKeyPoints();

    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations = false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    // nReserve means the number of feature in each GRID
    // They holds at least each GRID with 50% feature
    // In short, each cell of mGrid stores indice of undistored keypoints
    int nReserve = 0.5f*N / (FRAME_GRID_ROWS*FRAME_GRID_COLS);
    for(uint i = 0; i < FRAME_GRID_COLS; i++)
        for(uint j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    for(int i = 0; i < N; i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }

}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY)*mfGridElementHeightInv);

    // Keypoint's coordinates are undistored, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    //mpORBextractorLeft->operator()(im, cv::Mat(),mvKeys,mDescriptors);
    (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);

}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0) == 0.0)
    {
        mvKeysUn = mvKeys;
        return;
    }

    cv::Mat mat(N, 2, CV_32F);
    for(int i = 0; i < N; i++)
    {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    // Fill undistored keypoint vector
    mvKeysUn.resize(N);
    for(int i = 0; i < N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];   
        kp.pt.x =  mat.at<float>(i, 0);
        kp.pt.y =  mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2, CV_32F); // stores UL, UR, BL, BR
        mat.at<float>(0,0) = 0.0; mat.at<float>(0,1) = 0.0;
        mat.at<float>(1,0) = imLeft.cols; mat.at<float>(1,1) = 0.0;
        mat.at<float>(2,0) = 0.0; mat.at<float>(2,1) = imLeft.rows;
        mat.at<float>(3,0) = imLeft.cols; mat.at<float>(3,1) = imLeft.rows;

        //Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat,mat,mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0)); // min(UL.x, BL.x)
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0)); // min(UR.x, BR.x)
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**
 * @brief Get features in specific area(cell)
 * @param x point x-coordinate
 * @param y point y-coordinate
 * @param r window Size
 * @param minLevel the octave of minimum compare
 * @param maxLevel the octave of maximum compare
*/
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    //This codes filter out bounded cell index
    // mfGridElementWidthInv: The valuable that multiplied this variable become grid coordinate
    const int nMinCellX = max(0, static_cast<int>(floor(x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX >= FRAME_GRID_COLS)
        return vIndices; // Not in grid cells
    
    const int nMaxCellX = min(static_cast<int>(FRAME_GRID_COLS-1), static_cast<int>(ceil((x-mnMinX+r)*mfGridElementWidthInv)));
    if(nMaxCellX < 0)
        return vIndices;
    
    const int nMinCellY = max(0, static_cast<int>(floor(y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY >= FRAME_GRID_ROWS)
        return vIndices; // Not in grid cells
    
    const int nMaxCellY = min(static_cast<int>(FRAME_GRID_ROWS-1), static_cast<int>(ceil((y-mnMinY+r)*mfGridElementHeightInv)));
    if(nMaxCellY < 0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) ||(maxLevel>=0) ;


    for(int ix = nMinCellX; ix <= nMaxCellX; ix++) // Grid x-coordinate
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy]; // returns indices of features in grid cell;

            if(vCell.empty())
                continue;
            
            for(size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave < minLevel)
                        continue;
                    if(maxLevel >= 0)
                       if(kpUn.octave>maxLevel) 
                        continue;
                }

                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
                
            }
        }
    }
    
    return vIndices;
}
}