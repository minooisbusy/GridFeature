#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<string>
using namespace std;
using namespace cv;

void ComputePyramid(const Mat& image);

int main(int argc, char* argv[])
{
    if(argc<2){
        cout<<"argument error"<<endl;
        return 0;
    } 
    Mat image = imread(argv[1]);//Mat::zeros(256,256,CV_8UC1);
    if(image.empty())
    {
        cout <<"empty error"<<endl;
        return 0;
    }
    const uint levels = 5; 
    const float scaleFactor = 1.2f;
    const uint EDGE_THRESHOLD = 19;
    std::vector<float> mvScaleFactor(5, 0.0f);
    std::vector<float> mvInvScaleFactor(5, 0.0f);
    vector<Mat> mvImagePyramid(5,Mat());

    mvInvScaleFactor[0] = 1.0f;
    mvScaleFactor[0] = 1.0f;
    for(int i = 1; i < levels; i++ )
     mvScaleFactor[i] = mvScaleFactor[i-1] * scaleFactor;

    for(int i = 0; i < levels; i++ )
     mvInvScaleFactor[i] = 1.0f/mvScaleFactor[i];
    for(int level = 0; level < levels; ++level)
    {
        cout << "level = " <<level <<endl;
        float scale = mvInvScaleFactor[level]; 
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        if( level != 0)
        {
            resize(mvImagePyramid[level-1], mvImagePyramid[level],sz, 0, 0, INTER_LINEAR);
            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED); // BORDER_ISOLATED: DO NOT LOOK OUTSIDE
        }
        else
        {
            // Just Padding
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
        }
    string name = to_string(level) + "-th image";
    imshow(name, mvImagePyramid[level]);
    cout<< "Size["<<level<<"]="<<mvImagePyramid[level].size<<endl;
    }
    waitKey(0);


}