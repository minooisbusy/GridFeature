#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;

int main(int argc, char*argv[])
{
    Mat image = imread(argv[1], cv::IMREAD_GRAYSCALE);
    const int EDGE_THERSHOLD = 19;
    const float scale = 1.0f;///1.2f;
    Size sz = Size(cvRound(image.cols*scale),cvRound(image.rows*scale));
    // 9+6 = 15
    Size wholeSize = Size(sz.width+EDGE_THERSHOLD*2, sz.height + EDGE_THERSHOLD*2);
    Mat temp(wholeSize, image.type()); // gray_buf
    Mat pyramid = temp(Rect(EDGE_THERSHOLD,EDGE_THERSHOLD,sz.width, sz.height)); // gray

    copyMakeBorder(image, temp, EDGE_THERSHOLD,EDGE_THERSHOLD,EDGE_THERSHOLD,EDGE_THERSHOLD,
            BORDER_REFLECT101);

    for(int i=temp.rows/2-100; i<temp.rows/2 +100; i++)
    for(int j=temp.cols/2-100; j<temp.cols/2 +100; j++)
    temp.at<uchar>(i,j) = 0;
    imshow("temp", temp);
    imshow("pyramid", pyramid);
    waitKey(0);
    std::cout << temp.size << std::endl;
    std::cout << pyramid.size <<std::endl;

    return 0;
}