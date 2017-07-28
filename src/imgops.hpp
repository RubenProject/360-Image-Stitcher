#ifndef imgopsHVar
#define imgopsHVar

using namespace std;
using namespace cv;

void correctShift(Mat& img);

void bundleAdjustment(Mat& src, double factor);

Mat translateImg(Mat src, int y);

void joinImgs(Mat A, Mat B, Mat& out, int x);

Mat rotateImg(Mat src, double angle);

#endif
