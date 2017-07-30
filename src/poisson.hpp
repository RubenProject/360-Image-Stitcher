#ifndef poissonHVar
#define poissonHVar

using namespace cv;

Mat getGradientXp(Mat &img);

Mat getGradientYp(Mat& img);

Mat getGradientXn(Mat& img);

Mat getGradientYn(Mat& img);

int getLabel(int i, int j, int height, int width);

Mat getA(int height, int width);

Mat getLaplacian();

Mat getB1(Mat& img1, Mat& img2, int posX, int posY, Rect ROI);

Mat getB2(Mat& img1, Mat& img2, int posX, int posY, Rect ROI);

Mat getResult(Mat& A, Mat& B, Rect& ROI);

Mat poisson_blending(Mat& img1, Mat& img2, Rect ROI, int posX, int posY);


#endif
