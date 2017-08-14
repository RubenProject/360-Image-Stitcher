#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;

int main(int argc, char** argv)
{
   // Load the source image
   Mat src = imread( "cat.png", 1);

   // Create a destination Mat object
   Mat dst;

   int i = 21;
   int s = 75;
   bilateralFilter(src, dst, i, s, s);

   imwrite("bilateral21-75.png", dst );
}

