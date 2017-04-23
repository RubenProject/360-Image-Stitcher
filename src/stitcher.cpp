#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <stdio.h>
#include <iostream>


using namespace std;
using namespace cv;


#define epsilon 0.0000001


void ellipticalSquareToDisc(double x, double y, double& u, double& v);
void ImgToSquare(const Mat& myImage,Mat& Result);


inline float sgn(float input)
{
    float output = 1.0f;
    if (input < 0.0) {
        output = -1.0f;
    }
    return output;
}


static void help(char* progName)
{
    cout << endl
        <<  "This program takes input a circular image as input and outputs a square image" << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_name -- default ../data/lena.jpg] [G -- grayscale] "        << endl << endl;
}



void stretchSquareToDisc(float x, float y, float& u, float& v)
{
    if ( (fabs(x) < epsilon) || (fabs(y) < epsilon))  {
        u = x;
        v = y;
        return;
    }
    
    float x2 = x*x;
    float y2 = y*y;
    float hypothenusSquared = x2 + y2;

    float reciprocalHypothenus =  1.0f/sqrt(hypothenusSquared);
    
    float multiplier = 1.0f;

    if (x2 > y2) {
        multiplier = sgn(x) * x * reciprocalHypothenus;
    } else {
        multiplier = sgn(y) * y * reciprocalHypothenus;
    }

    u = x * multiplier;
    v = y * multiplier;
}




void ellipticalSquareToDisc(double x, double y, double& u, double& v)
{
    u = x * sqrt(1.0 - y*y/2.0);
    v = y * sqrt(1.0 - x*x/2.0);    
}


void ImgToSquare(const Mat& Img ,Mat& Res, int col)
{
    double x, y, u, v;
    double r = Img.rows / 2;
    cout << col << endl;
    CV_Assert(Img.depth() == CV_8U);  // accept only uchar images

    Res.create(Img.rows, Img.cols / 2, Img.type());


    //copy image to square, transform coords  
    for(int j = 0; j < Img.cols / 2; ++j)
    {
        for(int i= 0;i < Img.rows; ++i)
        {
            u = (i - r) / r;
            v = (j - r) / r;
            ellipticalSquareToDisc(u, v, x, y);
            x = x * r + r + col;
            y = y * r + r + col;
            if (x > 0 && x < Img.cols 
             && y > 0 && y < Img.rows){           
                Res.at<Vec3b>(j, i) = Img.at<Vec3b>((int)y, (int)x);
            }
        }
    }
}

int main( int argc, char* argv[])
{
    help(argv[0]);
    const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";

    Mat src, dst0, dst1;

    if (argc >= 3 && !strcmp("G", argv[2]))
        src = imread( filename, IMREAD_GRAYSCALE);
    else
        src = imread( filename, IMREAD_COLOR);

    if (src.empty())
    {
        cerr << "Can't open image ["  << filename << "]" << endl;
        return -1;
    }

    namedWindow("i0", WINDOW_NORMAL);
    namedWindow("o0", WINDOW_NORMAL);
    namedWindow("o1", WINDOW_NORMAL);
    resizeWindow("i0", 600, 300);
    resizeWindow("o0", 600, 600);
    resizeWindow("o1", 600, 600);
    imshow("i0", src);

    double t = (double)getTickCount();

    ImgToSquare(src, dst0, 0);
    ImgToSquare(src, dst1, src.cols/2);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Hand written function times passed in seconds: " << t << endl;

    imshow( "o0", dst0 );
    imshow( "o1", dst1 );
    waitKey();

  //![kern]
    Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                   -1,  5, -1,
                                    0, -1,  0);
  //![kern]

    return 0;
}
