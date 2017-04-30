#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <stdio.h>
#include <iostream>


using namespace std;
using namespace cv;


#define epsilon 0.0000001
#define PI 3.1415926535897



static void help(char* progName)
{
    cout << endl
        <<  "This program takes input a circular image as input and outputs a square image" << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_name -- default ../data/lena.jpg] [G -- grayscale] "        << endl << endl;
}





void ellipticalSquareToDisc(double x, double y, double& u, double& v)
{
    u = x * sqrt(1.0 - y*y/2.0);
    v = y * sqrt(1.0 - x*x/2.0);    
}


void fishToSquare(const Mat& Img ,Mat& Res)
{
    double x, y;
	float theta,phi,r;
	double sx, sy, sz;

	float FOV = PI * 1.00; // FOV of the fisheye, eg: 180 degrees
	float width = Img.cols;
	float height = Img.rows;


    CV_Assert(Img.depth() == CV_8U);  // accept only uchar images

    Res.create(Img.rows, Img.cols, Img.type());


    //copy image to square, transform coords  
    for(int j = 0; j < width; ++j)
    {
        for(int i= 0;i < height; ++i)
        {
	        // Polar angles, correction made for 1:1 image
	        theta = PI * (i / width - 0.5); // -pi to pi
	        phi = PI * (j / height - 0.5);	// -pi/2 to pi/2

	        // Vector in 3D space
	        sx = cos(phi) * sin(theta);
	        sy = cos(phi) * cos(theta);
	        sz = sin(phi);
	
	        // Calculate fisheye angle and radius
	        theta = atan2(sz, sx);
	        phi = atan2(sqrt(sx * sx + sz * sz), sy);
	        r = width * phi / FOV; 

	        // Pixel in fisheye space
	        x = 0.5 * width + r * cos(theta);
	        y = 0.5 * width + r * sin(theta);

            // Set pixel
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

    Mat src, dst0, dst1, out;

    if (argc >= 3 && !strcmp("G", argv[2]))
        src = imread( filename, IMREAD_GRAYSCALE);
    else
        src = imread( filename, IMREAD_COLOR);

    if (src.empty())
    {
        cerr << "Can't open image ["  << filename << "]" << endl;
        return -1;
    }

    //namedWindow("i0", WINDOW_NORMAL);
    //namedWindow("i1", WINDOW_NORMAL);
    namedWindow("o0", WINDOW_NORMAL);
    namedWindow("o1", WINDOW_NORMAL);
    //resizeWindow("i0", 600, 600);
    //resizeWindow("i1", 600, 600);
    resizeWindow("o0", 600, 600);
    resizeWindow("o1", 600, 600);
    
    Mat cut0 = src(Rect(0, 0, src.rows, src.cols/2));
    Mat cut1 = src(Rect(src.cols/2, 0, src.rows, src.cols/2));

    //imshow("i0", cut0);
    //imshow("i1", cut1);

    double t = (double)getTickCount();

    fishToSquare(cut0, dst0);
    fishToSquare(cut1, dst1);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Hand written function times passed in seconds: " << t << endl;

    imshow( "o0", dst0 );
    imshow( "o1", dst1 );
    hconcat(dst0, dst1, out);
    string outputFileName(filename);
    outputFileName.append("_NO_STITCH_S.JPG");
    imwrite(outputFileName, out);
    waitKey();


    return 0;
}
