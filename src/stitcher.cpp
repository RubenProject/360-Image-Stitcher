#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


#define epsilon 0.0000001
#define PI 3.1415926535897



static void help(char* progName)
{
    cout << endl
        <<  "This program takes input a circular image as input and outputs a square image" << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_name -- default ../data/lena.jpg] [G -- grayscale] "        << endl << endl;
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

void extractDescriptors(const Mat imgA, const Mat imgB)
{
    Mat imgA_gray, imgB_gray, imgMatch;
    vector<KeyPoint> keypointsA, keypointsB;
    Mat descriptorsA, descriptorsB;
    vector<DMatch> matches;

    //Detector
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);

    //Descriptor
    Ptr<FREAK> extractor = FREAK::create();

    //Matcher
    Ptr<DescriptorMatcher> matcher = new BFMatcher(NORM_HAMMING, false);
    
    //Convert images to grayscale
    cvtColor(imgA, imgA_gray, CV_BGR2GRAY);
    cvtColor(imgB, imgB_gray, CV_BGR2GRAY);

    //Detect
    double t = (double)getTickCount();
    detector->detect(imgA_gray, keypointsA);
    detector->detect(imgB_gray, keypointsB);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Feature Points detected in " << t <<" seconds" << endl;

    //Extract
    t = (double)getTickCount();
    extractor->compute(imgA_gray, keypointsA, descriptorsA);
    extractor->compute(imgB_gray, keypointsB, descriptorsB);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Feature Points extracted in " << t <<" seconds" << endl;

    //Match
    t = (double)getTickCount();
    matcher->match(descriptorsA, descriptorsB, matches, imgMatch);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Feature Points matched in " << t <<" seconds" << endl;
   
    //Display
    drawMatches(imgA, keypointsA, imgB, keypointsB, matches, imgMatch);
    namedWindow("matches", WINDOW_NORMAL);
    resizeWindow("matches", 600, 600);
    imshow("matches", imgMatch);
}




int main( int argc, char* argv[])
{
    help(argv[0]);
    const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";

    Mat src, dst0, dst1, out;

    if (argc >= 3 && !strcmp("G", argv[2])){
        cout << "gray" << endl;
        src = imread( filename, IMREAD_GRAYSCALE);
    } else {
        cout << "color" << endl;
        src = imread( filename, IMREAD_COLOR);
    }

    if (src.empty())
    {
        cerr << "Can't open image ["  << filename << "]" << endl;
        return -1;
    }

    Mat cut0 = src(Rect(0, 0, src.rows, src.cols/2));
    Mat cut1 = src(Rect(src.cols/2, 0, src.rows, src.cols/2));

    double t = (double)getTickCount();
    fishToSquare(cut0, dst0);
    fishToSquare(cut1, dst1);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Image transformed in " << t <<" seconds" << endl;
    
    extractDescriptors(dst0, dst1);

    hconcat(dst0, dst1, out);
    namedWindow("out", WINDOW_NORMAL);
    resizeWindow("out", 1200, 600);
    imshow("out", out);

    //string outputFileName(filename);
    //outputFileName.append("_NO_STITCH_S.JPG");
    //imwrite(outputFileName, out);

    waitKey();


    return 0;
}
