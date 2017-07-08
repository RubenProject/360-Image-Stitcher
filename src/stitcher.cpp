#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <climits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


#define OVERLAP_FACTOR 0.545
#define FOV_FACTOR 1.08
#define THREAD_COUNT 12
#define PI 3.1415926535897



static void help(char* progName)
{
    cout << endl
        <<  "This program takes input a circular image as input and outputs a square image" << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_name -- default ../data/lena.jpg] "        << endl << endl;
}


//transform the fisheye image to a rectangular image
void fishToSquare(const Mat img, Mat& res, int start, int end)
{
    double x, y;
	float theta,phi,r;
	double sx, sy, sz;

	float width = img.cols;
	float height = img.rows;
	float FOV = PI * FOV_FACTOR; // FOV of the fisheye, eg: 180 degrees

    CV_Assert(img.depth() == CV_8U);  // accept only uchar images

    res.create(img.rows, end - start, img.type());

    //copy image to square, transform coords  
    for(int i = start; i < end; i++)
    {
        for(int j = 0; j < res.rows; j++)
        {
	        // Polar angles, correction made for 1:1 image
	        theta = PI * (i / width - 0.5); // -pi/2 to pi/2
	        phi = PI * (j / height - 0.5);	// -pi/2 to pi/2

	        // Vector in 3D space
	        sx = cos(phi) * sin(theta);
	        sy = cos(phi) * cos(theta);
	        sz = sin(phi);
	
	        // Calculate fisheye angle and radius
	        theta = atan2(sz, sx);
	        phi = atan2(sqrt(sx * sx + sz * sz), sy);
	        r = width  * phi / FOV; 

	        // Pixel in fisheye space
	        x = 0.5 * width + r * cos(theta);
	        y = 0.5 * width + r * sin(theta);

            // Set pixel
            if (x >= 0 && x < img.cols 
             && y >= 0 && y < img.rows){           
                res.at<Vec3b>(j, i - start) = img.at<Vec3b>((int)y, (int)x);
            }else {
                

                res.at<Vec3b>(j, i - start) = Vec3b(0,0,0);
            }
        }
    }
}


void fishToSquare_threaded(const Mat img, Mat& res){
    Mat p_res[THREAD_COUNT];
    thread t[THREAD_COUNT];
    Mat temp;
    
    int width = img.cols;

    int start, end;

    //divide work over threads
    for (int i = 0; i < THREAD_COUNT; i++){
        start = i * width * 2 / THREAD_COUNT;
        end = (i + 1) * width * 2 / THREAD_COUNT;
        t[i] = thread(fishToSquare, img, ref(p_res[i]), start, end);
    }

    //join all partial solutions
    for (int i = 0; i < THREAD_COUNT; i++){
        t[i].join();
    }
    hconcat(p_res, THREAD_COUNT, res);
}


//corrects the shift present after projection
void correctShift(Mat& img){
    Mat res;
    Mat img_a = img(Rect(img.cols/4*3, 0, img.cols/4, img.rows));
    Mat img_b = img(Rect(0, 0, img.cols/4*3, img.rows));
    hconcat(img_a, img_b, res);
    img = res;
}


void extractDescriptors(const Mat imgA, const Mat imgB)
{
    vector<KeyPoint> keypointsA, keypointsB;
    Mat imgA_gray, imgB_gray, imgMatch;
    Mat descriptorsA, descriptorsB;
    vector<DMatch> matches, good_matches;

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
   

    double max_dist = 0; double min_dist = 10000;

    for (int i = 0; i < descriptorsA.rows; i++){
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    cout << "-- Max dist : " << max_dist << endl;
    cout << "-- Min dist : " << min_dist  << endl;

    for (int i = 0; i < descriptorsA.rows; i++){
        if (matches[i].distance < 3 * min_dist)
            good_matches.push_back(matches[i]);
    }

    //Display
    drawMatches(imgA, keypointsA, imgB, keypointsB, good_matches, imgMatch, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    namedWindow("matches", WINDOW_NORMAL);
    resizeWindow("matches", 600, 600);
    imshow("matches", imgMatch);

    vector<Point2f> A;
    vector<Point2f> B;

    for (int i = 0; i < (int)good_matches.size(); i++){
        A.push_back(keypointsA[good_matches[i].queryIdx].pt);
        B.push_back(keypointsB[good_matches[i].trainIdx].pt);
    }

    //homography
    Mat h = findHomography(A, B, RANSAC);
    cout << h << endl;

    Mat out;
    warpPerspective(imgA, out, h, imgA.size());

    namedWindow("result", WINDOW_NORMAL);
    resizeWindow("result", 600, 600);
    imshow("result", out);
    waitKey(0);
}


void bundleAdjustment(Mat& src){
    Mat res;
    float adjfact = 0.96;
    float update = (1 - adjfact) / src.rows;
    float x, y;
    res.create(src.rows, src.cols, src.type());
    for (int j = 0; j < res.rows; ++j){
        for (int i = 0; i < res.cols; ++i){
            x = i - (src.cols / 2);
            x *= adjfact;
            x += src.cols / 2;
            y = j;
            res.at<Vec3b>(j, i) = src.at<Vec3b>(y, x);
        }
        adjfact += update;
    }
    src = res;
}


void joinImgs(Mat in0, Mat in1, Mat& out){
    int width = in0.cols * OVERLAP_FACTOR;
    int x = (in0.cols - width) / 2;
    in0 = in0(Rect(x, 0, width, in0.rows));
    in1 = in1(Rect(x, 0, width, in1.rows));
    
    hconcat(in0, in1, out);
}


int main( int argc, char* argv[])
{
    help(argv[0]);
    const char* filename = argc >=2 ? argv[1] : "../img/original/360_0026.JPG";

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
    fishToSquare_threaded(cut0, dst0);
    fishToSquare_threaded(cut1, dst1);

    correctShift(dst0);
    correctShift(dst1);

    //bundleAdjustment(dst0);
    //bundleAdjustment(dst1);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Image transformed in " << t <<" seconds" << endl;
    
    Mat imgA_left, imgA_right, imgB_left, imgB_right;

    imgA_left = dst0(Rect(0, 0, dst0.cols/2, dst0.rows));
    imgA_right = dst0(Rect(dst0.cols/4*3, 0, dst0.cols/4, dst0.rows));
    imgB_left = dst1(Rect(0, 0, dst1.cols/4, dst1.rows));
    imgB_right = dst1(Rect(dst1.cols/2, 0, dst1.cols/2, dst1.rows));

    extractDescriptors(imgA_right, imgB_left);
    //extractDescriptors(imgB_right, imgA_left);

    joinImgs(dst0, dst1, out);


    //hconcat(dst0, dst1, out);
    namedWindow("out", WINDOW_NORMAL);
    resizeWindow("out", 1200, 600);
    imshow("out", out);

    string outputFileName(filename);
    outputFileName = outputFileName.substr(outputFileName.length() - 12);
    outputFileName = "../img/test/" + outputFileName;
    cout << "Writing to: " << outputFileName << endl;
    imwrite(outputFileName, out);

    waitKey();


    return 0;
}
