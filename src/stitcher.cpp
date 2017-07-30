#ifndef stitcherCCVar  
#define stitcherCCVar

#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <climits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "orientation.hpp"
#include "astar.hpp"
#include "projection.hpp"
#include "imgops.hpp"
#include "manual.hpp"
#include "poisson.hpp"

#include "stitcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

bool SIG_STOP;
bool ASTAR_PROGRESS[FIXED_POINTS];


Mat extractDescriptors(const Mat imgA, const Mat imgB){
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
    drawMatches(imgA, keypointsA, imgB, keypointsB, good_matches, imgMatch,
               Scalar::all(-1), Scalar::all(-1),
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
    cout << "homography matrix: " << endl << h << endl;

    Mat out;
    warpPerspective(imgA, out, h, imgA.size());

    namedWindow("result", WINDOW_NORMAL);
    resizeWindow("result", 600, 600);
    imshow("result", out);

    return h;
}


void adjustImg(Mat A, Mat B, Orientation o_A, Orientation o_B, Mat& out){
    if(o_A.b != 1.00)
        bundleAdjustment(A, o_A.b);
    if(o_B.b != 1.00)
        bundleAdjustment(B, o_B.b);
    if(o_A.r != 0.00)
        A = rotateImg(A, o_A.r);
    if(o_B.r != 0.00)
        B = rotateImg(B, o_B.r);
    if(o_A.y != 0)
        A = translateImg(A, o_A.y);
    if(o_B.y != 0)
        B = translateImg(B, o_B.y);
    joinImgs(A, B, out, o_A.x);
    A = out(Rect(0, 0, out.cols/2, out.rows));
    B = out(Rect(out.cols/2, 0, out.cols/2, out.rows));
    joinImgs(B, A, out, o_B.x);
    correctShift(out);
} 


void displayGrayMap(Mat img){
    float min = FLT_MAX, max = -1, temp;
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            temp = img.at<float>(i, j);
            if (temp < min)
                min = temp;
            if (temp > max)
                max = temp;
        }
    }
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            temp = img.at<float>(i, j);
            temp -= min; 
            temp /= max - min;
            temp *= 255;
            img.at<float>(i, j) = temp;
        }
    }
    namedWindow("gray", WINDOW_NORMAL);
    resizeWindow("gray", 600, 600);
    imshow("gray", img);
    waitKey(0);
}


float euclidDist(const Vec3b a, const Vec3b b){
    float bsquared = pow(b[0] - a[0], 2);
    float gsquared = pow(b[1] - a[1], 2);
    float rsquared = pow(b[2] - a[2], 2);
    return bsquared + gsquared + rsquared;
}


void outputCoords(int c, int width){
    cerr << "(" << c % width << ", " << c / width << ")";
}


void computePartialShortestPath(const float* weights, const int height, const int width,
                                const int start, const int goal,
                                vector<Point>& shortest_path, int threadnr){
    int* paths = new int[height * width];
    if (astar(weights, height, width, start, goal, paths)){
        int cur = goal;
        while(paths[cur] != start){
            cur = paths[cur];
            shortest_path.push_back(Point(cur % width, cur / width));
        }
    }
    if (!SIG_STOP){
        ASTAR_PROGRESS[threadnr] = true;
        cout << "thread " << threadnr << " finished!" << endl;
    }
}


void checkPathIntegrity(vector<Point>& path, int height, int width){
    vector<bool> rowFilled(height, false);
    for (int i = 0; i < (int)path.size(); i++)
        rowFilled[path[i].y] = true;

    for (int i = 0; i < (int)rowFilled.size(); i++){
        if (rowFilled[i] == false)
            path.push_back(Point(width/2, i));
    }
}


void fillMask(vector<vector<bool>>& mask, Point p){
    int i = p.x;
    while (i < (int)mask[p.y].size()){
        mask[p.y][i] = true;
        i++;
    }
}

bool applyMask(Mat A, Mat B, vector<vector<bool>> mask, Mat& out){
    if (A.rows != B.rows || A.cols != B.cols)
        return false;
    out.create(A.rows, A.cols, A.type());
    for (int i = 0; i < out.cols; i++){
        for (int j = 0; j < out.rows; j++){
            if (!mask[j][i])
                out.at<Vec3b>(j, i) = A.at<Vec3b>(j, i);
            else
                out.at<Vec3b>(j, i) = B.at<Vec3b>(j, i);
        }
    }
    return true;
}


void calcRGBweights(float* weights, Mat A, Mat B){
    const int height = A.rows;
    const int width = A.cols;
    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            weights[j * width + i] = euclidDist(A.at<Vec3b>(j, i), B.at<Vec3b>(j, i)); 
        }
    }
}


void calcHSVweights(float* weights, Mat A, Mat B){
    const int height = A.rows;
    const int width = A.cols;
    Mat A_hsv, B_hsv;
    cvtColor(A, A_hsv, COLOR_BGR2HSV);
    cvtColor(B, B_hsv, COLOR_BGR2HSV);
    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            weights[j * width + i] = euclidDist(A_hsv.at<Vec3b>(j, i), B_hsv.at<Vec3b>(j, i)); 
        }
    }
}


//returns true if threads finished correctly
bool monitorThreads(){
    double t_start = (double)getTickCount();
    double t_cur = 0;
    while (!SIG_STOP){
        t_cur = ((double)getTickCount() - t_start)/getTickFrequency();
        bool threads_finished = true;
        for (int i = 0; i < FIXED_POINTS; i++){
            if (!ASTAR_PROGRESS[i])
                threads_finished = false;
        }
        SIG_STOP = threads_finished;
        if (t_cur > TIME_OUT){
            SIG_STOP = true;
            return false; 
        }
    }
    return true;
}


void stitch(Mat A, Mat B, Mat& out, int x){
    //calculate and isolate overlap
    Mat A_extra, B_extra, temp, D, static_stitch;
    A_extra = A(Rect(0, 0, A.cols - x/4*3, A.rows));
    A = A(Rect(A.cols -x/4*3, 0, x/2, A.rows));
    B_extra = B(Rect(x/4*3, 0, B.cols - x/4*3, B.rows));
    B = B(Rect(x/4, 0, x/2, B.rows));
    
    hconcat(A(Rect(0 , 0, A.cols/2, A.rows)), B(Rect(B.cols/2, 0, B.cols/2, B.rows)), static_stitch);

    namedWindow("static_stitch", WINDOW_NORMAL);
    resizeWindow("static_stitch", 600, 600);
    imshow("static_stitch", static_stitch);
    
    const int height = A.rows;
    const int width = A.cols;
    float* weights = new float[height * width];

    cout << "RGB" << endl;
    calcRGBweights(weights, A, B);
    Mat C(height, width, CV_32FC1, weights);
    displayGrayMap(C);

    cout << "HSV" << endl;
    calcHSVweights(weights, A, B);
    Mat C_2(height, width, CV_32FC1, weights);
    displayGrayMap(C_2);

    //find shortest path from top to bottom through mat C
    vector<Point> shortest_path;
    vector<vector<Point>> partial_shortest_path(FIXED_POINTS, vector<Point>());
    int update = width * height / FIXED_POINTS;
    int start = width / 2;
    int goal = start + update;
    thread t[FIXED_POINTS];
    //start worker threads with the specified amount of fixed points along the symmetry axis
    cout << "calculating shortest path..." << endl;
    SIG_STOP = false;
    for (int i = 0; i < FIXED_POINTS; i++){
        ASTAR_PROGRESS[i] = false;
        cout << "starting thread " << i << ": ";
        outputCoords(start, width);
        cout << "->"; 
        outputCoords(goal, width) ;
        cout << endl;
        t[i] = thread(computePartialShortestPath, weights, height, width,
                        start, goal, ref(partial_shortest_path[i]), i);
        start += update;
        goal += update;
        if (i == FIXED_POINTS - 2)
            goal -= width;
    }

    if (!monitorThreads())
        cout << "threads timed out after " << TIME_OUT << " seconds." << endl;
    else
        cout << "threads finished correctly" << endl;

    //kill threads and combine partial solutions
    for (int i = 0; i < FIXED_POINTS; i++){
        if (ASTAR_PROGRESS[i]){
            t[i].join();
            for (int j = 0; j < (int)partial_shortest_path[i].size(); j++)
                shortest_path.push_back(partial_shortest_path[i][j]);
        } else {
            cout << "thread " << i << " failed to complete in time." << endl;
            t[i].join();
        }
    }

    //display the found path with E X T R A T H I C C line
    Mat color;
    cvtColor(C, color, COLOR_GRAY2BGR);
    for (int i = 0; i < (int)shortest_path.size(); i++){
        color.at<Vec3f>(shortest_path[i].y, shortest_path[i].x-1) = Vec3f(0, 0, 255);
        color.at<Vec3f>(shortest_path[i].y, shortest_path[i].x) = Vec3f(0, 0, 255);
        color.at<Vec3f>(shortest_path[i].y, shortest_path[i].x+1) = Vec3f(0, 0, 255);
    }

    //fill in any empty rows by setting a flag in the middle of the screen
    if ((int)shortest_path.size() != height)
        checkPathIntegrity(shortest_path, height, width);

    //make a mask for the transition
    vector<vector<bool>> mask(height, vector<bool>(width, false));
    cout << "creating mask for transition..." << endl;
    for (int i = 0; i < (int)shortest_path.size(); i++){
        fillMask(mask, shortest_path[i]);
    }

    //apply mask to stitching edge
    cout << "applying mask to transition..." << endl;
    Mat img[3];
    applyMask(A, B, mask, img[1]);
    img[0] = A_extra;
    img[2] = B_extra;

    hconcat(img, 3, out);
    cout << "done!" << endl;

    namedWindow("edge", WINDOW_NORMAL);
    resizeWindow("edge", 600, 600);
    imshow("edge", color);
    namedWindow("out", WINDOW_NORMAL);
    resizeWindow("out", 600, 600);
    imshow("out", out);
    waitKey(0);
}


void blurTransition(Mat A, Mat B, Mat& out, int x){
    const int DIVISIONS = 5;
    Mat C, D[DIVISIONS + 2], E[DIVISIONS + 2];
    A = A(Rect(0, 0, A.cols - x/2, A.rows));
    B = B(Rect(x/2, 0, B.cols - x/2 ,B.rows));
    hconcat(A, B, C);
    int width = C.cols;
    int blurwidth = 400;


    D[0] = C(Rect(0, 0, (width - blurwidth) / 2, C.rows));
    for (int i = 0; i < DIVISIONS; i++)
        D[i+1] = C(Rect((width - blurwidth) / 2 + blurwidth / DIVISIONS * i, 0, blurwidth / DIVISIONS, C.rows));
    D[DIVISIONS + 1] = C(Rect(width - (width - blurwidth) / 2, 0, (width - blurwidth) / 2, C.rows));

    E[0] = D[0];

    int k = 3;//kernel size
    for (int i = 0; i < DIVISIONS; i++){
        GaussianBlur(D[i + 1], E[i + 1], Size(k, k), 0, 0);
        if (i < DIVISIONS)
            k += 8;
        else 
            k -= 8;
    }
    E[DIVISIONS + 1] = D[DIVISIONS + 1];
    
    hconcat(E, DIVISIONS + 2, out);
}


void joinAndStitch(Mat A, Mat B, Orientation o_A, Orientation o_B, Mat& out){
    if(o_A.b != 1.00)
        bundleAdjustment(A, o_A.b);
    if(o_B.b != 1.00)
        bundleAdjustment(B, o_B.b);
    if(o_A.r != 0.00)
        A = rotateImg(A, o_A.r);
    if(o_B.r != 0.00)
        B = rotateImg(B, o_B.r);
    if(o_A.y != 0)
        A = translateImg(A, o_A.y);
    if(o_B.y != 0)
        B = translateImg(B, o_B.y);
    stitch(A, B, out, o_A.x);
    //blurTransition(A, B, out, o_A.x);
    A = out(Rect(0, 0, out.cols/2, out.rows));
    B = out(Rect(out.cols/2, 0, out.cols/2, out.rows));
    stitch(B, A, out, o_B.x);
    //blurTransition(B, A, out, o_B.x);
    correctShift(out);
} 


void blend(Mat A, Mat B, Mat& out, int x){

    Mat in1, in2, img1, img2;
    Mat blended;

    int overlap = 30;
    in1 = A(Rect(A.cols - x/2 - overlap / 2, 0, overlap, A.rows));
    A = A(Rect(0, 0, A.cols - x/2, A.rows));
    B = B(Rect(x/2, 0, B.cols - x/2 ,B.rows));
    hconcat(A, B, in2);

    Mat img_all[3];
    img_all[0] = in2(Rect(0, 0, in2.cols/4*3, in2.rows));
    img_all[2] = in2(Rect(in2.cols - in2.cols/4*3, 0, in2.cols/4*3, in2.rows));
    
    in2 = in2(Rect(in2.cols/8*3, 0, in2.cols/4, in2.rows));

    namedWindow("before", WINDOW_NORMAL);
    resizeWindow("before", 600, 600);
    imshow("before", in2);
    
    in1.convertTo(img1, CV_64FC3);
    in2.convertTo(img2, CV_64FC3);

    int x1 = in2.cols/2 - overlap / 2;
    int height = in2.rows;

    for (int i = 1; i < height / overlap; i++){
        Rect rc(0, i * overlap, overlap, overlap);
        blended = poisson_blending(img1, img2, rc, x1, i * overlap);
        blended.convertTo(blended, CV_8UC1);
        //the area you want to copy to
        //this trick works because creating a smaller mat from a bigger one 
        //does not copy the data it simply points to it...
        Rect rc2(x1, i * overlap, overlap, overlap);
        Mat roimat = in2(rc2);
        blended.copyTo(roimat);
        cout << "blending progress: " 
            << (float)i / ((height / overlap) - 1) * 100 
            << "%" << endl;
    }

    img_all[1] = in2;

    hconcat(img_all, 3, out);

    namedWindow("after", WINDOW_NORMAL);
    resizeWindow("after", 600, 600);
    imshow("after", img_all[1]);
    waitKey();
}


//will it blend?
void joinAndBlend(Mat A, Mat B, Orientation o_A, Orientation o_B, Mat& out){
    if(o_A.b != 1.00)
        bundleAdjustment(A, o_A.b);
    if(o_B.b != 1.00)
        bundleAdjustment(B, o_B.b);
    if(o_A.r != 0.00)
        A = rotateImg(A, o_A.r);
    if(o_B.r != 0.00)
        B = rotateImg(B, o_B.r);
    if(o_A.y != 0)
        A = translateImg(A, o_A.y);
    if(o_B.y != 0)
        B = translateImg(B, o_B.y);
    blend(A, B, out, o_A.x);
    A = out(Rect(0, 0, out.cols/2, out.rows));
    B = out(Rect(out.cols/2, 0, out.cols/2, out.rows));
    blend(B, A, out, o_B.x);
    correctShift(out);
}

#endif
