#include <vector>
#include <stack>
#include <string>
#include <iostream>
#include <thread>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <climits>

#include <linux/input.h>
#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <signal.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "AStar.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


#define TIME_OUT 60
#define FOV_FACTOR 1.08
#define THREAD_COUNT 8 
#define FIXED_POINTS 4
#define PI 3.1415926535897

#define INPUT_QUEUE "/dev/input/event2"
#define EVENT_LEN 16

struct Orientation {
    int x, y;
    double r, b, s;
};

bool SIG_STOP;
bool MANUAL_POSITIONING_DONE;
bool ASTAR_PROGRESS[FIXED_POINTS];
Mat preview;


static void help(char* progName)
{
    cout << endl
        <<  "This program stitches raw output of the Samsung Gear 360 together into a complete panorama" << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_name ../img/example.jpg] " << endl << endl;
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


Mat extractDescriptors(const Mat imgA, const Mat imgB)
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


void bundleAdjustment(Mat& src, double factor){
    Mat res;
    double update = (1 - factor) / src.rows;
    double x, y;
    res.create(src.rows, src.cols, src.type());
    for (int j = 0; j < res.rows; ++j){
        for (int i = 0; i < res.cols; ++i){
            x = i - (src.cols / 2);
            x *= factor;
            x += src.cols / 2;
            y = j;
            res.at<Vec3b>(j, i) = src.at<Vec3b>(y, x);
        }
        factor += update;
    }
    src = res;
}


void joinImgs(Mat A, Mat B, Mat& out, int x){
    Mat C;
    A = A(Rect(0, 0, A.cols - x/2, A.rows));
    B = B(Rect(x/2, 0, B.cols - x/2 ,B.rows));
    hconcat(A, B, out);
}


Mat translateImg(Mat src, int y){
    Mat res;
    if (y < 0){
        src = src(Rect(0, -y, src.cols, src.rows + y));
        Mat p(-y, src.cols, src.type());
        vconcat(src, p, res);
        return res;
    } else if (y > 0) {
        src = src(Rect(0, y, src.cols, src.rows - y));
        Mat p(y, src.cols, src.type());
        vconcat(p, src, res);
        return res;
    } else {
        res = src(Rect(0, 0, src.cols, src.rows));
        return res;
    }
}


Mat rotateImg(Mat src, double angle){
    Mat res;
    Point center = Point(src.cols/2, src.rows/2);
    Mat r = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(src, res, r, res.size());
    return res;
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


void readInputEvent(int fd, int& key){
    int rd, value, size = sizeof(struct input_event);
    struct input_event ev[64];
    while(true){
        if ((rd = read(fd, ev, size * 64)) < size)
            exit(0);

        value = ev[0].value;
        
        if (value != ' ' && ev[1].value == 1 && ev[1].type == 1){
            key = ev[1].code;
            cout << ev[1].code << endl;;
            break;
        }
    }
}


void parseInputPhysical(int key, Orientation& o_A, Orientation& o_B){
    if(key == 17) o_A.y += 1; //w
    else if(key == 30) o_A.x -= 1; //a
    else if(key == 31) o_A.y -= 1; //s
    else if(key == 32) o_A.x += 1; //d
    else if(key == 23) o_B.y += 1; //i
    else if(key == 36) o_B.x -= 1; //j
    else if(key == 37) o_B.y -= 1; //k
    else if(key == 38) o_B.x += 1; //l
    else if(key == 16) o_A.r -= 0.05; //q
    else if(key == 18) o_A.r += 0.05; //e
    else if(key == 22) o_B.r -= 0.05; //u
    else if(key == 24) o_B.r += 0.05; //o
    else if(key == 44) o_A.s -= 0.01; //z
    else if(key == 45) o_A.s += 0.01; //x
    else if(key == 49) o_B.s -= 0.01; //n
    else if(key == 50) o_B.s += 0.01; //m
    else if(key == 46) o_A.b -= 0.01; //c
    else if(key == 47) o_A.b += 0.01; //v
    else if(key == 51) o_B.b -= 0.01; //,
    else if(key == 52) o_B.b += 0.01; //.
    else if(key == 28) MANUAL_POSITIONING_DONE = true; //enter
    else cerr << "incorrect input!" << endl;
}


void parseInputASCII(int key, Orientation& o_A, Orientation& o_B){
    if(key == (int)'w') o_A.y += 1; //w
    else if(key == (int)'a') o_A.x -= 1; //a
    else if(key == (int)'s') o_A.y -= 1; //s
    else if(key == (int)'d') o_A.x += 1; //d
    else if(key == (int)'i') o_B.y += 1; //i
    else if(key == (int)'j') o_B.x -= 1; //j
    else if(key == (int)'k') o_B.y -= 1; //k
    else if(key == (int)'l') o_B.x += 1; //l
    else if(key == (int)'q') o_A.r -= 0.05; //q
    else if(key == (int)'e') o_A.r += 0.05; //e
    else if(key == (int)'u') o_B.r -= 0.05; //u
    else if(key == (int)'o') o_B.r += 0.05; //o
    else if(key == (int)'z') o_A.s -= 0.01; //z
    else if(key == (int)'x') o_A.s += 0.01; //x
    else if(key == (int)'n') o_B.s -= 0.01; //n
    else if(key == (int)'m') o_B.s += 0.01; //m
    else if(key == (int)'c') o_A.b -= 0.01; //c
    else if(key == (int)'v') o_A.b += 0.01; //v
    else if(key == (int)',') o_B.b -= 0.01; //,
    else if(key == (int)'.') o_B.b += 0.01; //.
    else if(key == 13) MANUAL_POSITIONING_DONE = true; //enter
    else cerr << "incorrect input!" << endl;
}


//display on different thread
void displayPreview(){
    namedWindow("Manual Position" , WINDOW_NORMAL);
    resizeWindow("Manual Position" , 1200, 600);
    while(!MANUAL_POSITIONING_DONE){
        imshow("Manual Position", preview);
        waitKey(30);
    }
    destroyWindow("Manual Position");
    return;
}


void interactImg(const Mat A, const Mat B, int scale, Orientation& o_A, Orientation& o_B){
    //Initial orientation
    o_A = {.x = 961, .y = -1,
            .r = 0.15,
            .b = 1.00,
            .s = 1.00};
    o_B = {.x = 924, .y = 0,
            .r = -0.9,
            .b = 1.00,
            .s = 1.00};

    o_A.x *= scale;
    o_A.y *= scale;
    o_B.x *= scale;
    o_B.y *= scale;
    return;

    ///Initial orientation
    o_A = {.x = 3600/scale, .y = 0,
            .r = 0.00,
            .b = 1.00,
            .s = 1.00};
    o_B = {.x = 3600/scale, .y = 0,
            .r = 0.00,
            .b = 1.00,
            .s = 1.00};

    int fd, key;
    char name[256];

    //downscaling for faster operations
    Mat A_s, B_s, out_s;
    resize(A, A_s, Size(A.cols/scale, A.rows/scale));
    resize(B, B_s, Size(B.cols/scale, B.rows/scale));

    if((fd = open(INPUT_QUEUE, O_RDONLY)) == -1)
        cerr << "cannot read from device" << endl;

    ioctl (fd, EVIOCGNAME (sizeof (name)), name);
    cerr << "reading from: " << name << endl;

    MANUAL_POSITIONING_DONE = false;
    adjustImg(A_s, B_s, o_A, o_B, out_s);
    preview = out_s;
    thread t = thread(displayPreview);

    while(!MANUAL_POSITIONING_DONE){
        readInputEvent(fd, key);
        parseInputPhysical(key, o_A, o_B);
        adjustImg(A_s, B_s, o_A, o_B, out_s);
        preview = out_s;
        cout << key << endl;
    }
    cout << "finalizing positioning" << endl;
    t.join();
    cout << "o_A:" << "x: " << o_A.x << " ,y: " << o_A.y << " ,r: " << o_A.r << " ,b: " << o_A.b << endl;
    cout << "o_B:" << "x: " << o_B.x << " ,y: " << o_B.y << " ,r: " << o_B.r << " ,b: " << o_B.b << endl;
    //adjust orientations for full resolution image
    o_A.x *= scale;
    o_A.y *= scale;
    o_B.x *= scale;
    o_B.y *= scale;
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
    cout << "succes" << endl;
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

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Image transformed in " << t <<" seconds" << endl;
    
    //Mat imgA_left, imgA_right, imgB_left, imgB_right;
    //int width_0 = dst0.cols/4;
    //int width_1 = width_0;

    //imgA_left = dst0(Rect(0, 0, width_0, dst0.rows));
    //imgA_right = dst0(Rect(dst0.cols - width_0, 0, width_0, dst0.rows));

    //imgB_left = dst1(Rect(0, 0, width_1, dst1.rows));
    //imgB_right = dst1(Rect(dst1.cols - width_1, 0, width_1, dst1.rows));

    //Mat h0 = extractDescriptors(imgA_right, imgB_left);
    //Mat h1 = //extractDescriptors(imgB_right, imgA_left);

    //Mat res0, res1;
    //res1 = dst1;
    //res0.create(dst0.rows, dst0.cols*2, dst0.type());
    //warpPerspective(dst0, res0, h0, res0.size());
    //res1.create(dst1.rows, dst1.cols*2, dst1.type());
    //warpPerspective(dst1, res1, h1, res1.size());
    //hconcat(dst1, dst0, out);

    Orientation o_A, o_B;
    interactImg(dst0, dst1, 4, o_A, o_B);
    joinAndStitch(dst0, dst1, o_A, o_B, out);
    //adjustImg(dst0, dst1, o_A, o_B, out);

    string outputFileName(filename);
    outputFileName = outputFileName.substr(outputFileName.length() - 12);
    outputFileName = "../img/result/" + outputFileName;
    cout << "Writing to: " << outputFileName << endl;
    imwrite(outputFileName, out);

    return 0;
}
