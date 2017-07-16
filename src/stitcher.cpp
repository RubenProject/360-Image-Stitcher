#include <vector>
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


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


#define OVERLAP_FACTOR 0.545
#define FOV_FACTOR 1.08
#define THREAD_COUNT 12
#define PI 3.1415926535897

#define INPUT_QUEUE "/dev/input/event4"
#define EVENT_LEN 16

struct Orientation {
    int x, y;
    double r, b, s;
};

bool MANUAL_POSITIONING_DONE;
Mat preview;


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


void joinImgs(Mat A, Mat B, Mat& out, int x, int y){
    out.create(A.rows, A.cols*2 - x, A.type());
    for (int i = 0; i < out.cols; i++){
        for (int j = 0; j < out.rows; j++){
            if ( i <  A.cols - x / 2){
                out.at<Vec3b>(j, i) = A.at<Vec3b>(j, i);
            } else if (j + y < B.rows && j + y >= 0){
                out.at<Vec3b>(j, i) = B.at<Vec3b>(j + y, i - B.cols + x);
            } else {
                out.at<Vec3b>(j, i) = Vec3b(0,0,0);
            }

        }
    }
}


Mat rotateImage(Mat src, double angle){
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
        A = rotateImage(A, o_A.r);
    if(o_B.r != 0.00)
        B = rotateImage(B, o_B.r);
    joinImgs(A, B, out, o_A.x, o_A.y);
    Mat A2 = out(Rect(0, 0, out.cols/2, out.rows));
    Mat B2 = out(Rect(out.cols/2, 0, out.cols/2, out.rows));
    joinImgs(B2, A2, out, o_B.x, o_B.y);
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
    o_A = {.x = 461, .y = 0,
            .r = -0.60,
            .b = 0.98,
            .s = 1.00};
    o_B = {.x = 461, .y = 1,
            .r = -0.1,
            .b = 0.98,
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


float euclidDist(Vec3b p_A, Vec3b p_B){
    float rsquared = pow((p_B[0] - p_A[0]), 2);
    float gsquared = pow((p_B[1] - p_A[1]), 2);
    float bsquared = pow((p_B[2] - p_A[2]), 2);
    return rsquared + gsquared + bsquared;
}



void displayGrayMap(Mat img){
    float min = 1000000, max = -1, temp;
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


void stitch(Mat A, Mat B, Mat& out, int x, int y){
    Mat C; //matrix for energy values
    //calculate and isolate overlap
    A = A(Rect(A.cols-x/4*3, 0, x/2, A.rows));
    B = B(Rect(x/4, 0, x/2, B.rows));
    namedWindow("A2", WINDOW_NORMAL);
    resizeWindow("A2", 600, 600);
    imshow("A2", A);
    namedWindow("B2", WINDOW_NORMAL);
    resizeWindow("B2", 600, 600);
    imshow("B2", B);
    C.create(A.rows, x/2, DataType<float>::type);
    for (int i = 0; i < C.rows; i++){
        for (int j = 0; j < C.cols; j++){
            C.at<float>(i, j) = euclidDist(A.at<Vec3b>(i, j), B.at<Vec3b>(i, j));
        }
    }
    displayGrayMap(C);

    //find shortest path from top to bottom through mat C
    //









    namedWindow("out", WINDOW_NORMAL);
    resizeWindow("out", 600, 600);
    imshow("out", out);
    waitKey(0);
}


void joinAndStitch(Mat A, Mat B, Orientation o_A, Orientation o_B, Mat& out){
    if(o_A.b != 1.00)
        bundleAdjustment(A, o_A.b);
    if(o_B.b != 1.00)
        bundleAdjustment(B, o_B.b);
    if(o_A.r != 0.00)
        A = rotateImage(A, o_A.r);
    if(o_B.r != 0.00)
        B = rotateImage(B, o_B.r);
    //joinImgs(B, A, out, o_A.x, o_A.y);
    stitch(B, A, out, o_A.x, o_A.y);
    Mat A2 = out(Rect(0, 0, out.cols/2, out.rows));
    Mat B2 = out(Rect(out.cols/2, 0, out.cols/2, out.rows));
    //joinImgs(B2, A2, out, o_B.x, o_B.y);
    stitch(B2, A2, out, o_B.x, o_B.y);
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
    interactImg(dst0, dst1, 8, o_A, o_B);
    joinAndStitch(dst0, dst1, o_A, o_B, out);
    //adjustImg(dst0, dst1, o_A, o_B, out);

    string outputFileName(filename);
    outputFileName = outputFileName.substr(outputFileName.length() - 12);
    outputFileName = "../img/result/" + outputFileName;
    cout << "Writing to: " << outputFileName << endl;
    imwrite(outputFileName, out);

    return 0;
}
