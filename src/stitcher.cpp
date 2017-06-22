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

#include <cuda_runtime_api.h>
#include <cuda.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


#define BLOCK_SIZE 16
#define PI 3.1415926535897

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


__global__
void ftos (Matrix X, Matrix Y, float FOV){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    double sx, sy, sz;
    float theta, phi, r;

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
    X.elements[j * X.width + i] = 0.5 * width + r * cos(theta);
    Y.elements[j * Y.width + i] = 0.5 * width + r * sin(theta);
}


void cuda_fishToSquare(const Mat img, Mat& res){
    Matrix X, Y;
    size_t size;
    
    CV_Assert(img.depth() == CV_8U);
    res.create(img.rows, img.cols*2, img.type());
   
    float FOV = PI * 1.12;

    //Allocate X in device memory
    Matrix d_X;
    d_X.width = res.cols;
    d_X.height = res.rows;
    size = res.cols * res.rows * sizeof(float);
    cudaMalloc(&d_X.elements, size);

    //Allocate Y in device memory
    Matrix d_Y;
    d_Y.width = res.cols;
    d_Y.height = res.rows;
    size = res.cols * res.rows * sizeof(float);
    cudaMalloc(&d_Y.elements, size);


    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(res.cols / dimBlock.x, res.rows / dimBlock.y);
    ftos<<<dimGrid, dimBlock>>>(d_X, d_Y, FOV);
    
    //Read X from device memory
    cudaMemcpy(d_X.elements, X.elements, size, cudaMemcpyDeviceToHost);

    //Read Y from device memory
    cudaMemcpy(d_Y.elements, Y.elements, size, cudaMemcpyDeviceToHost);

    //Free device memory
    cudaFree(d_X.elements);
    cudaFree(d_Y.elements);

    int x, y;

    for(int i = 0; i < res.cols; i++){
        for(int j = 0; j < res.rows; j++){
            
            //read coordinates
            x = X.elements[j * X.width + i];
            y = Y.elements[j * X.width + i];

            // Set pixel
            if (x >= 0 && x < img.cols 
             && y >= 0 && y < img.rows){           
                res.at<Vec3b>(j, i) = img.at<Vec3b>((int)y, (int)x);
            }
        }
    }

}



static void help(char* progName)
{
    cout << endl
        <<  "This program takes input a circular image as input and outputs a square image" << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_name -- default ../data/lena.jpg] "        << endl << endl;
}



//transform the fisheye image to a rectangular image
void fishToSquare(const Mat img, Mat& res)
{
    double x, y;
	float theta,phi,r;
	double sx, sy, sz;

	float width = img.cols;
	float height = img.rows;
	float FOV = PI * 1.12; // FOV of the fisheye, eg: 180 degrees

    CV_Assert(img.depth() == CV_8U);  // accept only uchar images

    res.create(img.rows, img.cols*2, img.type());

    //copy image to square, transform coords  
    for(int i = 0; i < res.cols; i++)
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
                res.at<Vec3b>(j, i) = img.at<Vec3b>((int)y, (int)x);
            }
        }
    }
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
    const char* filename = argc >=2 ? argv[1] : "../img/original/360_0014.JPG";

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

    correctShift(dst0);
    correctShift(dst1);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Image transformed in " << t <<" seconds" << endl;
    
    //extractDescriptors(dst0, dst1);

    hconcat(dst0, dst1, out);
    namedWindow("out", WINDOW_NORMAL);
    resizeWindow("out", 1200, 600);
    imshow("out", out);

    string outputFileName(filename);
    outputFileName.append("test.JPG");
    imwrite(outputFileName, out);

    waitKey();


    return 0;
}
