#ifndef stitcherHVar  
#define stitcherHVar

#define TIME_OUT 60
#define FIXED_POINTS 4

using namespace std;
using namespace cv;

Mat extractDescriptors(const Mat imgA, const Mat imgB);

void adjustImg(Mat A, Mat B, Orientation o_A, Orientation o_B, Mat& out);

void displayGrayMap(Mat img);

float euclidDist(const Vec3b a, const Vec3b b);

void outputCoords(int c, int width);

void computePartialShortestPath(const float* weights, const int height, const int width,
                                const int start, const int goal,
                                vector<Point>& shortest_path, int threadnr);

void checkPathIntegrity(vector<Point>& path, int height, int width);

void fillMask(vector<vector<bool>>& mask, Point p);

bool applyMask(Mat A, Mat B, vector<vector<bool>> mask, Mat& out);

void calcRGBweights(float* weights, Mat A, Mat B);

void calcHSVweights(float* weights, Mat A, Mat B);
//returns true if threads finished correctly
bool monitorThreads();

void stitch(Mat A, Mat B, Mat& out, int x);

void blurTransition(Mat A, Mat B, Mat& out, int x);

void joinAndStitch(Mat A, Mat B, Orientation o_A, Orientation o_B, Mat& out);

#endif
