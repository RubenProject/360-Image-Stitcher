#ifndef projectionHVar  
#define projectionHVar

#define THREAD_COUNT 8
#define FOV_FACTOR 1.08
#define PI 3.1415926535897

using namespace std;
using namespace cv;

void fishToSquare_threaded(const Mat img, Mat& res);

#endif
