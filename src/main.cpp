#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "orientation.hpp"
#include "projection.hpp"
#include "imgops.hpp"
#include "manual.hpp"
#include "stitcher.hpp"

using namespace std;
using namespace cv;


static void help(char* progName)
{
    cout << endl
        <<  "This program stitches raw output of the Samsung Gear 360 together into a complete panorama" << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_name ../img/example.jpg] " << endl << endl;
}


int main( int argc, char* argv[])
{
    help(argv[0]);
    const char* filename = argc >=2 ? argv[1] : "../img/original/360_0042.JPG";

    Mat src, A_fish, B_fish, A, B, out;

    src = imread(filename, IMREAD_COLOR);

    if (src.empty()){
        cerr << "Can't open image ["  << filename << "]" << endl;
        return -1;
    }

    A_fish = src(Rect(0, 0, src.rows, src.cols/2));
    B_fish = src(Rect(src.cols/2, 0, src.rows, src.cols/2));

    A_fish = equalize(A_fish);
    B_fish = equalize(B_fish);

    double t = (double)getTickCount();

    fishToSquare_threaded(A_fish, A);
    fishToSquare_threaded(B_fish, B);

    correctShift(A);
    correctShift(B);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Image transformed in " << t <<" seconds" << endl;

    Orientation o_A, o_B;
    interactImg(A, B, 4, o_A, o_B);
    //joinAndBlend(A, B, o_A, o_B, out);
    //joinAndStitch(A, B, o_A, o_B, out);
    adjustImg(A, B, o_A, o_B, out);


    string outputFileName(filename);
    outputFileName = outputFileName.substr(outputFileName.length() - 12);
    outputFileName = "../img/result/" + outputFileName;
    cout << "Writing to: " << outputFileName << endl;
    imwrite(outputFileName, out);

    return 0;
}
