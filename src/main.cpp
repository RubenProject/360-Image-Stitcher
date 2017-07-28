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
