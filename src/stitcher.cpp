#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

static void help(char* progName)
{
    cout << endl
        <<  "This program takes input a circular image as input and outputs a square image" << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_name -- default ../data/lena.jpg] [G -- grayscale] "        << endl << endl;
}


void ImgToSquare(const Mat& myImage,Mat& Result);

int main( int argc, char* argv[])
{
    help(argv[0]);
    const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";

    Mat src, dst0, dst1;

    if (argc >= 3 && !strcmp("G", argv[2]))
        src = imread( filename, IMREAD_GRAYSCALE);
    else
        src = imread( filename, IMREAD_COLOR);

    if (src.empty())
    {
        cerr << "Can't open image ["  << filename << "]" << endl;
        return -1;
    }

    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);

    imshow( "Input", src );
    double t = (double)getTickCount();

    ImgToSquare( src, dst0 );

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Hand written function times passed in seconds: " << t << endl;

    imshow( "Output", dst0 );
    waitKey();

  //![kern]
    Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                   -1,  5, -1,
                                    0, -1,  0);
  //![kern]

    return 0;
}
//! [basic_method]
void ImgToSquare(const Mat& Img ,Mat& Res)
{
  //! [8_bit]
    CV_Assert(Img.depth() == CV_8U);  // accept only uchar images
  //! [8_bit]

  //! [create_channels]
    const int nChannels = Img.channels();
    Res.create(Img.rows, Img.cols/2, Img.type());
    cout << "number of channels: " << nChannels << endl;
  //! [create_channels]

  //! [basic_method_loop]
    for(int j = 1 ; j < Img.rows-1; ++j)
    {
        for(int i= 1;i < (Img.cols/2)-1; ++i)
        {
            Res.at<Vec3b>(j, i) = Img.at<Vec3b>(j, i);
        }
    }
  //! [basic_method_loop]

  //! [borders]
    Res.row(0).setTo(Scalar(0));
    Res.row(Res.rows-1).setTo(Scalar(0));
    Res.col(0).setTo(Scalar(0));
    Res.col(Res.cols-1).setTo(Scalar(0));
  //! [borders]
}
//! [basic_method]
