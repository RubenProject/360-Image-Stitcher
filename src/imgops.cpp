#ifndef imgopsCCVar
#define imgopsCCVar

#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imgops.hpp"

using namespace std;

Mat equalize(Mat src){
    imwrite("not_equalized.jpg", src);
    if(src.channels() >= 3){
        Mat hsv;
        cvtColor(src, hsv, CV_BGR2HSV);

        vector<Mat> channels;
        split(hsv,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,result);
        cvtColor(hsv, result, CV_HSV2BGR);
        imwrite("equalized.jpg", result);
        return result;
    }
    return Mat();
}


//corrects the shift present after projection
void correctShift(Mat& img){
    Mat res;
    Mat img_a = img(Rect(img.cols/4*3, 0, img.cols/4, img.rows));
    Mat img_b = img(Rect(0, 0, img.cols/4*3, img.rows));
    hconcat(img_a, img_b, res);
    img = res;
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


Mat translateImg(Mat src, int y){
    Mat res;
    if (y < 0){
        src = src(Rect(0, -y, src.cols, src.rows + y));
        Mat p(-y, src.cols, src.type(), Scalar(0,0,0));
        vconcat(src, p, res);
        return res;
    } else if (y > 0) {
        src = src(Rect(0, y, src.cols, src.rows - y));
        Mat p(y, src.cols, src.type(), Scalar(0,0,0));
        vconcat(p, src, res);
        return res;
    } else {
        res = src(Rect(0, 0, src.cols, src.rows));
        return res;
    }
}


void joinImgs(Mat A, Mat B, Mat& out, int x){
    A = A(Rect(0, 0, A.cols - x/2, A.rows));
    B = B(Rect(x/2, 0, B.cols - x/2 ,B.rows));
    hconcat(A, B, out);
}


Mat rotateImg(Mat src, double angle){
    Mat res;
    Point center = Point(src.cols/2, src.rows/2);
    Mat r = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(src, res, r, res.size());
    return res;
}

#endif
