#ifndef manualCCVar
#define manualCCVar

#include <thread>
#include <iostream>

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

#include "orientation.hpp"
#include "stitcher.hpp"
#include "manual.hpp"

using namespace std;
using namespace cv;

Mat preview;
bool MANUAL_POSITIONING_DONE;

void readInputEvent(int fd, int& key){
    int rd, value, size = sizeof(struct input_event);
    struct input_event ev[64];
    while(true){
        if ((rd = read(fd, ev, size * 64)) < size)
            exit(0);

        value = ev[0].value;
        
        if (value != ' ' && ev[1].value == 1 && ev[1].type == 1){
            key = ev[1].code;
            cout << ev[1].code << endl;
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

#endif
