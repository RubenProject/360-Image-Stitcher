#ifndef manualHVar
#define manualHVar

#define INPUT_QUEUE "/dev/input/event4"
#define EVENT_LEN 16

using namespace std;
using namespace cv;

extern bool MANUAL_POSITIONING_DONE;

void readInputEvent(int fd, int& key);

void parseInputPhysical(int key, Orientation& o_A, Orientation& o_B);

void parseInputASCII(int key, Orientation& o_A, Orientation& o_B);
//display on different thread
void displayPreview();

void interactImg(const Mat A, const Mat B, int scale, Orientation& o_A, Orientation& o_B);

#endif
