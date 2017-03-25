#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include <thread>

using namespace cv;
using namespace std;

void BS(Ptr<BackgroundSubtractor> pMOG2, Mat a){
    Mat b;
    pMOG2->apply(a, b);
    imshow("MyVideo", a);
    imshow("sadf", b);
}

void runThread(VideoCapture cap, Ptr<BackgroundSubtractor> pMOG2, Mat frame){
    double bSuccess = cap.read(frame);
    if (!bSuccess){
        cout << "Cannot read the frame from video file" << endl;
    }


    thread f2(BS,pMOG2,frame);
    if (f2.joinable()){
        //main is blocked until funcTest1 is not finished
        f2.join();
    }
}

int main(int argc, char* argv[])
{
    Ptr<BackgroundSubtractor> pMOG2;
    VideoCapture cap("Fall.mp4"); // open the video file for reading

    if ( !cap.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
    pMOG2 = createBackgroundSubtractorMOG2();

    while(1)
    {
        Mat frame,gray;
        Mat fgMaskMOG2;

        double t = (double) getTickCount();
        bool bSuccess = cap.read(frame); // read a new frame from video

        if (!bSuccess){
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
        thread f1(BS,pMOG2,frame);

        bSuccess = cap.read(frame);
         if (!bSuccess){
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
        thread f2(BS,pMOG2,frame);

        bSuccess = cap.read(frame);
         if (!bSuccess){
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
        thread f3(BS,pMOG2,frame);

        bSuccess = cap.read(frame);
        if (!bSuccess){
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
        thread f4(BS,pMOG2,frame);


        if (f1.joinable()){
            //main is blocked until funcTest1 is not finished
            f1.join();
        }
        if (f2.joinable()){
            //main is blocked until funcTest1 is not finished
            f2.join();
        }
        if (f3.joinable()){
            //main is blocked until funcTest1 is not finished
            f3.join();
        }
        if (f4.joinable()){
            //main is blocked until funcTest1 is not finished
            f4.join();
        }

        //runThread(cap,pMOG2,frame);
        //runThread(cap,pMOG2,frame);
        //runThread(cap,pMOG2,frame);
        //runThread(cap,pMOG2,frame);

        t = ((double)getTickCount() - t)/getTickFrequency();
        t = (1/t) * 4;
        cout << t<< endl;

        if(waitKey(30) == 27){
            cout << "esc key is pressed by user" << endl;
            break;
       }
    }
    return 0;
}

