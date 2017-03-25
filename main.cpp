#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

void read_next_frame(VideoCapture cap, Mat& frame, int& index){



    bool bSuccess = cap.read(frame);
    if (!bSuccess){
        cout << "Cannot read the frame from video file" << endl;
        return;
    }
    index++;

    //resize frame
    resize(frame, frame, Size(320,240), 0, 0, INTER_LINEAR);
}

void MOG2(Ptr<BackgroundSubtractor> pMOG2, Mat frame, int index, Mat& foreground){
    pMOG2->apply(frame, foreground);

    // Remove shadow
    threshold(foreground,foreground, 127, 255, THRESH_BINARY);

    // Remove noise
    Mat element_erode = getStructuringElement( MORPH_RECT, Size( 3, 3 ));
    erode(foreground,foreground,element_erode);

    Mat element_dilate = getStructuringElement( MORPH_RECT, Size( 10, 10 ));
    dilate(foreground,foreground,element_dilate);

    Mat element_morpho = getStructuringElement( MORPH_RECT, Size( 7, 7 ));
    morphologyEx(foreground,foreground,MORPH_CLOSE,element_morpho);

}


void Draw_max_contour(Mat foreground, vector<vector<Point> > contours, Mat& max_contour, int& contour_area){
    int largest_contour_index = 0;
    Rect bounding_rect;
    int contour_w, contour_h;


    if (contours.size() > 0){
        double largest_area = contourArea( contours[0]);
        bounding_rect = boundingRect(contours[0]);
        for(int i = 0 ; i < contours.size() ; i++){
            double area = contourArea( contours[i]);
            if(area > largest_area){
                largest_area = area;
                largest_contour_index = i;                  //Store the index of largest contour
                bounding_rect = boundingRect(contours[i]);    // Find the bounding rectangle for biggest contour
            }
        }

        // compute contour area
        contour_w = bounding_rect.width;
        contour_h = bounding_rect.height;
        contour_area = contour_h*contour_w;

        if (40*40  < contour_area && contour_area < 150*150){
            Scalar color( 255,255,255);
            drawContours( max_contour, contours,largest_contour_index, color, CV_FILLED);
            rectangle(foreground, bounding_rect,  Scalar(0,255,255),1, 8,0);
        }
    }
}


int main(int argc, char* argv[])
{

    int frame_index = 0, contour_area;
    Mat frame, foreground;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Rect bounding_rect;

    // open the video file for reading
    VideoCapture cap("Fall.mp4");
    if ( !cap.isOpened() ){
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    // create background subtract mask
    Ptr<BackgroundSubtractor> pMOG2;
    pMOG2 = createBackgroundSubtractorMOG2(300,40,true);

    while(1){

        double t = (double) getTickCount();

        // read frame
        read_next_frame(cap,frame, frame_index);

        // foreground subtract
        MOG2(pMOG2,frame,frame_index,foreground);

        // find contour
        findContours( foreground, contours, hierarchy, 1, 2);

        // draw contour
        Mat max_contour(foreground.rows,foreground.cols,CV_8UC1,Scalar::all(0));
        Draw_max_contour(foreground,contours,max_contour,contour_area);






        imshow("foreground", foreground);
        imshow("asdffasdafsd", max_contour);



        t = ((double)getTickCount() - t)/getTickFrequency();
        t = (1/t) * 4;
        cout << frame_index<< endl;

        if(waitKey(30) == 27){
            cout << "esc key is pressed by user" << endl;
            break;
       }
    }
    return 0;
}


