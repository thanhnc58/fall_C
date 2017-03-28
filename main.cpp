#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/optflow/motempl.hpp>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <math.h>
#include <thread>
#include <mutex>

using namespace cv;
using namespace std;

const double MHI_DURATION = 0.5;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
const double MHI_THRESHOLD = 100;

deque<Point2i> mcqueue;
Mat mhi; // MHI
int prev_area;
bool DRAW = true;
mutex m;

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


void Draw_max_contour(Mat foreground, vector<vector<Point> > contours,
                      Mat& max_contour, int& contour_area, RotatedRect& min_elip){
    int largest_contour_index = 0;
    int contour_w, contour_h;
    RotatedRect min_rect;


    if (contours.size() > 0){
        double largest_area = contourArea( contours[0]);
        min_rect = minAreaRect(contours[0]);
        for(int i = 0 ; i < contours.size() ; i++){
            double area = contourArea( contours[i]);
            if(area > largest_area){
                largest_area = area;
                largest_contour_index = i;                  //Store the index of largest contour
                min_rect = minAreaRect(contours[i]);    // Find the bounding rectangle for biggest contour
            }
        }
        try{
            min_elip = fitEllipse(Mat(contours[largest_contour_index]));
        }
        catch (exception& e){
            cout << "Standard exception: " << e.what() << endl;
        }



        // compute contour area
        contour_w = min_rect.size.width;
        contour_h = min_rect.size.height;
        contour_area = contour_h*contour_w;

        if (40*40  < contour_area && contour_area < 150*150){
            Scalar color( 255,255,255);
            drawContours( max_contour, contours,largest_contour_index, color, CV_FILLED);
        }
    }
}

void  update_mhi( const Mat& img, Mat& mhi_out, int diff_threshold,
                 double& magnitude, double& mass_speed, double& angle, double& motion_ratio){

    double timestamp = (double)clock() /CLOCKS_PER_SEC; // get current time in seconds
    int centroid_x , cur_centroid_x , centroid_y , cur_centroid_y, cur_m, all_m, pass_m;
    int foreground_h, foreground_w, mhi_h, mhi_w, foreground_area;
    bool noise;
    vector<vector<Point> > contours , cur_contour;
    vector<Vec4i> hierarchy;
    vector<Point> max_contour;
    vector<Moments> m;
    Point2i prev_mc;
    Mat silh, orient, mask, segmask , mhi_out_2 , recent_motion;

    magnitude = 0;
    mass_speed = 0;

    // remove noise
    foreground_area = countNonZero(img);
    noise = false;
    if ( 0 < foreground_area && foreground_area < prev_area){
        if (fabs(foreground_area - prev_area ) > (0.4*prev_area)){
            noise = true;
        }
    }
    prev_area = foreground_area;
    if(foreground_area < 100 || noise){
        mhi = Mat::zeros(img.size(), CV_32F);
    }

    // set mhi
    if( mhi.empty() ) mhi = Mat::zeros(silh.size(), CV_32F);

    // update MHI
    threshold( img, silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it
    motempl::updateMotionHistory( silh, mhi, timestamp, MHI_DURATION );
    mhi.convertTo(mask, CV_8U, 255./MHI_DURATION, (MHI_DURATION - timestamp)*255./MHI_DURATION);
    mhi_out = Mat::zeros(mask.size(), CV_8U);
    insertChannel(mask, mhi_out, 0);

    if(DRAW){
        motempl::calcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
        vector<Rect> brects;
        motempl::segmentMotion(mhi, segmask, brects, timestamp, MAX_TIME_DELTA );
        cvtColor( mhi_out, mhi_out_2, COLOR_GRAY2BGR );
    }


    threshold( mhi_out, recent_motion, MHI_THRESHOLD, 255, CV_THRESH_BINARY );
    findContours( recent_motion, contours, hierarchy, 1, 2);
    if (contours.size()>0){

        // calculate center of mass of mhi
        max_contour = contours[0];
        for(int i = 0 ; i < contours.size() ; i++){
            double a = contourArea(contours[i]);
            if(contourArea(max_contour) < a ){
                max_contour = contours[i];
            }
        }
        Moments mu;
        mu = moments(max_contour,false);
        Point mc;
        centroid_x = int(mu.m10/mu.m00);
        centroid_y = int(mu.m01/mu.m00);
        mc = Point2i(centroid_x , centroid_y);

        if(DRAW) circle(mhi_out_2, mc, 2, (0, 255, 255), 3);

        // calculate mass speed
        if (mcqueue.size() > 0){
            double distance;
            prev_mc = mcqueue[mcqueue.size()-1];
            distance = sqrt(pow(mc.x - prev_mc.x, 2) + pow(mc.y - prev_mc.y, 2));
            if (distance >=15){
                for(int i = 0 ; i < mcqueue.size(); i++){
                    mcqueue[i] = mc;
                }
            }
        }
        mcqueue.push_back(mc);
        if (mcqueue.size() == 15){
            mcqueue.pop_front();
            prev_mc = mcqueue[0];
            mass_speed = sqrt(pow(mc.x - prev_mc.x, 2) + pow(mc.y - prev_mc.y, 2));
        }


        // calculate center of mass of foreground
        findContours( img , cur_contour, hierarchy, 1, 2);
        if(cur_contour.size() > 0){
            Moments cur_mu;
            Point cur_mc;
            cur_mu = moments(cur_contour[0],false);
            cur_centroid_x = int(cur_mu.m10/cur_mu.m00);
            cur_centroid_y = int(cur_mu.m01/cur_mu.m00);
            cur_mc = Point2i(cur_centroid_x , cur_centroid_y);
            if (DRAW){
                circle(mhi_out_2, cur_mc, 2, (255, 255, 0), 3);
                imshow("mhi_out2222", mhi_out_2);
            }
        }

        //calculate magnitude
        double dx , dy;
        dx = cur_centroid_x - centroid_x;
        dy = cur_centroid_y - centroid_y;
        magnitude = sqrt(dx * dx + dy * dy);
        //cout << "magnitude  " << magnitude << "  //";

        //calculate angle
        if (dx != 0 ){
            angle = atan(fabs(dx)/fabs(dy));
            if (dx > 0 && dy > 0) angle = angle;
            if (dx > 0 && dy < 0) angle = 2*M_PI - angle;
            if (dx < 0 && dy < 0) angle = angle + M_PI;
            if (dx < 0 && dy > 0) angle = M_PI - angle;
            angle *= 180/M_PI;
            //cout << "  " <<angle << "  angle  ";
        }

    }

    // calculate motion ratio
    cur_m = countNonZero(silh);
    all_m = countNonZero(mhi_out);
    pass_m = all_m - cur_m;
    if(all_m > 0 && cur_m > 400){
        motion_ratio = round(100*(pass_m/all_m));
    }
    else{
        motion_ratio = 0;
    }
}

void shape_feature(Mat frame, RotatedRect min_elip, float& elip_angle, float& elip_ratio){
    try{
        elip_angle = min_elip.angle;
        elip_ratio = min_elip.size.width / min_elip.size.height;
        Scalar color( 255,255,0);
        ellipse(frame,min_elip,color,1,8);
        //cout << " angle " << elip_angle << "  ratio " << elip_ratio << "  //";

    }
    catch (exception& e){
        cout << "Standard exception: " << e.what() << endl;
    }
}

void run_get_feature(VideoCapture cap, Mat& frame, int& frame_index,
                     Ptr<BackgroundSubtractor> pMOG2, Mat& foreground,
                     vector<vector<Point> > contours, vector<Vec4i> hierarchy,
                      Mat& motion, int& contour_area,
                 double& magnitude, double& mass_speed, double& angle, double& motion_ratio,
                 RotatedRect& min_elip, float& elip_angle, float& elip_ratio){


    // read frame
    m.lock();
    read_next_frame(cap,frame, frame_index);
    m.unlock();

    // foreground subtract
    MOG2(pMOG2,frame,frame_index,foreground);

    // find contour
    findContours( foreground, contours, hierarchy, 1, 2);

    // draw contour

    Mat max_contour(foreground.rows,foreground.cols,CV_8UC1,Scalar::all(0));
    Draw_max_contour(foreground,contours,max_contour,contour_area,min_elip);

    // calculate coefficient depend on mhi
    m.lock();
    update_mhi( max_contour, motion, 30 , magnitude, mass_speed, angle, motion_ratio);
    m.unlock();

    // calculate coefficient depend on elip shape
    shape_feature(frame, min_elip, elip_angle, elip_ratio);


    m.lock();
        imshow( "Motion", motion );
        imshow("foreground", foreground);
        imshow("asdffasdafsd", frame);
        cout << frame_index << "  magnitude " <<magnitude << " mass_speed "<< mass_speed
        << " angle " << angle << " motion_ratio "<< motion_ratio
        <<" elip_angle "<< elip_angle <<" elip_ratio "<< elip_ratio << endl;
    m.unlock();


}

int main(int argc, char* argv[])
{

    int frame_index = 0, contour_area;
    double angle, magnitude, motion_ratio, mass_speed;
    float elip_angle, elip_ratio;
    Mat frame, foreground;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    RotatedRect min_elip;
    Mat motion;

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

        thread t1(run_get_feature,cap, ref(frame), ref(frame_index), pMOG2, ref(foreground),
                        contours, hierarchy, ref(motion), ref(contour_area),
                        ref(magnitude), ref(mass_speed), ref(angle),
                  ref(motion_ratio), ref(min_elip), ref(elip_angle), ref(elip_ratio));

        thread t2(run_get_feature,cap, ref(frame), ref(frame_index), pMOG2, ref(foreground),
                        contours, hierarchy, ref(motion), ref(contour_area),
                        ref(magnitude), ref(mass_speed), ref(angle),
                  ref(motion_ratio), ref(min_elip), ref(elip_angle), ref(elip_ratio));

        thread t3(run_get_feature,cap, ref(frame), ref(frame_index), pMOG2, ref(foreground),
                        contours, hierarchy, ref(motion), ref(contour_area),
                        ref(magnitude), ref(mass_speed), ref(angle),
                  ref(motion_ratio), ref(min_elip), ref(elip_angle), ref(elip_ratio));
        thread t4(run_get_feature,cap, ref(frame), ref(frame_index), pMOG2, ref(foreground),
                        contours, hierarchy, ref(motion), ref(contour_area),
                        ref(magnitude), ref(mass_speed), ref(angle),
                  ref(motion_ratio), ref(min_elip), ref(elip_angle), ref(elip_ratio));
        if (t1.joinable()) t1.join();
        if (t2.joinable()) t2.join();
        if (t3.joinable()) t3.join();
        if (t4.joinable()) t4.join();

        /* elip angle sao lai tinh tu mhi, tinh motion ratio*/




        t = ((double)getTickCount() - t)/getTickFrequency();
        t = (1/t) * 4;
        cout << " fps " << t << endl;

        if(waitKey(30) == 27){
            cout << "esc key is pressed by user" << endl;
            break;
       }
    }
    return 0;
}


