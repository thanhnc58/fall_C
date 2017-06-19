#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/optflow/motempl.hpp>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <queue>
#include <math.h>
#include <thread>
#include <mutex>

using namespace cv;
using namespace std;

const double MHI_DURATION = 40;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
const double MHI_THRESHOLD = 100;
const double CONST_MC_SPEED_1 = 22;
const double CONST_MRATE_1 = 40;
const double CONST_MAGN_1 = 13;
const double MASS_EXTREAM_CONT = 50;

const double CONST_MC_SPEED_2 = 20;
const double CONST_MRATE_2 = 40;
const double CONST_MAGN_2 = 10;

const double CONST_AR = 0.4;
const double CONST_ANGLE = 0;

const double MIN_CONT_AREA = 40* 40;
const double MAX_CONT_AREA = 150 * 150;

deque<Point2i> mcqueue;
Mat mhi; // MHI
int prev_area , large_motion_frame , move_down_frame , apparently_fall_frame;
bool DRAW = true;
mutex m;
int mhi_timestamp = 0;
int falllll = 0;
int new_vid = 0;


int read_next_frame(VideoCapture cap, Mat& frame, int& index){



    bool bSuccess = cap.read(frame);
    if (!bSuccess){
        cout << "Cannot read the frame from video fileeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" << endl;
        return 0;
    }
    index++;


    //resize frame
    resize(frame, frame, Size(320,240), 0, 0, INTER_LINEAR);
    return 1;
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
                      Mat& max_contour, int& contour_area, RotatedRect& min_elip, RotatedRect& min_rect){
    int largest_contour_index = 0;
    int contour_w, contour_h;

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



        // compute contour area
        contour_w = min_rect.size.width;
        contour_h = min_rect.size.height;
        contour_area = contour_h*contour_w;

        if (40*40  < contour_area && contour_area < 150*150){
            Scalar color( 255,255,255);
            drawContours( max_contour, contours,largest_contour_index, color, CV_FILLED);
        }
    }
    else{
        contour_area = 0;
    }
}

void update_mhi( const Mat& img, Mat& mhi_out, int diff_threshold){
    mhi_timestamp = mhi_timestamp + 1;
    double timestamp = mhi_timestamp; // get current time in seconds
    //cout <<"timestaep: "<< timestamp << endl;
    int foreground_h, foreground_w, mhi_h, mhi_w, foreground_area;
    bool noise;
    vector<vector<Point> > contours , cur_contour;
    vector<Vec4i> hierarchy;
    Mat silh, orient, mask, segmask , mhi_out_2 , recent_motion;

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




    threshold( img, silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it

    // set mhi
    if( mhi.empty() ){
        mhi = Mat::zeros(img.size(), CV_32F);
        //cout << "asdffasd" ;

    }

    // update MHI
    motempl::updateMotionHistory( silh, mhi, timestamp, MHI_DURATION );
    mhi.convertTo(mhi_out, CV_8U, 255./MHI_DURATION, (MHI_DURATION - timestamp)*255./MHI_DURATION);


}

void mhi_coefficient(const Mat& img, Mat& mhi_out, int diff_threshold,
                 double& magnitude, double& mass_speed, double& angle, double& motion_ratio, RotatedRect min_rect){
    double timestamp = (double)clock() /CLOCKS_PER_SEC , cur_m, all_m, pass_m; // get current time in seconds
    int centroid_x , cur_centroid_x , centroid_y , cur_centroid_y;
    int foreground_h, foreground_w, mhi_h, mhi_w, foreground_area;
    bool noise;
    vector<vector<Point> > contours , cur_contour;
    vector<Vec4i> hierarchy;
    vector<Point> max_contour;
    vector<Moments> m;
    Point2i prev_mc;
    Mat silh, orient, mask, segmask , mhi_out_2 , recent_motion;

    if(DRAW){
        motempl::calcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
        vector<Rect> brects;
        motempl::segmentMotion(mhi, segmask, brects, timestamp, MAX_TIME_DELTA );
        cvtColor( mhi_out, mhi_out_2, COLOR_GRAY2BGR );
    }


    threshold( mhi_out, recent_motion, MHI_THRESHOLD, 255, CV_THRESH_BINARY );
    findContours( recent_motion, contours, hierarchy, 1, 2);
    //cout<<"contours size = "<< contours.size() << "//////////////";
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
//        if (mcqueue.size() > 0){
//            double distance;
//            prev_mc = mcqueue[mcqueue.size()-1];
//            distance = sqrt(pow(mc.x - prev_mc.x, 2) + pow(mc.y - prev_mc.y, 2));
//            if (distance >= 22){
//                for(int i = 0 ; i < mcqueue.size(); i++){
//                    mcqueue[i] = mc;
//                }
//            }
//        }
//        mcqueue.push_back(mc);
//        if (mcqueue.size() == 15){
//
//            mcqueue.pop_front();
//            prev_mc = mcqueue[0];
//            mass_speed = sqrt(pow(mc.x - prev_mc.x, 2) + pow(mc.y - prev_mc.y, 2));
//            cout << mc.x << "," << mc.y <<"  / "<< prev_mc.x << "," << prev_mc.y << endl;
//        }


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
        //cout << "magnitude  " << magnitude << "  //////////////////////";

        //calculate angle
        if (dx != 0 ){
            //cout << "dx: " << dx << "," << dy << endl;
            angle = atan(fabs(dy)/fabs(dx));
            if (dx > 0 && dy > 0) angle = angle;
            if (dx > 0 && dy < 0) angle = 2*M_PI - angle;
            if (dx < 0 && dy < 0) angle = angle + M_PI;
            if (dx < 0 && dy > 0) angle = M_PI - angle;
            //cout << "  " <<angle << "  angle  " << atan(1.0) << endl;
            angle *= 180/M_PI;
            //cout << "  " <<angle << "  angle  " << endl;
        }

    }

    // calculate motion ratio
    cur_m = countNonZero(img);

    all_m = countNonZero(mhi_out);
    pass_m = all_m - cur_m;
    if(all_m > 0 && cur_m > 400){
        motion_ratio = round(100*(pass_m/all_m));
    }
    else{
        motion_ratio = 0;
    }


}

void shape_feature(Mat frame, Mat motion,double& mass_speed,float& mass_extreame_distance, float& elip_angle, float& elip_ratio){
    RotatedRect min_rect;
    int largest_contour_index = 0;
    Mat elip_motion;
    RotatedRect min_elip;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int centroid_x , cur_centroid_x , centroid_y , cur_centroid_y;
    Point2i prev_mc;

    threshold( motion, elip_motion, 240, 255, CV_THRESH_BINARY );
    //imshow("elip_motionnnn",elip_motion);
    findContours( elip_motion, contours, hierarchy, 1, 2);
    if (contours.size()>0){
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
        //cout <<" test contour" <<contours.size() << endl;
        try{
            min_elip = fitEllipse(Mat(contours[largest_contour_index]));
        }
        catch (exception& e){
            cout << "Standard exception: " << e.what() << endl;
        }
        Moments mu;
        mu = moments(contours[largest_contour_index],false);
        Point mc;
        centroid_x = int(mu.m10/mu.m00);
        centroid_y = int(mu.m01/mu.m00);
        mc = Point2i(centroid_x , centroid_y);

        // calculate mass speed
        if (mcqueue.size() > 0){
            double distance;
            prev_mc = mcqueue[mcqueue.size()-1];
            distance = sqrt(pow(mc.x - prev_mc.x, 2) + pow(mc.y - prev_mc.y, 2));
            if (distance >= 22){
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
//            cout << mc.x << "," << mc.y <<"  / "<< prev_mc.x << "," << prev_mc.y << endl;
        }

        int w, h ;
        Point2f HP, HP2;
        w = min_rect.size.width;
        h = min_rect.size.height;
        Point2f box[4];
        min_rect.points(box);
        if (w < h){
            HP = ((box[1] + box[2]) / 2);
            HP2 = ((box[0] + box[3]) / 2);
        }

        else{
            HP = ((box[0] + box[1]) / 2);
            HP2 = ((box[2] + box[3]) / 2);
        }

        mass_extreame_distance = max(fabs(mc.y - HP.y), fabs(mc.y - HP2.y));
    }


    try{
        elip_angle = min_rect.angle;
        elip_ratio = min_elip.size.width / min_elip.size.height;
        if(elip_ratio < 1){
            elip_angle = -(elip_angle-90);
        }


        if(DRAW){
            Scalar color( 255,255,0);
            ellipse(frame,min_elip,color,1,8);
        }

        //cout << " angle " << elip_angle << "  ratio " << elip_ratio << "  //";

    }
    catch (exception& e){
        cout << "Standard exception: " << e.what() << endl;
    }
}

int predict(double magnitude, double mass_speed, double angle, double motion_ratio,
            float elip_angle, float elip_ratio, int foreground_area,int frame_index, float mass_extreame_distance){
    int result = 0;
    if (frame_index == 0 ){
        large_motion_frame = -1;
        apparently_fall_frame = -1;
        move_down_frame = -1;
    }

    //if it can be fall, observe in 50 following frame if it dont have any movement then give a conclusion
    if (apparently_fall_frame > 0){
        if ((frame_index - apparently_fall_frame ) > 100){
//            if ((frame_index - apparently_fall_frame ) > 80 || foreground_area > 0.5 * 40 * 40){
//                large_motion_frame = -1;
//                apparently_fall_frame = -1;
//                move_down_frame = -1;
//                return 0;
//            }

            if (foreground_area < 0.5 * 40*40){
                large_motion_frame = -1;
                apparently_fall_frame = -1;
                move_down_frame = -1;
                return 2;
            }
        }
        result = 1;
    }

    if ((MIN_CONT_AREA < foreground_area && foreground_area < MAX_CONT_AREA) &&
        (100 > motion_ratio && motion_ratio>= CONST_MRATE_2 && (60 > magnitude && magnitude >= CONST_MAGN_2)
        && 80 >= mass_speed && mass_speed >= CONST_MC_SPEED_2) &&
        (0 <= angle && angle <= 180)){
            /*
        if(80 > motion_ratio && motion_ratio>= CONST_MRATE_1) cout << "motion_ratio" << endl;
        if(80 >= mass_speed && mass_speed >= CONST_MC_SPEED_1) cout << " mass_speed" << endl;
        if(60 > magnitude && magnitude >= CONST_MAGN_1) cout << "mag" << endl;
        if(0 <= angle && angle <= 180) cout << "angle";
            */
        cout <<"index= "<< frame_index << "largeeeeeeeee" << endl;
        large_motion_frame = frame_index;
        }
    if (large_motion_frame > 0){
        if (frame_index - large_motion_frame <= 30){
            if ((MIN_CONT_AREA < foreground_area && foreground_area < MAX_CONT_AREA)
                && (40 <= angle && angle <= 140)){
                move_down_frame = frame_index;
               cout <<"index= "<< frame_index << "downnnnnnnnnnnnnnnnnnn" <<endl ;
            }
            else{
                large_motion_frame = -1;
                move_down_frame = -1;
            }
        }
    }
    if (move_down_frame > 0){
        if (frame_index - move_down_frame <= 20){
            if ( 1 > elip_ratio && elip_ratio > CONST_AR && mass_extreame_distance < MASS_EXTREAM_CONT && (
                elip_angle > 0 && fabs(elip_angle - 90) >= CONST_ANGLE )){
                    cout <<"index= "<< frame_index << "falllllllllllllll" <<endl ;
                apparently_fall_frame = frame_index;
            }
        }
        else{
            large_motion_frame = -1;
            move_down_frame = -1;
        }
    }
    return result;

}

void run_get_feature(VideoCapture cap, Mat& frame, int& frame_index,
                     Ptr<BackgroundSubtractor> pMOG2, Mat& foreground,
                     vector<vector<Point> > contours, vector<Vec4i> hierarchy,
                      Mat& motion, int& contour_area,
                 double& magnitude, double& mass_speed, double& angle, double& motion_ratio,
                 RotatedRect& min_elip, float& elip_angle, float& elip_ratio , float& mass_extreame_distance){

    RotatedRect min_rect;


    // read frame
//    m.lock();
//    read_next_frame(cap,frame, frame_index);
//    m.unlock();

    // foreground subtract
    MOG2(pMOG2,frame,frame_index,foreground);

    // find contour
    findContours( foreground, contours, hierarchy, 1, 2);

    // draw contour

    Mat max_contour(foreground.rows,foreground.cols,CV_8UC1,Scalar::all(0));
    Draw_max_contour(foreground,contours,max_contour,contour_area,min_elip, min_rect);

    // calculate coefficient depend on mhi
    m.lock();
    update_mhi( max_contour, motion, 30);
    m.unlock();

    mhi_coefficient(max_contour, motion, 30 , magnitude, mass_speed, angle, motion_ratio, min_rect);

    // calculate coefficient depend on elip shape
    shape_feature(frame, motion,mass_speed,mass_extreame_distance, elip_angle, elip_ratio);


    m.lock();
        imshow( "Motion", motion );
        imshow("foreground", foreground);
        imshow("Lecture", frame);
        cout << fixed;
        cout << setprecision(2);
        cout << frame_index << "contour " << contour_area <<" mag " << magnitude << " massS "<< mass_speed
        << " angle " << angle << " motionR "<< motion_ratio
        <<" Eangle "<< elip_angle <<" Eratio "<< elip_ratio << " mae "<<mass_extreame_distance << endl;
        int a = -1;
        a = predict(magnitude,mass_speed,angle,motion_ratio,elip_angle,elip_ratio,contour_area,frame_index, mass_extreame_distance);
        cout << a << endl;
        if(a == 2) falllll++;
        cout << falllll << endl;
    m.unlock();


}


struct FrameData{
    Mat frame;
    int index;
    Mat foreground;
    Mat max_contour;
    Mat motion;
    int contour_area;
    RotatedRect min_elip;
    RotatedRect min_rect;
    double magnitude;
    double mass_speed;
    double angle;
    double motion_ratio;
    float elip_angle;
    float elip_ratio;
    float mass_extreame_distance;

};


struct DisplayThread {
    typedef pair<Mat,int> FrameIndex;
    struct Compare {
        bool operator() (const FrameIndex& f1, const FrameIndex& f2) {
            return f1.second > f2.second;
        }
    };
    typedef priority_queue< FrameIndex, vector<FrameIndex>, Compare > FrameIndexQueue;

    unordered_map<string, FrameIndexQueue> queues;
    unordered_map<string, int> curIndex;
    mutex lock;

    thread getThread() {
        return thread([=]{ doDisplayImage(); });
    }
    void doDisplayImage() {
        while (true) {
            this_thread::sleep_for(chrono::milliseconds(1));
            bool hasImage = false;
            for (auto& p : queues) {
                const string& winName = p.first;
                auto& q = p.second;
//                cout << "checkDisplay win = " << winName << " index = " << curIndex[winName]
//                     << " top = " << q.top().second << endl;
                if (!q.empty() && q.top().second == 1) curIndex[winName] = 1;
                if (q.empty() || q.top().second != curIndex[winName]) continue;
//                cout << "display " << winName << " size = " << q.size() << endl;

//                cout << "doDisplay win = " << winName << " index = " << curIndex[winName] << endl;

                lock.lock();
                int qsize = q.size();
                FrameIndex f = q.top();
                q.pop();
                curIndex[winName]++;
                lock.unlock();

                if (qsize < 200) imshow(winName, f.first);
                hasImage = true;
            }
            if (hasImage) waitKey(1);
        }
    }

    void display(const string& windowName, const Mat& frame, int index) {
        lock.lock();
        if (queues.find(windowName) == queues.end()) {
            curIndex[windowName] = 1;
        }
        queues[windowName].push( make_pair(frame,index) );
        lock.unlock();
    }
};

typedef queue<FrameData> FrameDataQueue;
struct FrameDataCompare {
    bool operator() (const FrameData& f1, const FrameData& f2) {
        return f1.index > f2.index;
    }
};

typedef priority_queue< FrameData, vector<FrameData>, FrameDataCompare > FrameDataPQueue;

struct LockedThread {
    mutex lock;
};

struct HasDisplayThread {
    DisplayThread* mDisplayThread = nullptr;

    void setDisplayThread(DisplayThread* pThread) {
        mDisplayThread = pThread;
    }

    void display(const string& windowName, const Mat& frame, int index) {
        if (mDisplayThread == nullptr)
            cout << "null display thread" << endl;
        else
            mDisplayThread->display(windowName, frame, index);
    }
};

struct MHIThread : public  FrameDataPQueue, public LockedThread, public HasDisplayThread
{
    int curIndex = 1;
    Mat mhi; // MHI


    thread getThread() {
        return thread([=]{ doComputeMHI(); });
    }
    void doComputeMHI() {

        while (true) {
            this_thread::sleep_for(chrono::milliseconds(1));
            if(!this->empty() && this->top().index == 1) curIndex = 1;
            if (this->empty()  || this->top().index != curIndex) continue;
            lock.lock();
//            cout << "mhi size = " << size() << endl;
            cout << "cur index  " << curIndex <<endl;
            FrameData elem = this->top();
            curIndex++;
            this->pop();
            lock.unlock();

            update_mhi(elem.max_contour,elem.motion,30);
            mhi_coefficient(elem.max_contour, elem.motion, 30 , elem.magnitude,
                             elem.mass_speed, elem.angle, elem.motion_ratio, elem.min_rect);
            shape_feature(elem.frame, elem.motion, elem.mass_speed,elem.mass_extreame_distance, elem.elip_angle, elem.elip_ratio);
            int a = predict(elem.magnitude, elem.mass_speed, elem.angle, elem.motion_ratio,
                    elem.elip_angle, elem.elip_ratio, elem.contour_area, elem.index, elem.mass_extreame_distance);
            char message[100];
            sprintf(message, "%d", a);
//            if (a == 2) exit(1);

            //sleep(0.5);
            display("Motion", elem.motion, elem.index);

            std::cout << std::fixed;
            std::cout << std::setprecision(1);
            cout << elem.index << " " << "contour " << elem.contour_area <<" mag " << elem.magnitude << " massS "<< elem.mass_speed
                << " angle " << elem.angle << " motionR "<< elem.motion_ratio
                <<" Eangle "<< elem.elip_angle <<" Eratio "<< elem.elip_ratio << " mas_ex " << elem.mass_extreame_distance << endl;



            Mat frame = elem.frame.clone();
            if(a == 0 ){
                cv::putText(frame, " ", Point(100,100), cv::FONT_HERSHEY_SIMPLEX, 2, Scalar(255,255,255), 2);
            }
            if (a==1){
                    cv::putText(frame, "OPPSSS", Point(100,100), cv::FONT_HERSHEY_SIMPLEX, 2, Scalar(255,255,255), 2);
            }
            if (a == 2) {

                falllll++;
                cv::putText(frame, "Falllllll", Point(100,100), cv::FONT_HERSHEY_SIMPLEX, 2, Scalar(255,0,0), 2);
                //system("canberra-gtk-play -f gg.wav");
            }
            cout << falllll << endl;

            ellipse(frame,elem.min_elip,Scalar(255,255,0),1,8);
            display("Result", frame, elem.index);
        }
    }
    void pushFrameData(const FrameData& data) {
        lock.lock();
        push(data);
        lock.unlock();
    }
};

struct ContourThread : public FrameDataQueue, public LockedThread, public HasDisplayThread
{
    MHIThread* mMHIThread = nullptr;
    Ptr<BackgroundSubtractor> pMOG2;

    thread getThread() {
        return thread([=]{ doFindContour(); });
    }
    void doFindContour() {
        pMOG2 = createBackgroundSubtractorMOG2(300,40,true);
        while (true) {
            this_thread::sleep_for(chrono::milliseconds(1));
            if (this->empty()) continue;
            lock.lock();
//            cout << "contour size = " << size() << endl;
            FrameData data = this->front();
            this->pop();
            lock.unlock();
            if (data.index < 5) {
                pMOG2 = createBackgroundSubtractorMOG2(300,40,true);
                cout << "asdfjhaksjdfhkasdjfhkjasfdhjkahsdkjfhajksdhfksakd"<< endl;
            }
            getContour(data);
            displayData(data);
            if (mMHIThread) mMHIThread->pushFrameData(data);
        }
    }

    void getContour(FrameData& frameData) {
//        cout << "getContour id = " << frameData.index << endl;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        MOG2(pMOG2, frameData.frame, frameData.index, frameData.foreground);

        findContours( frameData.foreground, contours, hierarchy, 1, 2);
        Mat max_contour(frameData.foreground.rows,frameData.foreground.cols,CV_8UC1,Scalar::all(0));
        frameData.max_contour = max_contour;
        Draw_max_contour(frameData.foreground,contours,frameData.max_contour,frameData.contour_area,
                         frameData.min_elip, frameData.min_rect);
    }

    void displayData(const FrameData& data) {
//        cout << "displayData id = " << data.index << endl;
        display("Original", data.frame, data.index);
        display("Foreground", data.max_contour, data.index);
//        display("Contour", data.max_contour, data.index);
    }


    void setMHIThread(MHIThread* pThread) {
        mMHIThread = pThread;
    }

    void pushFrame(Mat frame, int index) {
        FrameData data;
        data.frame = frame;
        data.index = index;
        push(data);
    }
};

int main()
{
    Mat frame;
    int index = 0;
    int n = 1;
    string filename = "data/vid" + to_string(n) + ".mp4" ;
    VideoCapture cap(filename);
    if ( !cap.isOpened() ){
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    int n_contour = 1;
    ContourThread contourQueue[n_contour];
    DisplayThread displayQueue;
    MHIThread mhiQueue;

    mhiQueue.setDisplayThread(&displayQueue);

    vector<thread> allThreads;
    for (int i = 0; i < n_contour; i++) {
        contourQueue[i].setDisplayThread(&displayQueue);
        contourQueue[i].setMHIThread(&mhiQueue);
        allThreads.push_back( contourQueue[i].getThread() );
    }
    allThreads.push_back( displayQueue.getThread() );
    allThreads.push_back( mhiQueue.getThread() );

    bool shouldRead = true;
    while (true) {
        int check;
        this_thread::sleep_for(chrono::milliseconds(50));
        if (shouldRead) check = read_next_frame(cap, frame, index);
        cout << n <<endl ;

        if (frame.empty() || check == 0) {

            cout << "Empty frame" << endl;
            cap.release();
            mhi_timestamp = 0;
            n++;
            filename = "data/vid" + to_string(n) + ".mp4" ;
            cap.open(filename);
            new_vid = 1;
            index = 0;
            continue;
        }
        cout << index << endl;
        int threadID = index % n_contour;
        contourQueue[threadID].lock.lock();

        if ( contourQueue[threadID].size() < 100 ) {
            contourQueue[threadID].pushFrame(frame.clone(),index);
            shouldRead = true;
        } else  {
            shouldRead = false;
        }

        contourQueue[threadID].lock.unlock();
//        cout << index << endl;
    }

    for (auto& t : allThreads) t.join();
}

int temp_main(int argc, char* argv[])
{

    int frame_index = 0, contour_area;
    double angle, magnitude, motion_ratio, mass_speed;
    float elip_angle, elip_ratio,mass_extreame_distance;
    Mat frame, foreground;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    RotatedRect min_elip;
    Mat motion;
    int n = 13;
    //  open the video file for reading
    string filename = "vid1.mp4";
//  string filename = "Lecture_Room/video (" + to_string(n) + ").avi" ;
    VideoCapture cap(filename);
    if ( !cap.isOpened() ){
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    // create background subtract mask
    Ptr<BackgroundSubtractor> pMOG2;
    pMOG2 = createBackgroundSubtractorMOG2(300,40,true);

    while(1){

        double t = (double) getTickCount();

//        thread t1(run_get_feature,cap, ref(frame), ref(frame_index), pMOG2, ref(foreground),
//                        contours, hierarchy, ref(motion), ref(contour_area),
//                        ref(magnitude), ref(mass_speed), ref(angle),
//                  ref(motion_ratio), ref(min_elip), ref(elip_angle), ref(elip_ratio) , ref(mass_extreame_distance));
//
//
//        if (t1.joinable()) t1.join();
        int check = read_next_frame(cap,frame, frame_index);
        if (frame.empty() || check == 0 ) {
            //return 0;
            //cout << "sdfafsdafsdasfsfd";
            //index = 0;
            cout << "Empty frame" << endl;
            cap.release();
            mhi_timestamp = 0;
            n++;
            filename = "rtsp://192.168.1.8/user=admin_password=123456_channel=1_stream=1.sdp";
            cap.open(filename);
            pMOG2 = createBackgroundSubtractorMOG2(300,40,true);
            //index = 0;
            continue;
        }


        run_get_feature(cap,frame,frame_index,pMOG2,foreground,contours,hierarchy,motion,contour_area,magnitude,mass_speed
                        ,angle,motion_ratio,min_elip,elip_angle,elip_ratio,mass_extreame_distance);


        cout<< n << endl;
        t = ((double)getTickCount() - t)/getTickFrequency();
        t = (1/t) * 4;
        cout << " " << endl;
        char key = waitKey(5);
        if(key == 'p')
            waitKey() && 0xff;
        if(waitKey(1) == 27){
            cout << "esc key is pressed by user" << endl;
            break;
       }
    }
    return 0;
}
