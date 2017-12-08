//
//  INITIALSEED.hpp
//  video_segment
//
//  Created by Yanbo Zhu on 26.09.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#ifndef INITIALSEED_hpp
#define INITIALSEED_hpp
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;


//Point const a[2] = {Point(100, 100),Point(300, 300) };

class Initialseed
{
    
private:
    static void on_MouseHandle(int event, int x, int y, int flags, void* param);
    void on_Mouse(int event, int x, int y, int flags);
    void DrawLine( Mat &img, Point pt, Vec3b color  );
    double thresholdvalue;
    Point g_pt;
    vector<vector<Point>> set_defaultseed (vector<vector<Point>> seed, int x);
    #define Point_mark_window " Drawing Point "
    
public:
    //Mat MatInBackup = firstFrame.clone();
    
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));

    vector<Point> initialseedvektor;
    double differencegrow;
    int LoopThreshold;
    vector<double> RGThreshold;
    bool threshold_notchange = true;
    Vec3b color;
    vector < vector<double> >data;
    
    Initialseed(Mat Frame);
    Initialseed();
    Initialseed(int x, Mat firstFrame, int objektindex,  double defaultTH[], vector<vector<Point>> defaultSD);
    void modechoose(int x, Mat firstFrame, int objektindex,  double defaultTh[], vector<vector<Point>> defaultSD);
    //void drawpoint(Mat firstFrame, vector<Point> initialseedvektor, Vec3b color);
    void drawpoint(Mat firstFrame, vector<Point> initialseedvektor);
    void newseed(Mat firstFrame);
    
    //vector<double> C;
};



#endif /* INITIALSEED_hpp */
