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
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <iostream>
#include <time.h>

#include "Regiongrowing.hpp"

using namespace cv;
using namespace std;


class Initialseed
{
    
private:
    static void on_MouseHandle(int event, int x, int y, int flags, void* param);
    void on_Mouse(int event, int x, int y, int flags);
    void drawpoint(Mat firstFrame, vector<Point> initialseedvektor);
    double thresholdvalue;
    #define Point_mark_window "Drawing Point"
    int Threshoditerationmax = 300;
    double initialScalediff = 0.01;
    double thresholdstep = 0.1;

    
public:

    vector<Point> initialseedvektor;
    double differencegrow;
    int LoopThreshold;
    vector<double> RGThreshold;
    bool threshold_notchange = true;
    Vec3b color;
    vector < vector<double> >data;
    
    Mat preSegment; // ICP and for all segment image 
    Mat preCounter; // for rigidtrans
    vector<Point> Contourvector; //icp
    
    Initialseed();
    Initialseed(Mat Frame);
    Initialseed(int x, Mat firstFrame, int objektindex,  double defaultTH[], vector<vector<Point>> defaultSD);
    void modechoose(int x, Mat firstFrame, int objektindex,  double defaultTh[], vector<vector<Point>> defaultSD);
    
    void newseed(Mat firstFrame);
    void randomseed(Mat firstFrame, int width, int height);
    bool checkThreshold(Mat Frame, double relation);
    
};

#endif /* INITIALSEED_hpp */
