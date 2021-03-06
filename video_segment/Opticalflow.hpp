//
//  Opticalflow.hpp
//  video_segment
//
//  Created by Yanbo Zhu on 2018/2/3.
//  Copyright © 2018年 Zhu. All rights reserved.
//

#ifndef Opticalflow_hpp
#define Opticalflow_hpp

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;


class Opticalflow{
  
private:
    bool addNewPoints(vector<Point2f> pointvektor);
    bool subtractPoints(vector<Point2f> pointvektor);
    bool acceptTrackedPoint_move(int i, vector<uchar> status, vector<Point2f> pointvektor1, vector<Point2f> pointvektor2);
    bool acceptTrackedPoint(int i, vector<uchar> status, vector<Point2f> pointvektor1, vector<Point2f> pointvektor2);
    void gridpoint(int num, Mat image, vector<Point2f>& output);
    
    //int maxCount = 100;    // 检测的最大特征数 被currentmaxnum代替了
    int gridnum = 9;
    int currentmaxnum= gridnum*gridnum;    //  current trackbarvalue
    
    //goodFeaturesToTrack()
    double qLevel = 0.05;    // 特征检测的等级
    double minDist = 8.0;    // 两特征点之间的最小距离
    
public:
    void trackpath(Mat preframe , Mat nextframe, Mat output, vector<vector<Point2f> >& points, vector<Point2f>& initial);
    void matchedpairs(Mat preframe , Mat nextframe, vector<vector<Point2f> >& matchedpairs);
    void relationtrack(Mat preframe , Mat nextframe, vector<vector<Point2f> >& relationparis);
};

#endif /* Opticalflow_hpp */
