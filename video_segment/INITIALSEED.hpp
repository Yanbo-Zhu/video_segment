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

using namespace cv;
using namespace std;

class Initalseed
{
    
private:
    static void on_MouseHandle(int event, int x, int y, int flags, void* param);
    void on_Mouse(int event, int x, int y, int flags);
    void DrawLine( Mat &img, Point pt );
    double thresholdvalue, differencegrow;
    Point g_pt;
    
public:
    //Mat MatInBackup = firstFrame.clone();
    
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    vector<Point> initialseedvektor;
    
    void modechoose(int x, Mat firstframe);
    void drawpoint(Mat firstFrame, vector<Point> initialseedvektor);
    
    //void initalseed();
    //Initalseed(Mat x, Mat y);
};



#endif /* INITIALSEED_hpp */
