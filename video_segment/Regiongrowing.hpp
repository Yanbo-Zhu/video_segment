//
//  Regiongrowing.hpp
//  video_segment
//
//  Created by Yanbo Zhu on 09.10.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#ifndef Regiongrowing_hpp
#define Regiongrowing_hpp

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;


class Regiongrowing
{
    
private:
    double differenceValue(Mat MatIn, Point oneseed, Point nextseed, int DIR[][2], double rowofDIR, double B, double G, double R);
    Point centerpoint(vector<Point> seedtogetherBackup);
    
public:
    
    Mat RegionGrow(Mat MatIn, Mat MatBlur , double iGrowJudge, vector<Point> seedset);
    Point regioncenter ;
    vector<Point> seedtogether;
    
};


#endif /* Regiongrowing_hpp */
