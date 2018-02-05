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
#include "opencv2/calib3d/calib3d.hpp"
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
    //Point centerpoint(vector<Point> seedtogetherBackup);
    
    // ------- 原 Contour
    double drawAxis(Mat&, Point, Point, Scalar, const float);
    void getOrientation(const vector<Point> &, Mat&);
    double pixeldistance(Point p1, Point p2);
    int elementSize   = 2;
    
public:
    
    Mat RegionGrow(Mat MatIn, Mat MatBlur , double iGrowJudge, vector<Point> seedset);
    //Point regioncenter ;
    vector<Point> seedtogether;
    ~Regiongrowing();
    
    //--- 原 Counter
    Mat FindCounter (Mat MatOut , Mat Frame, Vec3b color);
    double Ratio;
    double EWlong;
    double EWshort;
    Point cntr ;
    double Degree;
    double Area;
    double rectanglewidth;
    double rectangleheight;
    double diagonallength;
};


#endif /* Regiongrowing_hpp */
