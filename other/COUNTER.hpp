//
//  COUNTER.hpp
//  video_segment
//
//  Created by Yanbo Zhu on 18.10.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#ifndef COUNTER_hpp
#define COUNTER_hpp

#include <iostream>
#include "stdio.h"
#include <time.h> //time_t time()  clock_t clock()
#include  <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Counter
{
    
private:
    double drawAxis(Mat&, Point, Point, Scalar, const float);
    void getOrientation(const vector<Point> &, Mat&);
    double pixeldistance(Point p1, Point p2);

public:
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

#endif /* COUNTER_hpp */
