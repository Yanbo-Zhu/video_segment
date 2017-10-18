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
    double getOrientation(const vector<Point> &, Mat&);
    double pixeldistance(Point p1, Point p2);
    
public:
    Mat FindCounter (Mat MatOut , Mat currentFrame, Vec3b color);
    double Ratio ;
    double Degree;
};

#endif /* COUNTER_hpp */
