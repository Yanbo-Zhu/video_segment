//
//  Function.hpp
//  video_segment
//
//  Created by Yanbo Zhu on 2018/2/5.
//  Copyright © 2018年 Zhu. All rights reserved.
//

#ifndef Function_hpp
#define Function_hpp

#include <stdio.h>
#include <iostream>
#include <time.h> //time_t time()  clock_t clock()
#include <math.h>
#include <float.h>
#include "fstream"
#include <string>
#include <list>
#include <vector>
#include <limits>  // numeric_limits<double>::epsilon(): 2.22045e-016
#include <numeric>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double decomposematrix(Mat H);
#endif /* Function_hpp */
