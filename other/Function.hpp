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

#include <sys/stat.h>
//#include <sys/types.h> // mkdir
#include <dirent.h>
//#include <fcntl.h>
#include <unistd.h> // 判断文件夹是否存在

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

double decomposematrix(Mat H);
double averagevalue(int num, vector<double> array);
double averagedifference(int num, vector<double> array);
int remove_directory(const char *path);
double pixeldistance(Mat img, Point2f p1, Point2f p2);

#endif /* Function_hpp */
