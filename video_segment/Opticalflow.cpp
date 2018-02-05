//
//  Opticalflow.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 2018/2/3.
//  Copyright © 2018年 Zhu. All rights reserved.
//

#include "Opticalflow.hpp"

void Opticalflow:: trackpath(Mat preframe , Mat nextframe, Mat output, vector<vector<Point2f> >& points, vector<Point2f>& initial){
    
    Mat nextframe_gray;
    Mat preframe_gray;
    cvtColor(preframe, preframe_gray, COLOR_BGR2GRAY);
    
    // ------- 为了计算间多帧前的点+现在帧的点 matchpairs
    vector<Point2f> features;    // 检测的特征
    vector<uchar> status;    // 跟踪特征的状态，特征的流发现为1，否则为0  is set to 1 if the flow for the corresponding features has been found.
    vector<float> err;       // the difference between patches around the original and moved points.
    
    
    // ------- 为了计算相邻两帧间的homography matrix
    vector<Point2f> features_pre;
    vector<Point2f> features_next;
    vector<uchar> status2;
    vector<float> err2;
    
    //OpenCV3版为：
    cvtColor(nextframe, nextframe_gray, COLOR_BGR2GRAY);
    //OpenCV2版为： cvtColor(frame, gray, CV_BGR2GRAY);
    

    // ----   为了计算相邻两帧间的homography matrix
    gridpoint(gridnum, preframe, features_pre);
    // goodFeaturesToTrack(srcImage1_gray, features_pre, currentmaxnum, qLevel, minDist, Mat(),3,true, 0.04);
    
    calcOpticalFlowPyrLK(preframe, nextframe, features_pre, features_next, status2, err2);
    
    int t = 0;
    for (size_t i=0; i<features_next.size(); i++)
    {
        if (acceptTrackedPoint(i, status2, features_pre, features_next))
        {
            features_pre[t] = features_pre[i];
            features_next[t++] = features_next[i];
        }
    }
    
    cout<< "match pair (interfarme) number: "<< t<< endl;
    features_pre.resize(t);
    features_next.resize(t);
    
    //-------- 为了计算间多帧前的点+现在帧的点 matchpairs and drow line of optical flow
    
    if (addNewPoints(points[0]))  // add new feature points
    {
        // goodFeaturesToTrack is used to find shi-tomasi corners
        //        corners – Output vector of detected corners.
        //        maxCorners – Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
        //        qualityLevel – Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
        //        minDistance – Minimum possible Euclidean distance between the returned corners.
        
        
        gridpoint(gridnum, preframe, features);
        //goodFeaturesToTrack(srcImage1_gray, features, currentmaxnum, qLevel, minDist, Mat(),3,true, 0.04);
        points[0].insert( points[0].end(), features.begin(), features.end());
        //cout<< "current corner number to detect: " << points[0].size() <<endl;
        initial.insert(initial.end(), features.begin(), features.end());
    }
    
    if(subtractPoints(points[0]))
    {
        points[0].resize(currentmaxnum);
        initial.resize(currentmaxnum);
    }
    
    //        Mat differenceframe;
    //        absdiff(srcImage2_gray, srcImage1_gray, differenceframe);
    //        //threshold(differenceframe,differenceframe,10,255,THRESH_BINARY);
    //        imshow("differenceframe", differenceframe);
    
    // Pyramidal Luca -kanade method
    //    prevImg – First 8-bit single-channel or 3-channel input image.
    //    nextImg – Second input image of the same size and the same type as prevImg .
    //    prevPts – Vector of 2D points for which the flow needs to be found. The point coordinates must be single-precision floating-point numbers.
    //    nextPts – Output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image. When OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
    //    status – Output status vector. Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.
    //    err – Output vector that contains the difference between patches around the original and moved points.
    // 特征点的新位置可能变化了，也可能没有变化，那么这种状态就存放在后一个参数status中。err就是新旧两个特征点位置的误差了，也是一个输出矩阵。
    
    calcOpticalFlowPyrLK(preframe, nextframe, points[0], points[1], status, err); //The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids.
    
    //cout<< "points[0].size(): " << points[0].size() <<endl;
    //cout<< "points[1].size(): " << points[1].size() <<endl;
    
    // 去掉一些不好的特征点 delete some bad feature points
    int k = 0;
    for (size_t i=0; i<points[1].size(); i++)
    {
        if (acceptTrackedPoint_move(i, status, points[0], points[1]))
        {
            initial[k] = initial[i];
            points[1][k++] = points[1][i];
        }
    }
    
    cout<< "match pairs (optical flow) number: "<< k<< endl;
    points[1].resize(k);
    initial.resize(k);
    
    // 显示特征点和运动轨迹
    for (size_t i=0; i<points[1].size(); i++)
    {
        line(output, initial[i], points[1][i], Scalar(0, 0, 255),2);
        circle(output, points[1][i], 2, Scalar(0, 255, 0), -1); //the last paramter thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn
    }
}

void Opticalflow:: matchedpairs(Mat preframe , Mat nextframe, vector<vector<Point2f> >& matchedpairs){
    
    Mat nextframe_gray;
    Mat preframe_gray;
    cvtColor(preframe, preframe_gray, COLOR_BGR2GRAY);
    cvtColor(nextframe, nextframe_gray, COLOR_BGR2GRAY);
    //OpenCV2版为： cvtColor(frame, gray, CV_BGR2GRAY);
    
    // ------- 为了计算相邻两帧间的homography matrix
    
    vector<Point2f> features_pre;
    vector<Point2f> features_next;
    vector<uchar> status2;
    vector<float> err2;
    
    // ----   为了计算相邻两帧间的homography matrix
    gridpoint(gridnum, preframe, features_pre);
    // goodFeaturesToTrack(srcImage1_gray, features_pre, currentmaxnum, qLevel, minDist, Mat(),3,true, 0.04);
    
    calcOpticalFlowPyrLK(preframe, nextframe, features_pre, features_next, status2, err2);
    
    int t = 0;
    for (size_t i=0; i<features_next.size(); i++)
    {
        if (acceptTrackedPoint(i, status2, features_pre, features_next))
        {
            features_pre[t] = features_pre[i];
            features_next[t++] = features_next[i];
        }
    }
    
    cout<< "match pair (interfarme) number: "<< t<< endl;
    features_pre.resize(t);
    features_next.resize(t);
    
    matchedpairs[0].assign(features_pre.begin(), features_pre.end());
    matchedpairs[1].assign(features_next.begin(), features_next.end());

}



bool Opticalflow:: addNewPoints(vector<Point2f> pointvektor)
{
    return pointvektor.size() <= currentmaxnum;
}

bool Opticalflow:: subtractPoints(vector<Point2f> pointvektor)
{
    return pointvektor.size() > currentmaxnum;
}

// brief: 决定哪些跟踪点被接受
bool Opticalflow:: acceptTrackedPoint_move(int i, vector<uchar> status, vector<Point2f> pointvektor1, vector<Point2f> pointvektor2)
{
    return status[i] && ((abs(pointvektor1[i].x - pointvektor2[i].x) + abs(pointvektor1[i].y - pointvektor2[i].y)) >0.2);
}

bool Opticalflow:: acceptTrackedPoint(int i, vector<uchar> status, vector<Point2f> pointvektor1, vector<Point2f> pointvektor2)
{
    return status[i] && ((abs(pointvektor1[i].x - pointvektor2[i].x) + abs(pointvektor1[i].y - pointvektor2[i].y)) > 0.2);
}

// grid point of original image
void Opticalflow:: gridpoint(int num, Mat image, vector<Point2f>& output){
    output.clear();
    for(int i=1; i<num+1; i++){
        for(int j=1; j<num+1; j++){
            Point2f pt((image.cols*j/(num+1)), (image.rows*i/(num+1)));
            output.push_back(pt);
            //line(image, pt, pt, Scalar(0, 0, 255),4,8,0);
        }
    }
}
