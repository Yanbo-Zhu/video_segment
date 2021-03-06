//
//  COUNTER.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 18.10.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#include "COUNTER.hpp"


Mat Counter::FindCounter (Mat MatOut , Mat Frame, Vec3b color)
{
    Mat MatoutGray;
    Mat FramemitCounter = Frame.clone();
    // dilate MatOut
    int elementSize   = 2;
    Mat element = getStructuringElement(MORPH_RECT, Size(2*elementSize+1,2*elementSize+1));
    dilate(MatOut, MatOut, element);
    erode(MatOut, MatOut, element);
    //morphologyEx(MatOut, MatOut, MORPH_CLOSE, element)
    
    cvtColor(MatOut,MatoutGray,CV_BGR2GRAY);
    threshold(MatoutGray,MatoutGray,20,255,THRESH_BINARY);
    
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    findContours(MatoutGray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    // 多边形逼近轮廓 + 获取矩形边界框
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center (contours.size());
    vector<float>radius (contours.size());
    for( unsigned int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 1, true );//用指定精度逼近多边形曲线
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );//计算点集的最外面（up-right）矩形边界  return variable type : Rect
        minEnclosingCircle( contours_poly[i], center[i], radius[i] );//对给定的 2D点集，寻找最小面积的包围圆形
    }
    
    for (size_t i = 0; i < contours.size(); i++)
    {
        // Calculate the area of each contour
        Area = contourArea(contours[i]);  // 面积就是包含了多少像素点
        
        // Ignore contours that are too small or too large
        //if (area < 1e2 || 1e5 < area) continue;
        
        // Draw each contour only for visualisation purposes
        drawContours(FramemitCounter, contours, static_cast<int>(i), color, 2, 8, hierarchy, 0);
        
        rectangle( FramemitCounter, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0 );// 绘制边框矩形
        rectanglewidth = boundRect[i].width;
        rectangleheight = boundRect[i].height;
        diagonallength = pixeldistance(boundRect[i].tl(),  boundRect[i].br());
//      circle( FramemitCounter, center[i], (int)radius[i], color, 1, 8, 0 ); // draw the circle 绘制圆形边框
        
        getOrientation(contours[i], FramemitCounter);
    }
    
    return FramemitCounter;
}


void Counter::getOrientation(const vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
    
    //Store the center of the object
    cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)), static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
    
    EWlong = eigen_val[0];
    EWshort = eigen_val[1];
    
    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    
    // 注意下面的参数
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    //cout<< "p1  Row:" << p1.y << "   Column: " << p1.x <<endl;
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    
    double pixelabstand[2];
    pixelabstand[0] = drawAxis(img, cntr, p1, Scalar(0, 255, 0), 2.5); // Green line. long axis
    pixelabstand[1] = drawAxis(img, cntr, p2, Scalar(255, 255, 0), 3); // lightly blue line . short axis
    
    //Ratio = pixelabstand[0]/ pixelabstand[1];
    Ratio = eigen_val[0]/ eigen_val[1];
    //cout<< "Ratio: " << Ratio <<endl;
    
    double angle = atan2( - (eigen_vecs[0].y), eigen_vecs[0].x); // orientation in radians
    //cout<< "Eigenvektor long axis::   Vektor in Row-direction: " << eigen_vecs[0].y << " / Vektor in Column-direction: " << eigen_vecs[0].x <<endl;
    
    Degree = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
    //cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
    //return angle;
}


double Counter::drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle;
    double hypotenuse; //直角三角形的斜边
    double twopixeldistance;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    
    twopixeldistance = pixeldistance(p, q);
    
    line(img, p, q, colour, 1, CV_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
    
    return twopixeldistance;
}

double Counter::pixeldistance(Point p1, Point p2)
{
    double a = p1.x - p2.x;
    double b = p1.y - p2.y;
    return (sqrt((a*a)+(b*b))); // a^2 not equal to a*a. a^2 has differnt meaning in Opencv
}

