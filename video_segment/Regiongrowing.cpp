//
//  Regiongrowing.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 09.10.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#include "Regiongrowing.hpp"

Mat Regiongrowing:: RegionGrow(Mat MatIn, Mat MatBlur , double iGrowJudge, vector<Point> seedset) //iGrowPoint: seeds 为种子点的判断条件，iGrowJudge: growing condition 为生长条件
{

    Mat Segment(MatIn.size(),CV_8UC3,Scalar(0,0,0));
    Mat MatLabel(MatIn.size(),CV_8UC1,Scalar(0));
    
    // intialize MatGrownow
    Mat MatGrownow(MatIn.size(),CV_8UC3,Scalar(0,0,0));
    
    for(size_t i=0;i<seedset.size();i++)
    {
        MatGrownow.at<Vec3b>(seedset[i]) = MatIn.at<Vec3b>(seedset[i]);
        //seedtogether.push_back(seedset[i])
    }
    
    seedtogether.clear();
    seedtogether = seedset;
    
    //生长方向顺序数据
    //int DIR[8][2]={{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
    int DIR[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    double rowofDIR= sizeof(DIR)/sizeof(DIR[0]);
    
    // calculate the initial B G R value
    double B = 0.0;
    double G = 0.0;
    double R = 0.0;
    
    B = MatBlur.at<Vec3b>(seedset.back())[0];
    G = MatBlur.at<Vec3b>(seedset.back())[1];
    R = MatBlur.at<Vec3b>(seedset.back())[2];
    
    //-----------------------------------------------------------------------
    while (!seedset.empty()) {
        
        Point oneseed = seedset.back(); //fetch one seed from seedvektor
        
        seedset.pop_back(); // delete this one seed from seedvektor
        
        //cout << "size of seedset: " << seedset.size() << "\n" << endl;
        
        MatLabel.at<uchar>(oneseed) = 255;
        Segment.at<Vec3b>(oneseed) = MatIn.at<Vec3b>(oneseed);
        
        B = (B+MatBlur.at<Vec3b>(oneseed)[0])/2.0;
        G = (G+MatBlur.at<Vec3b>(oneseed)[1])/2.0;
        R = (R+MatBlur.at<Vec3b>(oneseed)[2])/2.0;
        
        for(int iNum=0 ; iNum< rowofDIR ; iNum++)
        {
            Point nextseed;
            nextseed.x = oneseed.x + DIR[iNum][0];
            nextseed.y = oneseed.y + DIR[iNum][1];
            
            // check if it is boundry points
            //if(nextseed.x >0 && nextseed.x<(MatIn.cols-1) && nextseed.y>0 && nextseed.y<(MatIn.rows-1))
            //if ( nextseed.x  >0 && nextseed.x  < (MatIn.cols-1) && nextseed.y <(MatIn.rows-1) && nextseed.y >0 )

            if (nextseed.x < 0 || nextseed.y < 0 || nextseed.x > (MatIn.cols-2) || (nextseed.y > MatIn.rows-2))
                continue;
            
            if(MatLabel.at<uchar>(nextseed) != 255 )
            {
                
                //int d = differenceValue(oneseed, nextseed, DIR, rowofDIR, B, G, R);
                int d  = differenceValue(MatBlur, oneseed, nextseed, DIR, rowofDIR, B, G, R);
                
                if( iGrowJudge >= d ) // growing conditions 生长条件，自己调整
                {
                    seedset.push_back(nextseed);
                    seedtogether.push_back(nextseed);
                    MatGrownow.at<Vec3b>(nextseed) = MatIn.at<Vec3b>(nextseed);
                }
            }
        }

    }

    //regioncenter  = centerpoint(seedtogether);
    
    Mat element = getStructuringElement(MORPH_RECT, Size(2*elementSize+1,2*elementSize+1));
    dilate(Segment,Segment, element);
    dilate(Segment,Segment, element);
    erode(Segment,Segment, element);
    
    return Segment;
}

//--------------------------  difference value duction ---------
double Regiongrowing:: differenceValue(Mat MatIn, Point oneseed, Point nextseed, int DIR[][2], double rowofDIR, double B, double G, double R)
{
    // a: average value of all neighbour pixel to oneseed
    double B_oneseed = 0.0;
    double G_oneseed = 0.0;
    double R_oneseed = 0.0;
    for(int iNum=0; iNum< rowofDIR ; iNum++)
    {
        Point ANeighbour;
        ANeighbour.x = oneseed.x + DIR[iNum][0];
        ANeighbour.y = oneseed.y + DIR[iNum][1];
        B_oneseed = B_oneseed + MatIn.at<Vec3b>(ANeighbour)[0];
        G_oneseed = G_oneseed + MatIn.at<Vec3b>(ANeighbour)[1];
        R_oneseed = R_oneseed + MatIn.at<Vec3b>(ANeighbour)[2];
    }
    
    B_oneseed = B_oneseed/ (double)rowofDIR;
    G_oneseed = G_oneseed/ (double)rowofDIR;
    R_oneseed = R_oneseed/ (double)rowofDIR;
    //printf("BGR_ONESEED : %f, %f, %f \n", B_oneseed, G_oneseed, R_oneseed );
    
    //b : average value of all neighbour pixel to nextseed
    double B_nextseed = 0.0;
    double G_nextseed = 0.0;
    double R_nextseed = 0.0;
    for(int iNum=0; iNum< rowofDIR ; iNum++)
    {
        Point BNeighbour;
        BNeighbour.x = oneseed.x + DIR[iNum][0];
        BNeighbour.y = oneseed.y + DIR[iNum][1];
        B_nextseed = B_nextseed+ MatIn.at<Vec3b>(BNeighbour)[0];
        G_nextseed = G_nextseed+ MatIn.at<Vec3b>(BNeighbour)[1];
        R_nextseed = R_nextseed+ MatIn.at<Vec3b>(BNeighbour)[2];
    }
    
    B_nextseed = B_nextseed/ (double)rowofDIR;
    G_nextseed = G_nextseed/ (double)rowofDIR;
    R_nextseed = R_nextseed/ (double)rowofDIR;
    //printf("BGR_nextSEED : %f, %f, %f \n", B_nextseed, G_nextseed, R_nextseed );
    
    // 像素相减 x-y
    double B_diff1 = (B - MatIn.at<Vec3b>(nextseed)[0])*(B - MatIn.at<Vec3b>(nextseed)[0]);
    double G_diff1 = (G - MatIn.at<Vec3b>(nextseed)[1])*(G - MatIn.at<Vec3b>(nextseed)[1]);
    double R_diff1 = (R - MatIn.at<Vec3b>(nextseed)[2])*(R - MatIn.at<Vec3b>(nextseed)[2]);
    double d1 = B_diff1 + G_diff1 + R_diff1;

    // x-b
    double B_diff2 = (B_nextseed - MatIn.at<Vec3b>(oneseed)[0])*(B_nextseed - MatIn.at<Vec3b>(oneseed)[0]);
    double G_diff2 = (G_nextseed - MatIn.at<Vec3b>(oneseed)[1])*(G_nextseed - MatIn.at<Vec3b>(oneseed)[1]);
    double R_diff2 = (R_nextseed - MatIn.at<Vec3b>(oneseed)[2])*(R_nextseed - MatIn.at<Vec3b>(oneseed)[2]);
    double d2 = B_diff2 + G_diff2 + R_diff2;
    
    // y - a
    double B_diff3 = (B_oneseed - MatIn.at<Vec3b>(nextseed)[0])*(B_oneseed - MatIn.at<Vec3b>(nextseed)[0]);
    double G_diff3 = (G_oneseed - MatIn.at<Vec3b>(nextseed)[1])*(G_oneseed - MatIn.at<Vec3b>(nextseed)[1]);
    double R_diff3 = (R_oneseed - MatIn.at<Vec3b>(nextseed)[2])*(R_oneseed - MatIn.at<Vec3b>(nextseed)[2]);
    double d3 = B_diff3 + G_diff3 + R_diff3;
    //printf("d3 : %f \n", d3);
    
    double d = sqrt(d1 + d2 + d3);
    //printf("d : %f \n", d);
    return d;
}

//  --------- centerpoint of segment function
//Point Regiongrowing:: centerpoint(vector<Point> seedtogetherBackup){
//    
//    int x = 0;
//    int y = 0;
//    for(size_t i=0;i<seedtogetherBackup.size();i++){
//        x = x + seedtogetherBackup[i].x;
//        y = y + seedtogetherBackup[i].y;
//    }
//    
//    Point Center;
//    Center.x = x/seedtogetherBackup.size();
//    Center.y = y/seedtogetherBackup.size();
//    
//    return Center;
//}

Regiongrowing:: ~Regiongrowing(){
//    //cout << endl;
//    cout << "Klasse Regiongrowing " << regioncenter <<" wurde geloescht." << endl;
}


//------------------ 原counter 的 function
Mat Regiongrowing::FindCounter (Mat segment , Mat Frame, Vec3b color)
{
    Mat MatoutGray;
    
    cvtColor(segment,MatoutGray,CV_BGR2GRAY);
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
        drawContours(Frame, contours, static_cast<int>(i), color, 2, 8, hierarchy, 0);
        
        rectangle( Frame, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0 );// 绘制边框矩形
        rectanglewidth = boundRect[i].width;
        rectangleheight = boundRect[i].height;
        diagonallength = pixeldistance(boundRect[i].tl(),  boundRect[i].br());
        //      circle( FramemitCounter, center[i], (int)radius[i], color, 1, 8, 0 ); // draw the circle 绘制圆形边框
        
        getOrientation(contours[i], Frame);
    }
    
    return Frame;
}


void Regiongrowing::getOrientation(const vector<Point> &pts, Mat &img)
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
    
    // 注意下面公式里的参数
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


double Regiongrowing::drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
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

double Regiongrowing::pixeldistance(Point p1, Point p2)
{
    double a = p1.x - p2.x;
    double b = p1.y - p2.y;
    return (sqrt((a*a)+(b*b))); // a^2 not equal to a*a. a^2 has differnt meaning in Opencv
}




