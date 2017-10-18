//
//  main.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 21.09.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#include <iostream>
#include "stdio.h"
#include <time.h> //time_t time()  clock_t clock()

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>

#include "INITIALSEED.hpp"
#include "Regiongrowing.hpp"

using namespace cv;
using namespace std;

//-------global variable
int mode;
int Segmentnum;
double  differencegrow;
Point g_pt;
//vector<Point> seedvektor;
//Point regioncenter ;

//Mat FramewithCounter;
Mat firstFrame;
//Mat frame;
//Mat MatOut;
// Mat MatGrowCur;
//Mat MatGrowTemp;
//int iJudge;
vector<Mat> channels;
vector<Mat> channelsMatIn;
clock_t  clockBegin, clockEnd;
//vector<Point> seedtogether;
//-------

//#define WINDOW_NAME " point marking "

//----- global function
//void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawLine( Mat& img, Point pt );
Mat Counter (Mat MatOut , Mat currentFrame, Vec3b color);
void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const vector<Point> &, Mat&);
double pixeldistance(Point p1, Point p2);
//Mat RegionGrow(Mat MatIn, Mat MatGrowCur, double iGrowJudge, vector<Point> seedset);
//Mat RegionGrow(Mat MatIn, Mat MatBlur ,double iGrowJudge, vector<Point> seedset);
//double differenceValue(Mat MatIn, Point oneseed, Point nextseed, int DIR[][2], double rowofDIR, double B, double G, double R );
//Point centerpoint(vector<Point> seedtogetherBackup);
//void Countijudge(Mat Temp, int *pointerijudge);
//-------------------------------



int main( )
{
    //【1】读入视频
    VideoCapture vc;
    //vc.open( "/Users/zhu/Desktop/source/Rotation_descend_20_10m_small.mp4");
    //vc.open( "/Users/zhu/Desktop/source/ascend_5-50m.mp4");
    vc.open( "/Users/zhu/Desktop/source/Rotation_50m.mp4");
    
    
    if (!vc.isOpened())
    {
        cout << "Failed to open a video device or video file!\n" << endl;
        return 1;
    }
    
    int Fourcc = static_cast<int>(vc.get( CV_CAP_PROP_FOURCC ));
    int indexFrame  = vc.get(CV_CAP_PROP_POS_FRAMES);
    int FPS  = vc.get(CV_CAP_PROP_FPS);
    int FRAME_COUNT = vc.get(CV_CAP_PROP_FRAME_COUNT);
    int Width = vc.get(CV_CAP_PROP_FRAME_WIDTH);
    int Height = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
    printf("Fourcc: %d / indexFrame: %d / fps: %d / Frame_amount: %d / Width * Height : %d * %d \n", Fourcc ,indexFrame, FPS, FRAME_COUNT, Width, Height );
    //cout<< Fourcc<< endl;
    
    
//-----------------------------finding first seed point---------------
    //Mat firstFrame;
    Mat firstFrame;
    vc.read(firstFrame);
    imshow("first frame",firstFrame);
    waitKey(10);
    
    //Mat MatInBackup = firstFrame.clone();
    
    Mat MatOut(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    Mat Matfinal (firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    //imshow("play video", firstFrame);  //显示当前帧
    
    //cout<<"plaese choose method. \n tap 1, choose seeds by logging the threshold value. \n tap 2, choose seeds by clicking in orignal image. \n tap 3, choose seeds by default position of points" <<endl;
    //cin >> mode;
    cout<<"choose seeds by clicking in orignal image." <<endl;
    mode = 2;
    
    cout<<"How many initial segment do you want: " <<endl;
    cin >> Segmentnum;
    
    Initalseed s[Segmentnum];
    
    for( int i=0; i<Segmentnum; i++)
    {
       printf("plaese select initial seeds for object %d \n", i+1);
       s[i].modechoose(mode, firstFrame);

    }
    
    
//    cout<<"plaese select initial seeds for object 1" <<endl;
//    //Initalseed  s[1];
//    s[1].modechoose(mode, firstFrame);
//    //s1.drawpoint(firstFrame, s1.initialseedvektor);
//    
//    cout<<"plaese select initial seeds for object 2" <<endl;
//    Initalseed  s2;
//    s2.modechoose(mode, firstFrame);
    
    

//------------------------------- Start to apply Segmentation-method in Video
    
    cout<< "please set the threshold value for region growing"<<endl;
    
    cin >> differencegrow;
    
    // settung color for diffent segments
    RNG rng(time(0));
    Vec3b color[Segmentnum];
    for( int i=0; i<Segmentnum; i++)
    {
        color[i] = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
    
    
    bool stop(false);
    
    while(!stop)
    {
        Mat frame;//定义一个Mat变量，用于存储每一帧的图像

        bool bSuccess = vc.read(frame); // read a new frame from video
        
        
        //若视频播放完成，退出循环
        if (frame.empty())
        {
            //waitKey(0);
            break;
        }

        if (!bSuccess) //if not success, break loop
        {
            cout << "ERROR: Cannot read a frame from video" << endl;
            break;
        }
        
        int indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
        printf("----------------IndexFrame: %d --------------\n", indexFrame);
        
        //imshow("play video", frame);  //显示当前帧
        
        Mat frame_Blur;
        
        GaussianBlur(frame, frame_Blur, Size( 3, 3),0,0);
        
        //imshow("gaussian filtered image ", frame);
        //waitKey(10);
        
        //imshow("M.MatGrowCur", M.MatGrowCur);
        //waitKey(0);
        
        Regiongrowing R[Segmentnum];
        
        //Regiongrowing R1;
        
        Matfinal = frame.clone();
        Mat FramewithCounter = frame.clone();
        
        for( int i=0; i<Segmentnum; i++)
        {
            printf("Objekt %d Information: \n", i+1);
            MatOut = R[i].RegionGrow(frame, frame_Blur , differencegrow, s[i].initialseedvektor);
            
             s[i].initialseedvektor.clear();
             s[i].initialseedvektor.push_back(R[i].regioncenter);
            
            FramewithCounter = Counter(MatOut, frame, color[i]);
            
            for(size_t j=0; j<R[i].seedtogether.size(); j++)
            {
                Matfinal.at<Vec3b>(R[i].seedtogether[j]) = color[i];
            }
            
            
        }
        
        
        //addWeighted(frame,1, result, 10 ,0, result);
        
        imshow("segment counter", FramewithCounter);
        
        
//        Regiongrowing R2;
//        
//        MatOut = R2.RegionGrow(frame, frame_Blur , differencegrow, s2.initialseedvektor);
//        
//        s2.initialseedvektor.clear();
//        s2.initialseedvektor.push_back(R2.regioncenter);
//        
//        for(size_t i=0;i<R2.seedtogether.size();i++)
//        {
//            Matfinal.at<Vec3b>(R2.seedtogether[i]) = Vec3b(255,0,0);
//        }
        
        
//        // split to channel
//        Mat RedChannel;
//        Mat RedChannelMatIn;
//        
//        split(MatOut,channels);//分离色彩通道
//        
//        RedChannel = channels.at(2).clone();
//        
//        split(frame,channelsMatIn);//分离色彩通道
//        
//        RedChannelMatIn = channelsMatIn.at(2).clone();
//        
//        addWeighted(RedChannelMatIn,0.80, RedChannel, 20.0 ,0, RedChannelMatIn);
//        
//        RedChannelMatIn.copyTo(channelsMatIn.at(2));
//        
//        //  merge to channel
//        
//        merge (channelsMatIn, Matfinal);
//        
        
        //imshow("final image", Matfinal);
        
        
        //  define the stop-button and exit-button
        //waitKey(10);  //延时10ms
        int keycode = waitKey(1);
        if(keycode  == ' ')   //32是空格键的ASCII值
            waitKey(0);
        
        //if(keycode  == 27)  // 27 = ASCII ESC
            //stop=true;
        
        //R1.seedtogether.clear();
    }
    
    vc.release();
    cout << "Video playing over " << endl;
    
    waitKey(0);
    //system("pause");

    return 0;
}
// -------
Mat Counter (Mat MatOut , Mat currentFrame, Vec3b color)
{
    Mat MatoutGray;
    Mat FramemitCounter = currentFrame.clone();
    //Mat result (MatOut.size(),CV_8UC3, Scalar(0,0,0));
    //Mat result = zeros(MatOut.clone();
    
    // dilate MatOut
    int elementSize  = 2;
    Mat element = getStructuringElement(MORPH_RECT, Size(2*elementSize+1,2*elementSize+1));
    dilate(MatOut, MatOut, element);
    erode(MatOut, MatOut, element);
    //morphologyEx(MatOut, MatOut, MORPH_CLOSE, element)
    
    
    cvtColor(MatOut,MatoutGray,CV_BGR2GRAY);
    
    threshold(MatoutGray,MatoutGray,20,255,THRESH_BINARY);
    
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    findContours(MatoutGray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    for (size_t i = 0; i < contours.size(); ++i)
    {
        //cout<< "contours.size()" << contours.size() <<endl;
        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        // Ignore contours that are too small or too large
        if (area < 1e2 || 1e5 < area) continue;
        // Draw each contour only for visualisation purposes
        drawContours(FramemitCounter, contours, static_cast<int>(i), color, 2, 8, hierarchy, 0);
        //drawContours(result, contours, -1, Scalar(0, 0, 255), 1, 8, hierarchy, 0);
        getOrientation(contours[i], FramemitCounter);
    }
    
    return FramemitCounter;
}


double getOrientation(const vector<Point> &pts, Mat &img)
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
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                       static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    
    cout<< "center of the object: Row " << cntr.y << " Column: " << cntr.x << endl;
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    
   static vector<double> pixelabstand;
    drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1); // Green line long axis
    drawAxis(img, cntr, p2, Scalar(255, 255, 0), 3); // light blue line . short axis
    
    double Ratio;
    
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    cout << "angle: " << angle << endl;
    return angle;
}


void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle;
    double hypotenuse; //直角三角形的斜边
    double twopointdistance;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
        //double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
        //cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    twopointdistance = pixeldistance(p, q);
    
    pixelabstand.push_back(twopointdistance);
    
    line(img, p, q, colour, 1, CV_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
}

double pixeldistance(Point p1, Point p2)
{
    //cout<< p1.x <<" "<< pv[1].x <<" "<< pv[0].y<<" "<<pv[1].y<<endl;
    double a = p1.x - p2.x;
    double b = p1.y - p2.y;
    return (sqrt((a*a)+(b*b))); // a^2 not equal to a*a. a^2 has differnt meaning in Opencv
    
}



//-------------------------------------- VideoWriter function ----------------
//waitKey(0);

//    VideoWriter vw; //(filename, fourcc, fps, frameSize[, isColor])
//    vw.open( "./output1.avi", // 输出视频文件名
//            CV_FOURCC('8', 'B', 'P', 'S'), //CV_FOURCC('S', 'V', 'Q', '3'), //(int)vc.get( CV_CAP_PROP_FOURCC ), // 也可设为CV_FOURCC_PROMPT，在运行时选取 //fourcc – 4-character code of codec used to compress the frames.
//            (double)vc.get( CV_CAP_PROP_FPS ), // 视频帧率
//            Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ),
//                 (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
//            true ); // 是否输出彩色视频
//    
//    //vw << frame;
//    //vw.release(); // vokler 加入的 释放空间 消除buffer 影响
//    
//    /** 如果成功打开输出视频文件 */
//    
//    if ( vw.isOpened() )
//    {
//        while ( 1 ) //reclyce every frame
//        {
//            /** 读取当前视频帧 */
//            Mat frame;
//            
//            
//            bool bSuccess = vc.read(frame); // read a new frame from video
//            
//            if (!bSuccess) //if not success, break loop
//            {
//                cout << "ERROR+ZHU: Cannot read a frame from video file" << endl;
//                break;
//            }
//            
//            //vc >> frame;
//            
//            /** 若视频读取完毕，跳出循环 */
//            //                    if ( frame.empty() )
//            //                    {
//            //                        break;
//            //                    }
//            
//            /** 将视频写入文件 */
//            vw << frame;
//            //vw.write(frame);
//            
//            imshow("MyVideo", frame);
//            waitKey(1);
//            
//        }
//    }
//    
//    
//    /** 手动释放视频捕获资源 */
//    vc.release();
//    vw.release();
//    
//    
//    destroyAllWindows();
//    
//    
//    /** read the written video */
//    //【1】读入视频
//    VideoCapture vc2;
//    vc2.open( "/Users/zhu/Desktop/opencv_project/video_process_basic/build/Debug/output1.avi");
//    
//    if (!vc2.isOpened())
//    {
//        cout << "Failed to open a video device or video file!\n" << endl;
//        return 1;
//    }
//    
//    //【2】循环显示每一帧
//    while(1)
//    {
//        Mat frame;//定义一个Mat变量，用于存储每一帧的图像
//        
//        
//        bool bSuccess = vc2.read(frame); // read a new frame from video
//        
//        if (!bSuccess) //if not success, break loop
//        {
//            cout << "ERROR+ZHU: Cannot read a frame from output.avi" << endl;
//            break;
//        }
//        
//        imshow("play video", frame);  //显示当前帧
//        waitKey(1);  //延时1ms
//    }
//    
//    
//    vc2.release();
//    cout << "Finished writing" << endl;
    
    //return 0;



///----------  RegionGrowing function------------------------------

//Mat RegionGrow(Mat MatIn, Mat MatBlur , double iGrowJudge, vector<Point> seedset) //iGrowPoint: seeds 为种子点的判断条件，iGrowJudge: growing condition 为生长条件
//{
//    
//    //Mat MatGrowOld(MatIn.size(),CV_8UC3,Scalar(0,0,0));
//    //Mat MatGrownext(MatIn.size(),CV_8UC3,Scalar(0,0,0));
//    //Mat MatGrowTemp(MatIn.size(),CV_8UC3,Scalar(0,0,0));
//    //Mat MatGrownow(MatIn.size(),CV_8UC3,Scalar(0,0,0));
//    Mat Segment(MatIn.size(),CV_8UC3,Scalar(0,0,0));
//    Mat MatLabel(MatIn.size(),CV_8UC1,Scalar(0));
//    
//    // intialize MatGrownow
//    Mat MatGrownow(MatIn.size(),CV_8UC3,Scalar(0,0,0));
//    
//    for(size_t i=0;i<seedset.size();i++)
//    {
//        //cout << initialseedvektor[i] <<endl;
//        MatGrownow.at<Vec3b>(seedset[i]) = MatIn.at<Vec3b>(seedset[i]);
//        //seedtogether.push_back(seedset[i])
//    }
//    
//    
//    seedtogether = seedset;
//    
//    //生长方向顺序数据
//    int DIR[8][2]={{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
//    //int DIR[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
//    double rowofDIR= sizeof(DIR)/sizeof(DIR[0]);
//    //cout <<"rowofDIR: " << rowofDIR << "\n" << endl;
//    
//    // calculate the initial B G R value
//    double B = 0.0;
//    double G = 0.0;
//    double R = 0.0;
//    
//    
//    B = MatBlur.at<Vec3b>(seedset.back())[0];
//    G = MatBlur.at<Vec3b>(seedset.back())[1];
//    R = MatBlur.at<Vec3b>(seedset.back())[2];
//    
//    //-----------------------------------------------------------------------
//        while (!seedset.empty()) {
//            
//            Point oneseed = seedset.back(); //fetch one seed from seedvektor
//            
//            seedset.pop_back(); // delete this one seed from seedvektor
//            
//            //cout << "size of seedset: " << seedset.size() << "\n" << endl;
//            
//            MatLabel.at<uchar>(oneseed) = 255;
//            
//            Segment.at<Vec3b>(oneseed) = MatIn.at<Vec3b>(oneseed);
//            
//            B = (B+MatBlur.at<Vec3b>(oneseed)[0])/2.0;
//            G = (G+MatBlur.at<Vec3b>(oneseed)[1])/2.0;
//            R = (R+MatBlur.at<Vec3b>(oneseed)[2])/2.0;
//            
//            
//            for(int iNum=0 ; iNum< rowofDIR ; iNum++)
//            {
//                Point nextseed;
//                nextseed.x = oneseed.x + DIR[iNum][0];
//                nextseed.y = oneseed.y + DIR[iNum][1];
//                
//                // check if it is boundry points
//                
//                //if(nextseed.x >0 && nextseed.x<(MatIn.cols-1) && nextseed.y>0 && nextseed.y<(MatIn.rows-1))
//                
//                //if ( nextseed.x  >0 && nextseed.x  < (MatIn.cols-1) && nextseed.y <(MatIn.rows-1) && nextseed.y >0 )
//                //{
//                //cout << "inloop \n" << endl;
//                if (nextseed.x < 0 || nextseed.y < 0 || nextseed.x > (MatIn.cols-1) || (nextseed.y > MatIn.rows-1))
//                    continue;
//                
//                if(MatLabel.at<uchar>(nextseed) != 255 )
//                {
//                    
//                    //int d = differenceValue(oneseed, nextseed, DIR, rowofDIR, B, G, R);
//                    int d  = differenceValue(MatBlur, oneseed, nextseed, DIR, rowofDIR, B, G, R);
//                    
//                    if( iGrowJudge >= d ) // growing conditions 生长条件，自己调整
//                    {
//                        seedset.push_back(nextseed);
//                        seedtogether.push_back(nextseed);
//                        MatGrownow.at<Vec3b>(nextseed) = MatIn.at<Vec3b>(nextseed);
//                    }
//                }
//            }
//            
//            //imshow("MatGrownow", MatGrownow);
//            //waitKey(1);
//        }
//    //----------------------------------------------------------------------------
//    
//    //cout << "seedtogether.size:" << seedtogether.size() << endl;
//    regioncenter  = centerpoint(seedtogether);
//    //cout<<"regioncenter: " << regioncenter <<endl;
//    //seedtogether.clear();
//    
//    return Segment;
//}
//
//
////--------------------------  difference value duction ---------
//double differenceValue(Mat MatIn, Point oneseed, Point nextseed, int DIR[][2], double rowofDIR, double B, double G, double R)
//{
//    // a: average value of all neighbour pixel to oneseed
//    double B_oneseed = 0.0;
//    double G_oneseed = 0.0;
//    double R_oneseed = 0.0;
//    for(int iNum=0; iNum< rowofDIR ; iNum++)
//    {
//        Point ANeighbour;
//        ANeighbour.x = oneseed.x + DIR[iNum][0];
//        ANeighbour.y = oneseed.y + DIR[iNum][1];
//        B_oneseed = B_oneseed + MatIn.at<Vec3b>(ANeighbour)[0];
//        G_oneseed = G_oneseed + MatIn.at<Vec3b>(ANeighbour)[1];
//        R_oneseed = R_oneseed + MatIn.at<Vec3b>(ANeighbour)[2];
//        //printf("BGR_ONESEED : %f, %f, %f \n", B_oneseed, G_oneseed, R_oneseed );
//        
//    }
//    
//    B_oneseed = B_oneseed/ (double)rowofDIR;
//    G_oneseed = G_oneseed/ (double)rowofDIR;
//    R_oneseed = R_oneseed/ (double)rowofDIR;
//    //printf("BGR_ONESEED : %f, %f, %f \n", B_oneseed, G_oneseed, R_oneseed );
//    
//    
//    //b : average value of all neighbour pixel to nextseed
//    double B_nextseed = 0.0;
//    double G_nextseed = 0.0;
//    double R_nextseed = 0.0;
//    for(int iNum=0; iNum< rowofDIR ; iNum++)
//    {
//        Point BNeighbour;
//        BNeighbour.x = oneseed.x + DIR[iNum][0];
//        BNeighbour.y = oneseed.y + DIR[iNum][1];
//        B_nextseed = B_nextseed+ MatIn.at<Vec3b>(BNeighbour)[0];
//        G_nextseed = G_nextseed+ MatIn.at<Vec3b>(BNeighbour)[1];
//        R_nextseed = R_nextseed+ MatIn.at<Vec3b>(BNeighbour)[2];
//    }
//    
//    B_nextseed = B_nextseed/ (double)rowofDIR;
//    G_nextseed = G_nextseed/ (double)rowofDIR;
//    R_nextseed = R_nextseed/ (double)rowofDIR;
//    //printf("BGR_nextSEED : %f, %f, %f \n", B_nextseed, G_nextseed, R_nextseed );
//    
//    // 像素相减 x-y
//    double B_diff1 = (B - MatIn.at<Vec3b>(nextseed)[0])*(B - MatIn.at<Vec3b>(nextseed)[0]);
//    
//    double G_diff1 = (G - MatIn.at<Vec3b>(nextseed)[1])*(G - MatIn.at<Vec3b>(nextseed)[1]);
//    
//    double R_diff1 = (R - MatIn.at<Vec3b>(nextseed)[2])*(R - MatIn.at<Vec3b>(nextseed)[2]);
//    
//    double d1 = B_diff1 + G_diff1 + R_diff1;
//    //printf("d1 : %f \n", d1);
//    
//    // x-b
//    double B_diff2 = (B_nextseed - MatIn.at<Vec3b>(oneseed)[0])*(B_nextseed - MatIn.at<Vec3b>(oneseed)[0]);
//    
//    double G_diff2 = (G_nextseed - MatIn.at<Vec3b>(oneseed)[1])*(G_nextseed - MatIn.at<Vec3b>(oneseed)[1]);
//    
//    double R_diff2 = (R_nextseed - MatIn.at<Vec3b>(oneseed)[2])*(R_nextseed - MatIn.at<Vec3b>(oneseed)[2]);
//    
//    double d2 = B_diff2 + G_diff2 + R_diff2;
//    //printf("d2 : %f \n", d2);
//    
//    // y - a
//    double B_diff3 = (B_oneseed - MatIn.at<Vec3b>(nextseed)[0])*(B_oneseed - MatIn.at<Vec3b>(nextseed)[0]);
//    
//    double G_diff3 = (G_oneseed - MatIn.at<Vec3b>(nextseed)[1])*(G_oneseed - MatIn.at<Vec3b>(nextseed)[1]);
//    
//    double R_diff3 = (R_oneseed - MatIn.at<Vec3b>(nextseed)[2])*(R_oneseed - MatIn.at<Vec3b>(nextseed)[2]);
//    
//    double d3 = B_diff3 + G_diff3 + R_diff3;
//    //printf("d3 : %f \n", d3);
//    
//    double d = sqrt(d1 + d2 + d3);
//    //printf("d : %f \n", d);
//    return d;
//    
//}
//
////  ------------------- centerpoint of segment function
//Point centerpoint(vector<Point> seedtogetherBackup){
//    
//    int x = 0;
//    int y = 0;
//    for(size_t i=0;i<seedtogetherBackup.size();i++)
//    {
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



///--------------  on_MouseHandle   funciton----------
//void on_MouseHandle(int event, int x, int y, int flags, void* param)
//{
//    
//    Mat & image = *(Mat*) param;
//    //Mat *im = reinterpret_cast<Mat*>(param);
//    
//    //mouse ist not in window 处理鼠标不在窗口中的情况
//    if( x < 0 || x >= image.cols || y < 0 || y >= image.rows ){
//        return;
//    }
//    
//    if (event == EVENT_LBUTTONDOWN)
//        
//    {
//        
//        //g_pt = Point(x, y);
//        
//        Scalar colorvalue = image.at<Vec3b>(Point(x, y));
//        //Vec3b colorvalue = image.at<Vec3b>(Point(x, y));
//        cout<<"at("<<x<<","<<y<<") pixel value: " << colorvalue <<endl;
//        
//        //调用函数进行绘制
//        DrawLine( image, Point(x, y));//画线
//        
//        seedvektor.push_back(Point(x, y));
//        
//    }
//}




