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

using namespace cv;
using namespace std;

//-------global variable
int mode;
double thresholdvalue, differencegrow;
Point g_pt;
//vector<Point> seedvektor;
Point regioncenter ;

Mat firstFrame;
//Mat frame;
//Mat MatOut;
// Mat MatGrowCur;
//Mat MatGrowTemp;
//int iJudge;
vector<Mat> channels;
vector<Mat> channelsMatIn;
clock_t  clockBegin, clockEnd;
vector<Point> seedtogether;
//-------

//#define WINDOW_NAME " point marking "

//----- global function
//void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawLine( Mat& img, Point pt );
//Mat RegionGrow(Mat MatIn, Mat MatGrowCur, double iGrowJudge, vector<Point> seedset);
Mat RegionGrow(Mat MatIn, Mat MatBlur ,double iGrowJudge, vector<Point> seedset);
double differenceValue(Mat MatIn, Point oneseed, Point nextseed, int DIR[][2], double rowofDIR, double B, double G, double R );
Point centerpoint(vector<Point> seedtogetherBackup);
//void Countijudge(Mat Temp, int *pointerijudge);
//-------------------------------



int main( )
{
    //【1】读入视频
    VideoCapture vc;
    //vc.open( "/Users/zhu/Desktop/source/Rotation_descend_20_10m_small.mp4");
    vc.open( "/Users/zhu/Desktop/source/ascend_5-50m.mp4");
    //vc.open( "/Users/zhu/Desktop/source/Rotation_50m.mp4");
    
    
    if (!vc.isOpened())
    {
        cout << "Failed to open a video device or video file!\n" << endl;
        return 1;
    }
    
    int Fourcc = static_cast<int>(vc.get( CV_CAP_PROP_FOURCC ));
    int indexFrame  = vc.get(CV_CAP_PROP_POS_FRAMES);
    int FPS  = vc.get(CV_CAP_PROP_FPS);
    int FRAME_COUNT = vc.get(CV_CAP_PROP_FRAME_COUNT);
    
    printf("Fourcc: %d / indexFrame: %d / fps: %d / Frame_amount: %d \n", Fourcc ,indexFrame, FPS, FRAME_COUNT);
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
    
    cout<<"plaese choose method. \n tap 1, choose seeds by logging the threshold value. \n tap 2, choose seeds by clicking in orignal image. \n tap 3, choose seeds by default position of points" <<endl;
    cin >> mode;
    
    Initalseed M;
    M.modechoose(mode, firstFrame);
    M.drawpoint(firstFrame, M.initialseedvektor);
    

//------------------------------- Start to apply Segmentation-method in Video
    
    cout<< "please set the difference value for region growing"<<endl;
    
    cin >> differencegrow;
    
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
        printf("indexFrame: %d \n", indexFrame);
        
        //imshow("play video", frame);  //显示当前帧
        
        Mat frame_Blur;
        
        GaussianBlur(frame, frame_Blur, Size( 3, 3),0,0);
        
        //imshow("gaussian filtered image ", frame);
        //waitKey(10);
        
        //imshow("M.MatGrowCur", M.MatGrowCur);
        //waitKey(0);
        
        MatOut = RegionGrow(frame, frame_Blur , differencegrow, M.initialseedvektor);
        
        
         M.initialseedvektor.clear();
         M.initialseedvektor.push_back(regioncenter);
        
        imshow("Segments image", MatOut);
        
        
        //initialize  MatGrowCur (image for orinigal seeds before region growing)
//        for(size_t i=0;i<seedvektor.size();i++)
//        {
//            MatGrowCur.at<Vec3b>(seedvektor[i]) = firstFrame.at<Vec3b>(seedvektor[i]);
//        }
        
        Matfinal = frame.clone();
        
        for(size_t i=0;i<seedtogether.size();i++)
        {
            Matfinal.at<Vec3b>(seedtogether[i]) = Vec3b(0,0,255);
        }
        
        
        
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
        
        imshow("final image", Matfinal);
        
        
        //  define the stop-button and exit-button
        //waitKey(10);  //延时10ms
        int keycode = waitKey(1);
        if(keycode  == ' ')   //32是空格键的ASCII值
            waitKey(0);
        
        //if(keycode  == 27)  // 27 = ASCII ESC
            //stop=true;
    }
    
    vc.release();
    cout << "Video playing over " << endl;
    
    
    waitKey(0);
    //system("pause");

    return 0;
    

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
}


///----------  RegionGrowing function------------------------------

Mat RegionGrow(Mat MatIn, Mat MatBlur , double iGrowJudge, vector<Point> seedset) //iGrowPoint: seeds 为种子点的判断条件，iGrowJudge: growing condition 为生长条件
{
    
    //Mat MatGrowOld(MatIn.size(),CV_8UC3,Scalar(0,0,0));
    //Mat MatGrownext(MatIn.size(),CV_8UC3,Scalar(0,0,0));
    //Mat MatGrowTemp(MatIn.size(),CV_8UC3,Scalar(0,0,0));
    //Mat MatGrownow(MatIn.size(),CV_8UC3,Scalar(0,0,0));
    Mat Segment(MatIn.size(),CV_8UC3,Scalar(0,0,0));
    Mat MatLabel(MatIn.size(),CV_8UC1,Scalar(0));
    
    // intialize MatGrownow
    Mat MatGrownow(MatIn.size(),CV_8UC3,Scalar(0,0,0));
    
    for(size_t i=0;i<seedset.size();i++)
    {
        //cout << initialseedvektor[i] <<endl;
        MatGrownow.at<Vec3b>(seedset[i]) = MatIn.at<Vec3b>(seedset[i]);
    }
    
    seedtogether.clear();
    seedtogether = seedset;
    
    //生长方向顺序数据
    int DIR[8][2]={{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
    //int DIR[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    double rowofDIR= sizeof(DIR)/sizeof(DIR[0]);
    //cout <<"rowofDIR: " << rowofDIR << "\n" << endl;
    
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
                //{
                //cout << "inloop \n" << endl;
                if (nextseed.x < 0 || nextseed.y < 0 || nextseed.x > (MatIn.cols-1) || (nextseed.y > MatIn.rows-1))
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
            
            //imshow("MatGrownow", MatGrownow);
            //waitKey(1);
        }
    //----------------------------------------------------------------------------
    
    //cout << "seedtogether.size:" << seedtogether.size() << endl;
    regioncenter  = centerpoint(seedtogether);
    //cout<<"regioncenter: " << regioncenter <<endl;
    //seedtogether.clear();
    
    return Segment;
}


//--------------------------  difference value duction ---------
double differenceValue(Mat MatIn, Point oneseed, Point nextseed, int DIR[][2], double rowofDIR, double B, double G, double R)
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
        //printf("BGR_ONESEED : %f, %f, %f \n", B_oneseed, G_oneseed, R_oneseed );
        
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
    //printf("d1 : %f \n", d1);
    
    // x-b
    double B_diff2 = (B_nextseed - MatIn.at<Vec3b>(oneseed)[0])*(B_nextseed - MatIn.at<Vec3b>(oneseed)[0]);
    
    double G_diff2 = (G_nextseed - MatIn.at<Vec3b>(oneseed)[1])*(G_nextseed - MatIn.at<Vec3b>(oneseed)[1]);
    
    double R_diff2 = (R_nextseed - MatIn.at<Vec3b>(oneseed)[2])*(R_nextseed - MatIn.at<Vec3b>(oneseed)[2]);
    
    double d2 = B_diff2 + G_diff2 + R_diff2;
    //printf("d2 : %f \n", d2);
    
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

//  ------------------- centerpoint of segment function
Point centerpoint(vector<Point> seedtogetherBackup){
    
    int x = 0;
    int y = 0;
    for(size_t i=0;i<seedtogetherBackup.size();i++)
    {
        x = x + seedtogetherBackup[i].x;
        y = y + seedtogetherBackup[i].y;
    }
    
    Point Center;
    Center.x = x/seedtogetherBackup.size();
    Center.y = y/seedtogetherBackup.size();
    
    return Center;
}



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




