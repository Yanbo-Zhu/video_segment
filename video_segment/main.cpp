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
#include "COUNTER.hpp"

using namespace cv;
using namespace std;

//-------global variable
int mode;
int Segmentnum;
//double  differencegrow;
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
//void DrawLine( Mat& img, Point pt );

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
   
    
    //vc.open( "/Users/yanbo/Desktop/source/Rotation_50m.mp4");
    vc.open( "/Users/yanbo/Desktop/source/80_10_descend_rotation.mp4");
    //vc.open( "/Users/yanbo/Desktop/source/5-70.mp4");
    
    
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
    
//-------------------------------------- VideoWriter function ----------------
    
    VideoWriter vw; //(filename, fourcc, fps, frameSize[, isColor])
    vw.open( "/Users/yanbo/Desktop/output1.avi", // 输出视频文件名
            CV_FOURCC('8', 'B', 'P', 'S'), //CV_FOURCC('S', 'V', 'Q', '3'), //(int)vc.get( CV_CAP_PROP_FOURCC ), // 也可设为CV_FOURCC_PROMPT，在运行时选取 //fourcc – 4-character code of codec used to compress the frames.
            (double)vc.get( CV_CAP_PROP_FPS ), // 视频帧率
            Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ),
                 (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
            true ); // 是否输出彩色视频
    
    if (!vw.isOpened())
    {
        cout << "Failed to write the video! \n" << endl;
        return 1;
    }
    
    
//-----------------------------finding first seed point---------------
    //Mat firstFrame;
    Mat firstFrame;
    vc.read(firstFrame);

    vc.set(CV_CAP_PROP_POS_FRAMES, 0);
    
    //imshow("first frame",firstFrame);
    //waitKey(10);
    
    //Mat MatInBackup = firstFrame.clone();
    
    Mat MatOut(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    Mat Matfinal (firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    //imshow("play video", firstFrame);  //显示当前帧
    
    //cout<<"plaese choose method. \n tap 1, choose seeds by logging the threshold value. \n tap 2, choose seeds by clicking in orignal image. \n tap 3, choose seeds by default position of points" <<endl;
    //cin >> mode;
    cout<<"Choose seeds by clicking in orignal image." <<endl;
    mode = 2;
    
    cout<<"How many initial segment do you want: " <<endl;
    cin >> Segmentnum;
    
    Initalseed s[Segmentnum];
    
    // settung color for diffent segments
    RNG rng(time(0));
    Vec3b color[Segmentnum];
    for( int i=0; i<Segmentnum; i++)
    {
        color[i] = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
    
    
    for( int i=0; i<Segmentnum; i++)
    {
        printf("\n********************Setting for object %d ***************\n", i+1);
        printf("Plaese select initial seeds \n");
        s[i].modechoose(mode, firstFrame);
        s[i].drawpoint(firstFrame, s[i].initialseedvektor, color[i]);
        s[i].data.resize(4);
        printf("\nPlease set the threshold value for region growing\n");
        cin >> s[i].differencegrow;
    }
    
//------------------------------- Start to apply Segmentation-method in Video
    

    
    bool stop(false);
    
    while(!stop)
    {
        Mat frame;//定义一个Mat变量，用于存储每一帧的图像
        
        int indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
        printf("\n----------------------------IndexFrame: %d -----------------------", indexFrame);

        bool bSuccess = vc.read(frame); // read a new frame from video
        //imshow("frame", frame);
        
        //若视频播放完成，退出循环
        if (frame.empty())
        {
            cout << "video play over  " <<endl;
            //waitKey(0);
            break;
        }

        if (!bSuccess) //if not success, break loop
        {
            cout << "ERROR: Cannot read a frame from video" << endl;
            break;
        }
        
        Mat frame_Blur;
        
        GaussianBlur(frame, frame_Blur, Size( 3, 3),0,0);
        
        Regiongrowing R[Segmentnum];
        Counter C[Segmentnum];
        //Regiongrowing R1;
        
        Matfinal = frame.clone();
        Mat FramewithCounter = frame.clone();
        
        for( int i=0; i<Segmentnum; i++)
        {
            printf("\n*********** Objekt %d Information ********************* \n", i+1);
            MatOut = R[i].RegionGrow(frame, frame_Blur , s[i].differencegrow, s[i].initialseedvektor);

            FramewithCounter = C[i].FindCounter(MatOut, FramewithCounter, color[i]);
           
            cout<< "threshold for RegionGrow : " << s[i].differencegrow << endl;
            
            if (indexFrame == 0){
                s[i].data[3].push_back(1.0);
                s[i].data[0].push_back(C[i].EWlong);
                s[i].data[1].push_back(C[i].EWshort);
                s[i].data[2].push_back(C[i].Ratio);
                s[i].initialseedvektor.clear();
                s[i].initialseedvektor.push_back(R[i].regioncenter);
            }
            
            else {
                
                //cout<<"s[i].data[0].back(): "<<s[i].data[0].back() <<"  C[i].EWlong: "<<  C[i].EWlong <<endl;
                double scale = ( (C[i].EWlong/s[i].data[0].back()) + (C[i].EWshort/s[i].data[1].back()) )/2 ;
                //cout<< "EWlong[indexFrame-1] " << EWlong[indexFrame-1] << " EWlong[indexFrame-2] "<< EWlong[indexFrame-2] << endl;
                printf("Scale (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, scale );
                
                
                // update the thereshlod value for region growing becasue scale varies largely
                double ScaleDifference = scale -  s[i].data[3].back();
                printf("ScaleDifference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, ScaleDifference );
                
                if (abs(ScaleDifference)> 0.2 ) {
                    printf("!!!!!!!!!!!Update the thereshlod value for region growing becasue scale varies largely \n");
                    
                    if ( ScaleDifference > 0.2)
                        s[i].differencegrow = s[i].differencegrow - 0.2;
                    
                    else
                        s[i].differencegrow = s[i].differencegrow + 0.2;
                    
                    printf("new differencegrow: %f", s[i].differencegrow);
                    //vc.set(CV_CAP_PROP_POS_FRAMES, indexFrame);
                }
               
                else{
                    s[i].data[3].push_back(scale);
                    s[i].data[0].push_back(C[i].EWlong);
                    s[i].data[1].push_back(C[i].EWshort);
                    s[i].data[2].push_back(C[i].Ratio);
                    s[i].initialseedvektor.clear();
                    s[i].initialseedvektor.push_back(R[i].regioncenter);
                }
            }
            
        
            
            
//            vector<double>::iterator iter;
//            vector<double>::iterator iter2;
//            //vector<double> v1 = s[i].data[0];
//            //cout<< "s[i].data[3].size(): " << s[i].data[3].size() << endl;
//            cout << "Scale vector = " ;
//            for (iter= s[i].data[3].begin(); iter != s[i].data[3].end(); iter++){
//                cout << *iter << " ";}
//            cout << endl;
//
//            cout << "Ratio vector = " ;
//            for (iter2= s[i].data[2].begin(); iter2 != s[i].data[2].end(); iter2++){
//                cout << *iter2 << " ";}
//            cout << endl;

            //imshow("segment counter", FramewithCounter);
            //waitKey(100);
            
            for(size_t j=0; j<R[i].seedtogether.size(); j++)
            {
                Matfinal.at<Vec3b>(R[i].seedtogether[j]) = color[i];
            }
        }

        imshow ("segment", Matfinal);
        imshow("segment counter", FramewithCounter);
        
        vw << FramewithCounter;
        //addWeighted(frame,1, result, 10 ,0, result);
        
        
        //waitKey(100);
        
        
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
        
        //if (keycode > 0 )
           // waitKey(0);
    }
    
    vc.release();
    vw.release();
    cout << "Video playing over" << endl;
    
    //system("pause");
    waitKey(0);
    
/****************** read the written video */
    
    VideoCapture vc2;
    vc2.open( "/Users/yanbo/Desktop/output1.avi");

    if (!vc2.isOpened())
    {
        cout << "Failed to open a video device or video file!\n" << endl;
        return 1;
    }

    //【2】循环显示每一帧
    while(1)
    {
        Mat frame;//定义一个Mat变量，用于存储每一帧的图像


        bool bSuccess = vc2.read(frame); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            cout << "ERROR: Cannot read a frame from output.avi" << endl;
            break;
        }

        imshow("play the written video", frame);  //显示当前帧
        waitKey(1);  //延时1ms
    }

    vc2.release();
    cout << "The written video plays over" << endl;
    
    
    return 0;
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




