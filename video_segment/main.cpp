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
#include <math.h>
#include <float.h>
#include "fstream"
#include <string>
#include <list>
#include <vector>
#include <limits>  // numeric_limits<double>::epsilon(): 2.22045e-016
#include <numeric>

//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include "opencv2/features2d/features2d.hpp"

#include "INITIALSEED.hpp"
#include "Regiongrowing.hpp"
#include "Opticalflow.hpp"
//#include "COUNTER.hpp"
#include "Function.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

//-------global variable

int FPSfacotr = 7 ;

// region growing
int considerNum = 10;
int multipleScaleDiff = 10;

double Areadifferencefactor = 0.2 ;
double confidenceintervalfactor = 1.04 ;
int Loopiterationmax = 10;

Size kernelsize(3,3);
#define EPSILON 1e-13   // arrcuracy value
vector<Point> relationpointvektor;
clock_t  clockBegin, clockEnd;
//#define windowNameRG String(XXX) //播放窗口名称

// Opical flow
vector<vector<Point2f> > points(2);    // point0为特征点的原来位置，point1为特征点的新位置  point0:the points which his flow needs to be found.  point1: calculated new positions of input features in the second image.
vector<Point2f> initial;    // 初始化跟踪点的位置

//----- global function
Mat putStats(vector<string> stats, Mat frame,Vec3b color,Point* origin, char word);
double averagevalue(int num, vector<double> array);
double averagedifference(int num, vector<double> array);
void on_MouseHandle(int event, int x, int y, int flags, void* param);
double pixeldistance(Mat& img, vector<Point> pv);

void Featurematch(Mat preframe , Mat nextframe, vector<Point2f>& obj_last, vector<Point2f>& obj_next);
//double decomposematrix(Mat H);


//setting for trackbar
char const *windowNameRG = "Region growing ";
char const *windowNameOP = "Optical flow";

char const *trackBarName="Frame index";    //trackbar控制条名称
double totalFrame=1.0;     //视频总帧数
//double currentFrame=1.0;    //当前播放帧
int trackbarValue=1;    //trackbar控制量
int trackbarMax=200;   //trackbar控制条最大值
double frameRate = 0.1;  //视频帧率

int controlRate = 1;
VideoCapture vc;    //声明视频对象   !!!!!!!!!!!!!!!!!!!!!!!此处重要

void TrackBarFunc(int ,void(*))
{
    controlRate=((double)trackbarValue/trackbarMax)*totalFrame; //trackbar控制条对视频播放进度的控制
    vc.set(CV_CAP_PROP_POS_FRAMES,controlRate);   //设置当前播放帧
}


int main( )
{
    
//--------------some default seed points when i choose 2 during modechoose
// Point (x,y )  x= column  y = row
    vector<vector<Point>> defaultseed(4);
    defaultseed[0].push_back(Point(515,376)); // black roof 有缺陷 在中间  threshold 5
    //defaultseed[0].push_back(Point(781,379)); // grass  threshold 4
    defaultseed[1].push_back(Point(215,240)); // light blue roof left
    defaultseed[2].push_back(Point(530,234)); // white boot
    defaultseed[3].push_back(Point(491,356)); // black roof bottem
    double defaultThreshold[] = {6, 11, 12 ,8} ;
    
///--------- VideoCapture ----------------
    
    //VideoCapture vc;
    //vc.open( "/Users/yanbo/Desktop/source/Rotation_50m.mp4");
    char path[] = "/Users/yanbo/Desktop/source/" ;
    string videofilename = "70_20_descend";
    //string videofilename = "skew_descend";
    //string videofilename = "swiss_rotation";
    string videoInputpath;
    videoInputpath.assign(path);
    videoInputpath.append(videofilename);
    videoInputpath.append(".mp4");

    vc.open(videoInputpath);
    
    if (!vc.isOpened())
    {
        cout << "Failed to open a video device or video file!\n" << endl;
        return 1;
    }
    
    int Fourcc = static_cast<int>(vc.get( CV_CAP_PROP_FOURCC ));
    //int IndexFrame  = vc.get(CV_CAP_PROP_POS_FRAMES);
    int FPS  = vc.get(CV_CAP_PROP_FPS);
    int FRAME_COUNT = vc.get(CV_CAP_PROP_FRAME_COUNT);
    int Width = vc.get(CV_CAP_PROP_FRAME_WIDTH);
    int Height = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
    printf("Fourcc: %d / fps: %d / Total Frame: %d / Width*Height : %d*%d = %d / Width*Height/300 : %d / Width*Height/40: %d \n", Fourcc, FPS, FRAME_COUNT, Width, Height, Width*Height, Width*Height/300, Width*Height/50);
    
//    string XXX = "Segment counter ";
//    char *windowNameRG = new char[50];
//    strcat(windowNameRG,XXX.c_str());
//    strcat(windowNameRG,videofilename.c_str());
    
//-------------------------------------- VideoWriter Region growing ----------------

    char *savePath = new char[50];
    strcpy(savePath,path);
    strcat(savePath,"output/");
    strcat(savePath,videofilename.c_str());
    strcat(savePath,"_output_1");     // savePath::  /Users/yanbo/Desktop/source/output/70_20_descend_output
    cout<< "savePath: " << savePath <<endl;
    cout<<endl;
    
    char savePathvideoRG[50] ;
    strcpy(savePathvideoRG,savePath);
    strcat(savePathvideoRG, "_RG");
    strcat(savePathvideoRG, ".mov");
    remove(savePathvideoRG);
    
    //char const *savePathvideo = "/Users/yanbo/Desktop/source/output/output2.txt";
//    if(remove(savePathvideo)==0)
//    {
//        cout<<"Old output video delete successful"<<endl;
//    }
//    else
//    {
//        cout<<"Old output video delete failed"<<endl;
//    }
    
    VideoWriter vwRG; //(filename, fourcc, fps, frameSize[, isColor])
    
    vwRG.open( savePathvideoRG, // 输出视频文件名
            (int)vc.get( CV_CAP_PROP_FOURCC ),//CV_FOURCC('S', 'V', 'Q', '3'), // //CV_FOURCC('8', 'B', 'P', 'S'), // 也可设为CV_FOURCC_PROMPT，在运行时选取 //fourcc – 4-character code of codec used to compress the frames.
            (double)(vc.get( CV_CAP_PROP_FPS )/FPSfacotr), // 视频帧率
            Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ),
                 (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
            true ); // 是否输出彩色视频
    
    if (!vwRG.isOpened())
    {
        cout << "Failed to write the video! \n" << endl;
        return 1;
    }
    
//---------------- Three method Videowriter
    
    char savePathvideo3m[50] ;
    strcpy(savePathvideo3m,savePath);
    strcat(savePathvideo3m,"_threemethod");
    strcat(savePathvideo3m, ".mov");
    remove(savePathvideo3m);
    
    //char const *savePathvideo = "/Users/yanbo/Desktop/source/output/output2.txt";
    VideoWriter vw3method; //(filename, fourcc, fps, frameSize[, isColor])
    vw3method.open( savePathvideo3m, // 输出视频文件名
            (int)vc.get( CV_CAP_PROP_FOURCC ),//CV_FOURCC('S', 'V', 'Q', '3'), // //CV_FOURCC('8', 'B', 'P', 'S'), // 也可设为CV_FOURCC_PROMPT，在运行时选取 //fourcc – 4-character code of codec used to compress the frames.
            (double)(vc.get( CV_CAP_PROP_FPS )/FPSfacotr), // 视频帧率
            Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ),
                 (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
            true ); // 是否输出彩色视频
    
    if (!vw3method.isOpened())
    {
        cout << "Failed to write the video! \n" << endl;
        return 1;
    }

//---------------- optical tracking path Videowriter
    
    char savePathvideoOP[50] ;
    strcpy(savePathvideoOP,savePath);
    strcat(savePathvideoOP,"_OP");
    strcat(savePathvideoOP, ".mov");
    remove(savePathvideoOP);
    
    //char const *savePathvideo = "/Users/yanbo/Desktop/source/output/output2.txt";
    VideoWriter vwop; //(filename, fourcc, fps, frameSize[, isColor])
    vwop.open( savePathvideoOP, // 输出视频文件名
                   (int)vc.get( CV_CAP_PROP_FOURCC ),//CV_FOURCC('S', 'V', 'Q', '3'), // //CV_FOURCC('8', 'B', 'P', 'S'), // 也可设为CV_FOURCC_PROMPT，在运行时选取 //fourcc – 4-character code of codec used to compress the frames.
                   (double)(vc.get( CV_CAP_PROP_FPS )/FPSfacotr), // 视频帧率
                   Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ),
                        (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
                   true ); // 是否输出彩色视频
    
    if (!vwop.isOpened())
    {
        cout << "Failed to write the video! \n" << endl;
        return 1;
    }
    
//------------------------------------------------------
    Mat firstFrame;
    vc.read(firstFrame);
    Mat preFrame = firstFrame.clone();
    //Mat preallsegment(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    //vc.set(CV_CAP_PROP_POS_FRAMES, 1);
    
//--------------- Setting the relation between real distrance and pixel distance
    
    Mat firstframeBackup; //= firstFrame.clone();
    firstFrame.copyTo(firstframeBackup);

    cout<<"Setting the relation between real distrance and pixel distance"<<endl;
    cout<<" Please mark two point on the image"<<endl;
    //char const *Pixel_realtion_window= "Pixel-distance relation";
    #define Pixel_realtion_window "Pixel-distance relation"
    
//    namedWindow( Pixel_realtion_window );
//
//    setMouseCallback(Pixel_realtion_window,on_MouseHandle,(void*)&firstframeBackup);
//
//    while(true)
//    {
//        imshow( Pixel_realtion_window, firstframeBackup );
//        if( waitKey( 10 ) == 13 ) break;//按下enter键，程序退出
//    }
//
//    //cout<< "relationpoint vektor size= " << relationpointvektor.size() <<endl;
//    if (relationpointvektor.size() != 2){
//        cout<< "\n!!!!!!!You did not mark 2 points. Porgramm breaks"  <<endl;
//        return 0;
//    }

    relationpointvektor.push_back(Point(272,437));
    relationpointvektor.push_back(Point(329,335));
//    relationpointvektor.push_back(Point(584,222));
//    relationpointvektor.push_back(Point(497,268));

    double pixeld = pixeldistance(firstframeBackup, relationpointvektor);
    cout<< "Pixeldistance: " << pixeld <<endl;

    imshow( Pixel_realtion_window , firstframeBackup );
    waitKey(1);
    //destroyWindow(Pixel_realtion_window);
    //destroyAllWindows();
 
    cout<< " Please input the real distance (m) for this line"<<endl;
    double distance;
    //cin >> distance;

    distance = 100;
    cout<<  distance << endl;
    
    double initialpixelrelation;
    initialpixelrelation = distance/ pixeld;
    cout<< " The relation: " << initialpixelrelation << " m/pixel \n" <<endl;
    

//-----------------------------finding first seed point---------------
    
    cout<<"How many initial segment do you want: " <<endl;
    int Segmentinitialnum;
    cin >> Segmentinitialnum;
    //Segmentnum  = 1;
    
    Initialseed s[Segmentinitialnum];
    vector<Initialseed>  vectorS;
    //deque<Initialseed>  vectorS;
    
    // ---------check weather input threshold is qualified or not
    int i =0;
    while( i < Segmentinitialnum)
    {
        printf("\n********************Setting for object %d ***************\n", i+1);
        cout<<"Plaese choose method. \n tap 1, choose seeds by logging the threshold value. \n tap 2, choose seeds by clicking in orignal image. \n tap 3, choose seeds by default position of points"<<endl;
        int mode;
        cin >> mode;
        s[i] = Initialseed(mode, firstFrame, i, defaultThreshold, defaultseed);
        
        if(s[i].checkThreshold(firstFrame)){
            cout<< "Object " << i+1 << " was created successfully "<<endl;
            vectorS.push_back( s[i] );
            i++;
        }
    }
    
//------------------create text for recording the inital seed point before RG
    //char const *savePathtxt = "/Users/yanbo/Desktop/source/output/output2.txt";
    string savePathtxt;
    savePathtxt.assign(savePath);
    savePathtxt.append(".txt");
    
    ofstream outputtext;
    outputtext.open(savePathtxt,ios::out|ios::trunc);
    if (!outputtext.is_open())
    {
        cout << "Failed to open outputtext file! \n" << endl;
        return 1;
    }
    
    outputtext << "Pixel-real distance relation: " << endl ;
    for(size_t i=0; i< relationpointvektor.size(); i++)
    {
        outputtext << "Row: " << relationpointvektor[i].y << " Column: "  << relationpointvektor[i].x << endl;
    }
    outputtext << "Pixeldistance: " << pixeld << " Real distance: " << distance <<endl;
    outputtext << "The relation: " << initialpixelrelation << " m/pixel \n" << endl;
    relationpointvektor.clear();
    
    for( int i=0; i< Segmentinitialnum; i++)
    {
        outputtext << "Objekt: " << i+1 << endl;
        outputtext << "Initial Threshold: " << s[i].differencegrow << endl;
        outputtext << "Row  Column " << endl;
        for(size_t j=0;j<s[i].initialseedvektor.size();j++)
        {
            outputtext << s[i].initialseedvektor[j].y << "  " << s[i].initialseedvektor[j].x << endl ;
        }
        outputtext << "\n";
    }
    outputtext << flush;
    outputtext.close();

//------------------- creating Trackbar
    totalFrame = vc.get(CV_CAP_PROP_FRAME_COUNT);  //获取总帧数
    frameRate = vc.get(CV_CAP_PROP_FPS);   //获取帧率
    double pauseTime=1000/frameRate; // 由帧率计算两幅图像间隔时间
    namedWindow(windowNameRG, WINDOW_NORMAL);
    //在图像窗口上创建控制条
    createTrackbar(trackBarName,windowNameRG,&trackbarValue,trackbarMax,TrackBarFunc);
    //TrackBarFunc(0,0);
    
//------------------------------- Start to apply Segmentation-method in Video
    
    bool stop(false);
    bool all_threshold_notchange(true);
    Mat frame_backup;
    int indexFrame;
    bool bSuccess;
    //double totaltime = 0;
    //clock_t start,end;
    
    double totoalRGtime = 0, totoalOPtime = 0, totoalFMtime = 0;
    double AccumscaleRG = 1.0, AccumscaleOP = 1.0, AccumscaleFM = 1.0;
    clock_t start_RG, end_RG, start_OP, end_OP, start_FM, end_FM;
    
    double relationRG = initialpixelrelation , relationOP = initialpixelrelation, relationFM = initialpixelrelation;
    
    //start = clock();
    
    while(!stop)
    {
        
        Mat frame;//定义一个Mat变量，用于存储每一帧的图像
        
        int Segmentnum;
        
        vector<double> templateScaleSameframe;
        //Mat MatOut(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
        
        cout <<"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
        
        Segmentnum = vectorS.size();
        //cout<<"                  Amount of Segments: " << Segmentnum <<endl;
        
        
        if (all_threshold_notchange){    // if threshold for this frame did not change, read the next frame
            
            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
            printf("\n----------------IndexFrame: %d -----------------\n", indexFrame);
            bSuccess = vc.read(frame); // read a new frame from video
            frame_backup = frame.clone();
            
            for( int i=0; i<Segmentnum; i++)
            {
                vectorS[i].RGThreshold.clear();
                vectorS[i].RGThreshold.push_back(vectorS[i].differencegrow);
                vectorS[i].LoopThreshold = 1;
            }
        }
    
        else{   // if threshold of any segment in this frame changed , read the same frame
            
            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
            bool success = vc.set(CV_CAP_PROP_POS_FRAMES, indexFrame-1);
            if (!success) {
                cout << "Cannot set frame position from video file at " << indexFrame-1 << endl;
                return -1;
            }
            
            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
            printf("\n-----------------IndexFrame: %d ------------------\n", indexFrame);

            bSuccess = vc.read(frame);
        }
        
        if (frame.empty())
        {
            cout << "Video play over/ in loop" <<endl;
            //waitKey(0);
            break;
        }

        if (!bSuccess) //if not success, break loop
        {
            cout << "ERROR: Cannot read a frame from video" << endl;
            break;
        }

//----------------- add the text(frame index number) to written video frame
        
        vector<string> text; // RG frameindex time relation
        vector<string> scaletext; // three methods
        vector<string> seedtext;  // RG
        vector<string> Thresholdtext; // RG
        vector<string> timetext; // three methods  time and pixel relation
        vector<string> optext; // Optical flow / Scale/ frame index / time
        vector<string> pixelrelatext; // three methods pixel relation
        
        char Buffer[60];
        char Scalechar[70];
        char Timechar[70];
        char Totoaltimechar[70];
        char Acscale[70];
        char pixelrela[70];
        
        Point ptTopLeft(10, 10);
        Point* ptrTopLeft = &ptTopLeft;
        
        Point ptTopLeft2(10, 10);     // for time value (three methode in same frame )
        Point* ptrTopLeft2 = &ptTopLeft2;
        
        Point ptTopright(frame.cols-10, 10); // RG time/ Frame index/ relation
        Point* ptrTopright = &ptTopright;
        
        Point ptBottomMiddle(frame.cols/2, frame.rows);  // for optical flow / Scale value/ Frame index
        Point* ptrBottomMiddle = &ptBottomMiddle;
        
        Point ptBottomMiddle2(frame.cols/2, frame.rows); //  for counter
        Point* ptrBottomMiddle2 = &ptBottomMiddle2;
        
        Point ptBottomMiddle3(frame.cols/2, frame.rows); //  for time value (three methode in same frame )
        Point* ptrBottomMiddle3 = &ptBottomMiddle3;

// --------  Feature detection and classic matching method
        start_FM = clock();
        cout<< "Feature detection and matching" <<endl;
        vector<Point2f> obj_last;
        vector<Point2f> obj_next;
        Featurematch(preFrame, frame, obj_last, obj_next);
        
        Mat H_Featurematch = findHomography( obj_last, obj_next ,CV_RANSAC );
        //cout<<"Homography matrix:" <<endl << H_Featurematch <<endl << endl;
        double scaleFreaturematch = decomposematrix(H_Featurematch);
        cout<< "Scale (Freaturematch):" << scaleFreaturematch << endl;
        
        sprintf( Buffer, "Frame %d", indexFrame);
        scaletext.push_back(Buffer);
        
        sprintf(Scalechar , "Scale (index %d to %d)/ FM: %.7f/ ", indexFrame, indexFrame-1, scaleFreaturematch);
        end_FM = clock();
        
        sprintf(Timechar, "Time of Frameindex %d/ FM: %.5f/ ",indexFrame, (double)(end_FM - start_FM) / CLOCKS_PER_SEC);
        
        cout<<endl;
        
// --------  optical flow
        start_OP = clock();
        
        cout<< "Optical flow" <<endl;
        Mat OPoutput = frame.clone();
        
        Opticalflow OP;
        
        vector<vector<Point2f> > matchingpairs(2);
        OP.matchedpairs(preFrame, frame, matchingpairs);
        
        //swap(points[1], points[0]);
        
        Mat H_OP = findHomography( matchingpairs[0], matchingpairs[1] ,CV_RANSAC );
        double scaleOP = decomposematrix(H_OP);
        cout<< "Scale (optical flow):" << scaleOP << endl;
        
        sprintf(Scalechar+ strlen(Scalechar), "OP: %.7f/ ", scaleOP);
        
        end_OP = clock();
        
        if(all_threshold_notchange){
            OP.trackpath(preFrame , frame, OPoutput, points, initial);
            swap(points[1], points[0]);
            
            sprintf( Buffer, "Frame: %d/Scale: %.7f/Interframe Time: %.5f", indexFrame, scaleOP, (double)(end_OP - start_OP) / CLOCKS_PER_SEC);
            optext.push_back(Buffer);
            OPoutput = putStats(optext, OPoutput, Vec3b(0,0,170), ptrBottomMiddle, 'b' );
            
            moveWindow(windowNameOP, 400, 500); // int x = column, int y= row
            imshow(windowNameOP, OPoutput);  //显示图像
            vwop << OPoutput;
        }
        
        sprintf(Timechar+strlen(Timechar), "OP: %.5f/ ", (double)(end_OP - start_OP) / CLOCKS_PER_SEC);
        
        cout<<endl;
        
// -------------------------------
        cout<< "Region growing" ;
        start_RG = clock();
        Mat frame_Blur = frame.clone();
        GaussianBlur(frame, frame_Blur, kernelsize,0,0);
        //blur( image, out, Size(3, 3));
        
        Regiongrowing R[Segmentnum];
        
        Mat Matsegment= frame.clone(); // segment 整块的输出
        Mat FramewithCounter = frame.clone(); // segment 的 counter 输出
        Mat framethreemethode = frame.clone(); //
        Mat MatOut;
        Mat Matallsegment(frame.size(),CV_8UC3,Scalar(0,0,0));
// ------------
       
//        for( int i=0; i<Segmentnum; i++)
//        {
//            char seedinfo[50];
//            for(size_t j=0; j<vectorS[i].initialseedvektor.size(); j++)
//            {
//                double B = frame.at<Vec3b>(vectorS[i].initialseedvektor[j])[0];
//                double G = frame.at<Vec3b>(vectorS[i].initialseedvektor[j])[1];
//                double R = frame.at<Vec3b>(vectorS[i].initialseedvektor[j])[2];
//                sprintf(seedinfo, "Obj %d: Seed before Segmentation (%d, %d) intensity: %.2f", i+1, vectorS[i].initialseedvektor[j].y, vectorS[i].initialseedvektor[j].x, (B+G+R)/3);
//                seedtext.push_back(seedinfo);
//            }
//            FramewithCounter = putStats(seedtext,FramewithCounter, vectorS[i].color, ptrTopLeft, 't');
//            seedtext.clear();
//        }

//        char frameindex[10];
//        sprintf( frameindex, "Frame %d",indexFrame);
//        text.push_back(frameindex);
//        FramewithCounter = putStats(text,FramewithCounter,Vec3b(255,255,255), ptrBottomMiddle2, 'b' );
//        text.clear();
        
 //------------------------------------------------------------------
        
            for( int i=0; i<Segmentnum; i++)
            {
                printf("\n** Objekt %d Information ****************", i+1);
                printf("\n** Cyele index for Threshold: %d\n", vectorS[i].LoopThreshold);
                MatOut = R[i].RegionGrow(frame, frame_Blur , vectorS[i].differencegrow, vectorS[i].initialseedvektor);
                
             // If true, the function finds an optimal affine transformation with no additional restrictions (6 degrees of freedom). Otherwise, the class of transformations to choose from is limited to combinations of translation, rotation, and uniform scaling (5 degrees of freedom).
//                Mat Affine = estimateRigidTransform(vectorS[i].preSegment,MatOut,false);  // estimateRigidTransform 有可能得到的 matrix 为0矩阵
//                //cout<<"Affine:" << endl << Affine<<endl;
//                double scaleRG_Affine = decomposematrix(Affine);
//                cout<< "scale RG Affine:" << scaleRG_Affine << endl;
                
                addWeighted(Matallsegment, 1, MatOut, 1, 0.0, Matallsegment);

                Mat Matoutbackup = MatOut.clone();
                FramewithCounter = R[i].FindCounter(Matoutbackup, FramewithCounter, vectorS[i].color);
                
                //cout << "EWlong: " << R[i].EWlong<< endl;
                //cout << "EWshort: " << R[i].EWshort<< endl;
                //cout << "Ratio: "  << C[i].Ratio <<endl;
                //cout << "Degree: "  << C[i].Degree <<endl;
                //cout << "Rectangle width: "  << C[i].rectanglewidth <<endl;
                //cout << "Rectangle width * pixelrelation : "  << C[i].rectanglewidth  * pixelrelation <<endl;
                //cout << "Rectangle height: "  << C[i].rectangleheight <<endl;
                cout << "contour area: "  << R[i].Area << endl;
                printf("Threshold: %.8f \n", vectorS[i].differencegrow );
               
                //double scale = ( (C[i].EWlong/vectorS[i].data[0].back()) + (C[i].EWshort/vectorS[i].data[1].back()) )/2 ;
                double scale = sqrt( ( R[i].EWlong* R[i].EWshort)/(vectorS[i].data[0].back() * vectorS[i].data[1].back()) );
                double scaleArea = sqrt( R[i].Area / vectorS[i].data[6].back() );
                //cout<< "EWlong[indexFrame-1] " << EWlong[indexFrame-1] << " EWlong[indexFrame-2] "<< EWlong[indexFrame-2] << endl;
                //printf("Scale (index %d to %d): %lf \n", indexFrame, indexFrame-1, scale );
                cout<< "Scale (index " << indexFrame << " to " << indexFrame-1<< "): " << scale <<endl;
                cout<< endl;
                
                templateScaleSameframe.push_back(scale);
                
                // text infomation show in frame
                double B_Centre = frame.at<Vec3b>(R[i].cntr)[0];
                double G_Centre = frame.at<Vec3b>(R[i].cntr)[1];
                double R_Centre = frame.at<Vec3b>(R[i].cntr)[2];
                
                sprintf(Buffer, "%d: (%d,%d)/intensity: %.2f/Scale: %.5f", i+1, R[i].cntr.y, R[i].cntr.x, (B_Centre + G_Centre + R_Centre) /3.0 , scale);
                seedtext.push_back(Buffer);
                
                FramewithCounter = putStats(seedtext, FramewithCounter, vectorS[i].color, ptrTopLeft, 't');
                
                sprintf(Buffer, "Threshold(obj %d): %.8f", i+1, vectorS[i].differencegrow );
                Thresholdtext.push_back(Buffer);
            
        //---------computeing the average scale and average Scale Difference
                
                double averageScale = averagevalue(considerNum, vectorS[i].data[3]);
                //cout <<"Average Scale of previous "<< considerNum <<" frames: " << averageScale<< endl;
                
//                vector<double>::iterator iter;
//                cout<< "ScaleDifference.size(): " << vectorS[i].data[4].size() << endl;
//                cout << "ScaleDifference vector = " << endl;;
//                for (iter= vectorS[i].data[4].begin(); iter != vectorS[i].data[4].end(); iter++)  {cout << *iter << " ";}
//                cout << endl;
  
                double averageScaleDifference = averagedifference(considerNum, vectorS[i].data[4]);
                
                cout << multipleScaleDiff <<"* Average ScaleDifference of previous "<< considerNum <<" frames: " <<multipleScaleDiff * averageScaleDifference<< endl;
                
                double ScaleDifference = scale -  averageScale;
                printf("ScaleDifference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, ScaleDifference );
                cout<< endl;
                
        // ---------- average Ratio and average Ratio Difference
                
                double averageRatio = averagevalue(considerNum, vectorS[i].data[2]);
                double averageRatioDifference = averagedifference(considerNum, vectorS[i].data[5]);
                double RatioDifference = R[i].Ratio - averageRatio;
//                printf("Ratio Difference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, RatioDifference );
                
        //------------ Area Difference
                double Areadiffernce = R[i].Area - vectorS[i].data[6].back();
                printf("Area Difference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, Areadiffernce );
                cout<< Areadifferencefactor << "* Area in last frame : " <<  Areadifferencefactor *(vectorS[i].data[6].back())  << endl ;
    
//                cout<< "RGThreshold vector Size: " << vectorS[i].RGThreshold.size() << endl;
//                cout << "RGThreshold vector = " << endl;;
//                for (iter = vectorS[i].RGThreshold.begin(); iter != vectorS[i].RGThreshold.end(); iter++) {
//                    cout << *iter << " ";
//                }
                
        //--------- update the thereshlod value for region growing if scale varies largely
                //cout<< 0.2*vectorS[i].data[6].back() <<endl;
                
                double newthreshold;
                if ( abs(Areadiffernce) <=  Areadifferencefactor *(vectorS[i].data[6].back()) )
                {
                    //printf("the Area of segment %d is stable (Area difference is smaller 0.2* Area in last frame) \n", i+1);
                    
                    if (abs(ScaleDifference) > multipleScaleDiff * averageScaleDifference ) {
                        
                        if ( ScaleDifference > 0.0){
                            newthreshold = vectorS[i].differencegrow - (0.03 / pow(2, vectorS[i].LoopThreshold - 1));
                            printf("!!!!!!!!!!!Update/ the threshlod value will be decreased/ Segment is too lagre (scale difference positive) \n");
                        }
                        else{
                            newthreshold = vectorS[i].differencegrow + (0.03/ pow(2, vectorS[i].LoopThreshold - 1));
                            printf("!!!!!!!!!!!Update/ the threshlod value will be increased/ Segment is too small (scale difference negative) \n");
                        }

                        
                        printf("New RG_Threshlod: %f \n", newthreshold);
            
                        vector<double>::iterator iterfind;
                        //iterfind = find( vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), newthreshold);
                        iterfind  = find_if(vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), [newthreshold](double b) { return fabs(newthreshold - b) < EPSILON; });
                        
                        if(iterfind == vectorS[i].RGThreshold.end()){
                            cout << "New RG_Threshlod ist not available in RG_Threshlod vector. Using New RG_Threshlod for next loop" << endl;
                            vectorS[i].RGThreshold.push_back(newthreshold);
                            vectorS[i].differencegrow = newthreshold;
                        }
                        else {
                            cout << "New RG_Threshlod ist already available in RG_Threshlod vector. Using old RG_Threshlod for next loop  " << endl;
                            vectorS[i].LoopThreshold = vectorS[i].LoopThreshold + 1;
                            //cout << "s[i].LoopThreshold: " << s[i].LoopThreshold  << endl;
                            
                            if(vectorS[i].LoopThreshold > Loopiterationmax){
                                //cout<< endl <<"######Delete this segment and Break the video becasue of infinitv Loop" << endl;
                                break;
                            }
                        }
                        
                        printf("RG_Threshold for next loop %f \n", vectorS[i].differencegrow);

                        //vc.set(CV_CAP_PROP_POS_FRAMES, indexFrame);  // 此处ser frame index 会引起死机
                        vectorS[i].threshold_notchange = false;
                    }
                
                
                    else{
                        //Sclae diffrence is acceptable / abs(ScaleDifference) < multipleScaleDiff * averageScaleDifference
                            
                        vectorS[i].threshold_notchange = true;
                        bool test_threshold_notchange = true;
                        
                        for( int i=0; i<Segmentnum; i++)
                        {
                            test_threshold_notchange = test_threshold_notchange && vectorS[i].threshold_notchange;  //防止 某个object 为ture 另一obj为false.  ture的那个 持续性输入
                        }
                            
                            while(test_threshold_notchange){
                                vectorS[i].data[0].push_back(R[i].EWlong);   // Green line. long axis
                                vectorS[i].data[1].push_back(R[i].EWshort);    // lightly blue line . short axis
                                vectorS[i].data[2].push_back(R[i].Ratio);
                                vectorS[i].data[3].push_back(scale);
                                vectorS[i].data[4].push_back(ScaleDifference);
                                vectorS[i].data[5].push_back(RatioDifference);
                                vectorS[i].data[6].push_back(R[i].Area);
                                vectorS[i].data[7].push_back(scaleArea);
                                
                                vectorS[i].preSegment  = MatOut.clone();
                                vectorS[i].initialseedvektor.clear();
                                //s[i].initialseedvektor.push_back(R[i].regioncenter);
                                vectorS[i].initialseedvektor.push_back(R[i].cntr);
                                break;
                            }
                    }
                    
                }
                
                
                else { // Area of segment is unstable / abs(Areadiffernce) >  0.2*(vectorS[i].data[6].back()
                    
                    //cout<<"Area of segment " << i+1 <<" is unstable" << endl;
                    
                    if (Areadiffernce <0){
                    printf("!!!!!!!!!!!Update / the current segment is too small\n");
                    newthreshold = vectorS[i].differencegrow + (0.05/ pow(2, vectorS[i].LoopThreshold - 1));
                    }
                    
                    else {
                    printf("!!!!!!!!!!!Update threshlod value becasue segment is too large \n");
                    newthreshold = vectorS[i].differencegrow - (0.05/ pow(2, vectorS[i].LoopThreshold - 1));
                    }
                    
                    cout<<"new RG_Threshold: " <<  newthreshold << endl;
                  
                    vector<double>::iterator iterfind2;
                    iterfind2  = find_if(vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), [newthreshold](double b) { return fabs(newthreshold - b) < EPSILON; });
                    //iterfind2  = find_if(v.begin(),v.end(),dbl_cmp(10.5, 1E-8));
                    
                    if(iterfind2 == vectorS[i].RGThreshold.end()){
                        cout << "New RG_Threshlod ist not available in RG_Threshlod vector. Using New RG_Threshlod for next loop" << endl;
                        vectorS[i].RGThreshold.push_back(newthreshold);
                        vectorS[i].differencegrow = newthreshold;
                        cout << endl;
                    }
                    
                    else {
                        cout << "New RG_Threshlod ist already available in RG_Threshlod vector. Using old RG_Threshlod for next loop  " << endl;
                        vectorS[i].LoopThreshold = vectorS[i].LoopThreshold + 1;
                        
                        if(vectorS[i].LoopThreshold > Loopiterationmax){
                            //cout<< "Delete this segment and Break the video becasue of infinitv Loop" << endl;
                            break;
                        }
                    }
                    
                    printf("RG_Threshold for next loop %f \n", vectorS[i].differencegrow);
                    
                    vectorS[i].threshold_notchange = false;
                    
                }
                
                
                for(size_t j=0; j<R[i].seedtogether.size(); j++)
                {
                    Matsegment.at<Vec3b>(R[i].seedtogether[j]) = vectorS[i].color;
                    //Matallsegment.at<Vec3b>(R[i].seedtogether[j]) = frame.at<Vec3b>(R[i].seedtogether[j]);
                }
                
                //Matsegment = putStats(Thresholdtext, Matsegment, vectorS[i].color, ptrBottomMiddle, 'b' );
                FramewithCounter = putStats(Thresholdtext, FramewithCounter, vectorS[i].color, ptrBottomMiddle2, 'b' );
                Thresholdtext.clear();
                seedtext.clear();
                
            }////segment big循环在这里截止
   
        end_RG = clock();
        sprintf( Timechar+strlen(Timechar), "RG: %.5f", (double)(end_RG - start_RG) / CLOCKS_PER_SEC);
        timetext.push_back(Timechar);
        totoalRGtime +=(double)(end_RG - start_RG) / CLOCKS_PER_SEC;
        
   //------------------------ Scale of different Segment in the same Frame to build  gaussian model
        
        cout<< endl << "***build gaussian model (Scale of different objects in the same Frame)"  << endl;
       
        for (int i = 0 ; i< templateScaleSameframe.size(); i++){
            cout<< templateScaleSameframe[i] << " ";
        }
        cout<<endl;
        
        double sum = accumulate( templateScaleSameframe.begin(), templateScaleSameframe.end(), 0.0);
        double mean =  sum / templateScaleSameframe.size(); // 均值 average value
        
        double accum  = 0.0;
        for_each (templateScaleSameframe.begin(), templateScaleSameframe.end(), [&](const double d) {
            accum  += (d-mean)*(d-mean);
        });
        double stdev = sqrt(accum/(templateScaleSameframe.size()-1)); //方差 standard deviation
        cout<< "mean: " << mean <<  "/ stdev: " << stdev << endl;
        
        for (int i = 0 ; i< templateScaleSameframe.size(); i++){
            
            if (templateScaleSameframe[i] > (mean + confidenceintervalfactor *stdev) || templateScaleSameframe[i] < (mean - confidenceintervalfactor *stdev))
            cout<< "the Scale of Segment " << i+1 << " is not qualified"<< endl;
            
            else cout<< "the Scale of Segment " << i+1 << " is qualified"<< endl;
        }
        

    //--------------------------
        
        // Frameindex
        sprintf( Buffer, "Frame %d",indexFrame);
        text.push_back(Buffer);
    
        // ---  running time
        //end = clock();
        //totaltime = (double)(end - start) / CLOCKS_PER_SEC;
        //printf( "%f seconds\n", (double)(end - start) / CLOCKS_PER_SEC); // 在linux系统下，CLOCKS_PER_SEC 是1000000，表示的是微秒。 CLOCKS_PER_SEC，它用来表示一秒钟会有多少个时钟计时单元
        
        sprintf(Buffer, "%f s", totoalRGtime );
        text.push_back(Buffer);
        
        // update bool all_threshold_notchange
        all_threshold_notchange = true; //不能放在括号外  如果放在括号外 上一局的的all_threshold_notchange会影响
        for( int i=0; i<Segmentnum; i++)
        {
            all_threshold_notchange = all_threshold_notchange && vectorS[i].threshold_notchange;
        }
        
        //  --  update Pixelrelation
        if (all_threshold_notchange){    // if threshold for this frame did not change,  next frame will be read in next loop
            
            double averageScaleoneFrame = 0.0;
            for( int i=0; i<Segmentnum; i++)
            {
                averageScaleoneFrame += vectorS[i].data[7].back();
            }
            
            cout<< endl << "****************" <<endl;
            averageScaleoneFrame /= Segmentnum;
            cout<< "average Scale of all Segment in same Frame: " << averageScaleoneFrame<<endl;
            
            sprintf(Scalechar+strlen(Scalechar), "RG: %.7f", averageScaleoneFrame);
            scaletext.push_back(Scalechar);
            
            // pixelrelation for 3 method
            relationRG /= averageScaleoneFrame;
            cout<< "Pixelrelation: " << relationRG << "m/pixel"<<endl;
            
            relationOP /= scaleOP;
            relationFM /= scaleFreaturematch;
            
            //for optical flow and freature match
            preFrame = frame.clone();
            
            // accumulated Scale for op/fm /rg
            AccumscaleFM *= scaleFreaturematch;
            sprintf(Acscale, "Accumulated Sclae/ FM: %.7f/ ", AccumscaleFM);
            
            AccumscaleOP *= scaleOP;
            sprintf(Acscale+strlen(Acscale), "OP: %.7f/ ", AccumscaleOP);
            
            AccumscaleRG *= averageScaleoneFrame;
            sprintf(Acscale+strlen(Acscale), "RG: %.7f/ ", AccumscaleRG);
            scaletext.push_back(Acscale);
            
            // for show time of OP FM and RG
            totoalFMtime +=(double)(end_FM - start_FM) / CLOCKS_PER_SEC;
            sprintf(Totoaltimechar, "Totaltime/ FM: %.5f/ ", totoalFMtime);
            
            totoalOPtime +=(double)(end_OP - start_OP) / CLOCKS_PER_SEC;
            sprintf(Totoaltimechar +strlen(Totoaltimechar), "OP: %.5f/ ", totoalOPtime);
            
            sprintf(Totoaltimechar +strlen(Totoaltimechar), "RG: %.5f", totoalRGtime);
            timetext.push_back(Totoaltimechar);
            
            // pixel relation three menthod
            sprintf(pixelrela, "Pixel relation/ FM: %.6f/ OP: %.6f/ RG: %.6f", relationFM, relationOP,  relationRG);
            timetext.push_back(pixelrela);
        }
        
        framethreemethode = putStats(timetext,framethreemethode, Vec3b(0,0,170), ptrBottomMiddle3, 'b' );
        framethreemethode = putStats(scaletext,framethreemethode, Vec3b(0,230,230), ptrTopLeft2, 't' );
        imshow(" framethreemethode ", framethreemethode);

        sprintf(Buffer, "%.5fm/p", relationRG);
        text.push_back(Buffer);
        
        FramewithCounter= putStats(text,FramewithCounter, Vec3b(0,0,170), ptrTopright, 'r' );
        text.clear();
        
        moveWindow(windowNameRG, 700, 0); // int x = column, int y= row
        imshow(windowNameRG, FramewithCounter);  //显示图像
        // ---   Trackbar activate
        //TrackBarFunc(0,0);
        //controlRate++; // for trackbar !!!!!!

        //imshow ("Matsegment", Matsegment);

        vwRG << FramewithCounter;
        vw3method << framethreemethode;
        
//-------------delete the unuseful segment
        
        for( int i=0; i<vectorS.size(); i++){
            if (vectorS[i].LoopThreshold > Loopiterationmax) {
                cout<<endl << "!!!! Delete objekt " << i+1 << " because of infinitv loop" << endl << endl;
                vector<Initialseed>::iterator iterdelete = vectorS.begin() + i ;
                //deque<Initialseed>::iterator iterdelete = vectorS.begin() + i;
                vectorS.erase(iterdelete);
            }
        }
        
//------ check the  number of all segments. if segment amount is not enough. add a new object
        while (vectorS.size() < Segmentnum){
            cout<< "The objekt amount are not enough" << endl << "Setting for New objeckt: "<< vectorS.size() +1 <<endl;
            //Initialseed newobject = Initialseed(frame);
            Initialseed newobject = Initialseed(frame);
        
            if(newobject.checkThreshold(frame)){
                cout<< endl << "New segment sucessfully found" << endl;
                vectorS.push_back(newobject);
            }
        }

        
//-------------define the button
        
        int keycode = waitKey(0); // equal to  waitKey(10);  //延时10ms
        
        if(keycode  == ' ')   //32是空格键的ASCII值
            waitKey(0);
        
        if(keycode  == 27)  // 27 = ASCII ESC
            stop=true;
        //if(keycode  == 13)  // 13 = ASCII Enter
        
        if(keycode  == 45){  // 45 =  - minus
            cout <<endl<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
            cout<<"Please input the objekt number which you want to delete  "<<endl;
            int deletenum;
            cin >> deletenum;
            vector<Initialseed>::iterator iterdelete = vectorS.begin() + (deletenum - 1);
            //deque<Initialseed>::iterator iterdelete = vectorS.begin() + (deletenum - 1);
            vectorS.erase(iterdelete);
        }
        
        if(keycode  == 61){  // 43 =  + minus 加号（shift and = ）    等号 = ascii 61
            cout <<endl<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
            cout<<"Please input how much object you want to add "<<endl;
            int addnum;
            cin >> addnum;
            int i =0;
            while(i<addnum){
                cout<<"Setting for New projekt: "<< i+1 <<endl;
                
                Initialseed newobject = Initialseed(frame);

                if (newobject.checkThreshold(frame)){
                    cout<< endl << "New segment sucessfully found" << endl;
                    vectorS.push_back(newobject);
                    i++;
                }
            }
        }
        
        if(keycode  == 109){  // 109 =  m
            Mat FramewithCounterBackup;
            FramewithCounter.copyTo(FramewithCounterBackup);
            cout <<endl<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
            cout<<"Test! real distrance of two pixel length"<<endl;
            cout<<"Please mark two point on the image"<<endl;
            //#define Pixel_realtion_window "Pixel-distance relation"
            namedWindow( Pixel_realtion_window );
            setMouseCallback(Pixel_realtion_window,on_MouseHandle,(void*)&FramewithCounterBackup);
            
            while(1)
            {
                imshow( Pixel_realtion_window, FramewithCounterBackup );
                if( waitKey( 10 ) == 13 ) break;//按下enter键，程序退出
            }
            
            if (relationpointvektor.size() != 2){
                cout<< "\n!!!!!!!You did not mark 2 points. Porgramm breaks"  <<endl;
                return 0;
            }
            
            double pixeld = pixeldistance(FramewithCounterBackup, relationpointvektor);
            cout<< "Pixeldistance: " << pixeld <<endl;
            cout<< "The real length: " << relationRG * pixeld  << " m \n" << endl;
            
            imshow( Pixel_realtion_window, FramewithCounterBackup );
            waitKey(0);
            destroyWindow(Pixel_realtion_window);
        }
        relationpointvektor.clear();
        
    }
    
    cout << "Video plays over(outside loop)" << endl;
    
    ofstream outputtext2;
    outputtext2.open(savePathtxt,ios::out|ios::app);
    //outputtext2 << "Duration: " << totaltime << "s" << endl;
    outputtext2.close();
    
    vc.release();
    vwRG.release();
    vw3method.release();
    vwop.release();
    //destroyAllWindows();
    
/****************** play the written video */
//
//    VideoCapture vc2;
//    vc2.open( "/Users/yanbo/Desktop/source/output/output1.mov");
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
//        bool bSuccess = vc2.read(frame); // read a new frame from video
//        int indexFrame = vc2.get(CV_CAP_PROP_POS_FRAMES);
//        //cout<< "indexFrame" << indexFrame <<endl;
//
//        if (frame.empty())
//        {
//            cout << "video play over/ in loop " <<endl;
//            //waitKey(0);
//            break;
//        }
//
//        if (!bSuccess) //if not success, break loop
//        {
//
//            cout << "ERROR: Cannot read a frame "<< indexFrame <<" from output.avi" << endl;
//            break;
//        }
//
//        imshow("The written video", frame);  //显示当前帧
//        waitKey(1);  //延时1ms
//    }
//
//    vc2.release();
//    cout << "The written video plays over" << endl;
    
    exit(0);
    return 0;
}

//insert text lines to video frame
Mat putStats(vector<string> stats, Mat frame,Vec3b color, Point* origin, char word ){ // 也可用 Point& pt
    int font_face = FONT_HERSHEY_COMPLEX_SMALL;
    double font_scale = 0.85;
    int thickness = 1;
    int baseline;

    //origin.y = frame.rows ;
    switch (word) {
        case 'b':
            
            for(int i=0; i<stats.size(); i++){
                //获取文本框的长宽
                Size text_size = getTextSize(stats[i].c_str(), font_face, font_scale, thickness, &baseline);
                //文字在图像中的左下角 坐标 Origin of text ist Bottom-left corner of the text string in the image
                //origin.x = frame.cols/2 - text_size.width / 2;
                //origin.y = FramewithCounter.rows / 2 + text_size.height / 2;
                //origin.y = origin.y - 1.2*text_size.height ;
                (*origin).x = (*origin).x- text_size.width / 2;
                (*origin).y = (*origin).y - 1.2*text_size.height ;
                putText(frame, stats[i].c_str(), *origin, font_face , font_scale, color, thickness, 8, false);  //When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
                (*origin).x = (*origin).x+ text_size.width / 2;
            }
            break;
        
        case 't' :
            for(int i=0; i<stats.size(); i++){
                //获取文本框的长宽
                Size text_size = getTextSize(stats[i].c_str(), font_face, font_scale, thickness, &baseline);
                (*origin).y = (*origin).y + 1.3*text_size.height ;
                putText(frame, stats[i].c_str(), *origin, font_face , font_scale, color, thickness, 8, false);
            }
            break;
            
        case 'r' :
            for(int i=0; i<stats.size(); i++){
                //获取文本框的长宽
                Size text_size = getTextSize(stats[i].c_str(), font_face, font_scale, thickness, &baseline);
                (*origin).x = (*origin).x- text_size.width;
                (*origin).y = (*origin).y + 1.3*text_size.height ;
                putText(frame, stats[i].c_str(), *origin, font_face , font_scale, color, thickness, 8, false);
                (*origin).x = (*origin).x+ text_size.width;
            }
            break;
            
        default:
            break;
    }
    
    return frame;
}

double averagevalue(int num, vector<double> array){
    double average = array.back();
    vector<double>::reverse_iterator it;
    
    if (array.size()< num){
        for(it = array.rbegin(); it!= array.rend(); it++)
        {
            average = (average + *it)/2.0;
        }
    }
    
    else {
        for(it = array.rbegin(); it!= array.rbegin() + num; it++)
        {
            average = (average + *it)/2.0;
        }
    }
    return average;
}

double averagedifference(int num, vector<double> array){
    double average = abs(array.back());
    vector<double>::reverse_iterator it;
//    if (array.size() == 1){
//        average = 0.01;
//    }
    
    if (array.size()< num ){
        for(it = array.rbegin(); it!= array.rend(); it++)
        {
            average = (average + abs(*it))/2.0;
        }
    }
    
    else {
        for(it = array.rbegin(); it!= array.rbegin() + num; it++)
        {
            average = (average + abs(*it))/2.0;
        }
    }
    return average;
}


void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
    Mat& image = *(Mat*) param;
    //Mat *im = reinterpret_cast<Mat*>(param);
    if( x < 0 || x >= image.cols || y < 0 || y >= image.rows )
        return;
    
    if (event == EVENT_LBUTTONDOWN)
    {
        //左键按下消息
        Point g_pt = Point(x, y);
        cout<<"Row: "<<y<<", Column: "<< x <<endl;
        line(image, g_pt, g_pt, Scalar(0, 0, 255),4,8,0);
        relationpointvektor.push_back(g_pt);
    }
}

double pixeldistance(Mat& img, vector<Point> pv)
{
    double a = pv[0].x - pv[1].x;
    double b = pv[0].y - pv[1].y;
    line(img, pv[0], pv[1], Scalar(0, 0, 255),2,8,0);//随机颜色
    return(sqrt(a*a+b*b)); // a^2 not equal to a*a. a^2 has differnt meaning in Opencv
}

void Featurematch(Mat preframe , Mat nextframe, vector<Point2f>& obj_last, vector<Point2f>& obj_next){
    
    int minHessian = 2000;//SURF算法中的hessian阈值    the higher the minHessian, the fewer keypoints you will obtain, but you expect them to be more repetitive
    const int GOOD_PTS_MAX = 50;
    const float GOOD_PORTION = 0.05;
    
    //define Feature Detector and extractor（SURF） 特征检测类对象
    Ptr<SURF> f2d = xfeatures2d::SURF::create(minHessian);
    //Ptr<ORB> f2d = ORB::create();
    //Ptr<SIFT> f2d = xfeatures2d::SIFT::create();

    vector<KeyPoint> keypoints_pre, keypoints_next;//vector模板类，存放任意类型的动态数组
    Mat descriptors_pre, descriptors_next;
    
    //Ptr<SURF> detector = xfeatures2d::SURF::create(minHessian);
//    f2d->detect( preframe, keypoints_pre );
//    f2d->detect( nextframe, keypoints_next );
//    Ptr<SURF> extractor = SURF::create();
//    //Ptr<SIFT> extractor = xfeatures2d::SIFT::create();
    //SurfFeatureDetector detector( minHessian );  //opencv 2.0
//    //SurfDescriptorExtractor extractor;
//    f2d->compute( preframe, keypoints_pre, descriptors_pre );
//    f2d->compute( nextframe, keypoints_next, descriptors_next );
    
    f2d->detectAndCompute(preframe, Mat(), keypoints_pre, descriptors_pre);
    f2d->detectAndCompute(nextframe, Mat(), keypoints_next, descriptors_next);
    
    // Matching descriptor vectors using FLANN/BruteForce/ DescriptorMatcher matcher
    //BFMatcher matcher;
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    //Ptr<DescriptorMatcher> matcher(new BFMatcher(NORM_HAMMING, true));
    
    FlannBasedMatcher matcher;
    
    vector< DMatch > matches;
    matcher.match( descriptors_pre, descriptors_next, matches );
    
    
    //---------  two methods to choose good match pair
    //    //计算出关键点之间距离的最大值和最小值  Quick calculation of max and min distances between keypoints
    //    double max_dist = 0; double min_dist = 100;//最小距离和最大距离
    //    for( int i = 0; i < descriptors_pre.rows; i++ )
    //    {
    //        double dist = matches[i].distance;
    //        if( dist < min_dist ) min_dist = dist;
    //        if( dist > max_dist ) max_dist = dist;
    //    }
    //
    //    printf(">Max dist 最大距离 : %f \n", max_dist );
    //    printf(">Min dist 最小距离 : %f \n", min_dist );
    //
    //    //save only "good" matches ( whose distance is less than 3*min_dist ) 存下匹配距离小于3*min_dist的点对
    //    vector< DMatch > good_matches;
    //    for( int i = 0; i < descriptors_pre.rows; i++ )
    //    {
    //        if( matches[i].distance < 3*min_dist )
    //        {
    //            good_matches.push_back( matches[i]);
    //        }
    //    }
    
    //        // whose distance is less than 0.5*max_dist
    //        vector<DMatch> good_matches;
    //        double dThreshold = 0.5;    //匹配的阈值，越大匹配的点数越多
    //        for(int i=0; i<matches.size(); i++)
    //        {
    //            if(matches[i].distance < dThreshold * max_dist)
    //            {
    //                good_matches.push_back(matches[i]);
    //            }
    //        }
    
    // lowe's algotithm for select best matcher-- page 415 /418 书： 学习opencv3
    
    
    //-- Sort matches and preserve top 10% matches
    sort(matches.begin(), matches.end());
    vector< DMatch > good_matches;
//    double minDist = matches.front().distance;
//    double maxDist = matches.back().distance;
//    cout << "\nMax distance: " << maxDist << endl;
//    cout << "Min distance: " << minDist << endl;
    
    const int ptsPairs = min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back( matches[i]);
    }

    cout << "Calculating homography matrice using " << ptsPairs << " point pairs." << endl;
    
    //绘制出匹配到的关键点   draw the Good Match pairs
    Mat img_matches;
    drawMatches( preframe, keypoints_pre, nextframe, keypoints_next,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );
    //        Mat out;
    //        drawKeypoints(srcImage2, keypoints_next, out, Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //        imshow("", out);
    
//    vector<Point2f> obj_last;
//    vector<Point2f> obj_next;
    
    //从匹配成功的匹配对中获取关键点  -- Get the keypoints from the good matches
    for( unsigned int i = 0; i < good_matches.size(); i++ )
    {
        obj_last.push_back( keypoints_pre[ good_matches[i].queryIdx ].pt );
        obj_next.push_back( keypoints_next[ good_matches[i].trainIdx ].pt );
    }
    
//    //---- Get the four border points from the image_1 ( the object to be "detected" )
//    vector<Point2f> last_obj_corners(4);
//    last_obj_corners[0] = Point(0,0);
//    last_obj_corners[1] = Point( preframe.cols, 0 );
//    last_obj_corners[2] = Point( preframe.cols, srcImage1.rows );
//    last_obj_corners[3] = Point( 0, preframe.rows );
//
//    vector<Point2f> next_obj_corners(4);
//
//    // making perspective Transformation by H
//    perspectiveTransform( last_obj_corners, next_obj_corners, H);
//
//    Mat dst;
//    warpPerspective(srcImage1, dst,H, srcImage1.size());
//    imshow( "dst", dst );
    
    //绘制出四个角之间的直线  -- Draw lines between the corners (the mapped object in the scene - image_2 )
    //        line( img_matches, next_obj_corners[0] + Point2f( static_cast<float>(srcImage1.cols), 0), next_obj_corners[1] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar(255, 0, 123), 4 );
    //        line( img_matches, next_obj_corners[1] + Point2f( static_cast<float>(srcImage1.cols), 0), next_obj_corners[2] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );
    //        line( img_matches, next_obj_corners[2] + Point2f( static_cast<float>(srcImage1.cols), 0), next_obj_corners[3] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );
    //        line( img_matches, next_obj_corners[3] + Point2f( static_cast<float>(srcImage1.cols), 0), next_obj_corners[0] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );

}





