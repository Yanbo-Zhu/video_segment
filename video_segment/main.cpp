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
#include "fstream"
#include <string>


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
//Mat FramewithCounter;
Mat firstFrame;

vector<Mat> channels;
vector<Mat> channelsMatIn;
clock_t  clockBegin, clockEnd;
Mat putStats(vector<string> stats, Mat frame,Vec3b color,Point* origin, char word);
double averagevalue(int num, vector<double> array);
double averagedifference(int num, vector<double> array);
vector<vector<Point>> defaultseed(4);

//-------

//#define WINDOW_NAME " point marking "

//----- global function

//-------------------------------

//setting for trackbar
char const *windowName="Segment counter"; //播放窗口名称
char const *trackBarName="Frame index";    //trackbar控制条名称
double totalFrame=1.0;     //视频总帧数
//double currentFrame=1.0;    //当前播放帧
int trackbarValue=1;    //trackbar控制量
int trackbarMax=255;   //trackbar控制条最大值
double frameRate=1.0;  //视频帧率
VideoCapture vc;    //声明视频对象   !!!!!!!!!!!!!!!!!!!!!!!此处重要
double controlRate=0.1;

void TrackBarFunc(int ,void(*))
{
    controlRate=(double)trackbarValue/trackbarMax*totalFrame; //trackbar控制条对视频播放进度的控制
    //vc.set(CV_CAP_PROP_POS_FRAMES,controlRate);   //设置当前播放帧
}


int main( )
{
// some default seed points when i choose 2 during modechoose
    defaultseed[0].push_back(Point(233,246));
    defaultseed[1].push_back(Point(530,234)); // white boot
    
    double defaultThreshold[] = {5, 9} ;
//    cv::Mat image = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
//    //设置蓝色背景
//    image.setTo(cv::Scalar(100, 0, 0));
    
///-------------------------------------- VideoCapture ----------------
    //【1】读入视频
    //VideoCapture vc;
    char path[] = "/Users/yanbo/Desktop/source/" ;
    string videofilename = "70_20_descend";
    string videoInputpath;
    videoInputpath.assign(path);
    videoInputpath.append(videofilename);
    videoInputpath.append(".mp4");
    //vc.open( "/Users/yanbo/Desktop/source/Rotation_50m.mp4");
    //vc.open( "/Users/yanbo/Desktop/source/80_10_descend_rotation.mp4");
    //vc.open( "/Users/yanbo/Desktop/source/5-70.mp4");
    //vc.open( "/Users/yanbo/Desktop/source/80_10_descend.mp4");
    //vc.open( "/Users/yanbo/Desktop/source/70_20_descend_highDim.mp4");
    //vc.open( "/Users/yanbo/Desktop/source/70_20_descend.mp4");
    vc.open(videoInputpath);
    
    if (!vc.isOpened())
    {
        cout << "Failed to open a video device or video file!\n" << endl;
        return 1;
    }
    
    int Fourcc = static_cast<int>(vc.get( CV_CAP_PROP_FOURCC ));
    int IndexFrame  = vc.get(CV_CAP_PROP_POS_FRAMES);
    int FPS  = vc.get(CV_CAP_PROP_FPS);
    int FRAME_COUNT = vc.get(CV_CAP_PROP_FRAME_COUNT);
    int Width = vc.get(CV_CAP_PROP_FRAME_WIDTH);
    int Height = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
    printf("Fourcc: %d / indexFrame: %d / fps: %d / Total Frame: %d / Width * Height : %d * %d \n", Fourcc ,IndexFrame, FPS, FRAME_COUNT, Width, Height );
    
//-------------------------------------- VideoWriter ----------------

    char *savePath = new char[50];
    strcpy(savePath,path);
    strcat(savePath,"output/");
    strcat(savePath,videofilename.c_str());
    strcat(savePath,"_output7");     // savePath::  /Users/yanbo/Desktop/source/output/70_20_descend_output
    
    char savePathvideo[50] ;
    strcpy(savePathvideo,savePath);
    strcat(savePathvideo, ".mov");
    
    //char const *savePathvideo = "/Users/yanbo/Desktop/source/output/output2.txt";
    if(remove(savePathvideo)==0)
    {
        cout<<"Old output video delete successful"<<endl;
    }
    else
    {
        cout<<"Old output video delete failed"<<endl;
    }
    
    cout<<endl;
    
    VideoWriter vw; //(filename, fourcc, fps, frameSize[, isColor])
    vw.open( savePathvideo, // 输出视频文件名
            (int)vc.get( CV_CAP_PROP_FOURCC ),//CV_FOURCC('S', 'V', 'Q', '3'), // //CV_FOURCC('8', 'B', 'P', 'S'), // 也可设为CV_FOURCC_PROMPT，在运行时选取 //fourcc – 4-character code of codec used to compress the frames.
            (double)(vc.get( CV_CAP_PROP_FPS )/6), // 视频帧率
            Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ),
                 (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
            true ); // 是否输出彩色视频
    
    if (!vw.isOpened())
    {
        cout << "Failed to write the video! \n" << endl;
        return 1;
    }
    
//---------------- MatGrow Videowriter
    
    char savePathvideoGrow[50] ;
    strcpy(savePathvideoGrow,savePath);
    strcat(savePathvideoGrow,"_Grow");
    strcat(savePathvideoGrow, ".mov");
    
    //char const *savePathvideo = "/Users/yanbo/Desktop/source/output/output2.txt";
    if(remove(savePathvideoGrow)==0)
    {
        cout<<"Old Grow output video delete successful"<<endl;
    }
    else
    {
        cout<<"Old Grow output video delete failed"<<endl;
    }
    
    cout<<endl;
    
    VideoWriter vwGrow; //(filename, fourcc, fps, frameSize[, isColor])
    vwGrow.open( savePathvideoGrow, // 输出视频文件名
            (int)vc.get( CV_CAP_PROP_FOURCC ),//CV_FOURCC('S', 'V', 'Q', '3'), // //CV_FOURCC('8', 'B', 'P', 'S'), // 也可设为CV_FOURCC_PROMPT，在运行时选取 //fourcc – 4-character code of codec used to compress the frames.
            (double)(vc.get( CV_CAP_PROP_FPS )/6), // 视频帧率
            Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ),
                 (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
            true ); // 是否输出彩色视频
    
    if (!vwGrow.isOpened())
    {
        cout << "Failed to write the video! \n" << endl;
        return 1;
    }
    

//-----------------------------finding first seed point---------------
    //Mat firstFrame;
    Mat firstFrame;
    vc.read(firstFrame);

    vc.set(CV_CAP_PROP_POS_FRAMES, 0.0);
    
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
    cout<<"plaese choose method. \n tap 1, choose seeds by logging the threshold value. \n tap 2, choose seeds by clicking in orignal image. \n tap 3, choose seeds by default position of points"<<endl;
    
    //mode = 2;
    cin >> mode;
    
    cout<<"How many initial segment do you want: " <<endl;
    cin >> Segmentnum;
    //Segmentnum  = 1;
    
    Initalseed s[10];
    
    // settung color for diffent segments
    RNG rng(time(0));
    //RNG& rng = theRNG();
    Vec3b color[Segmentnum];
    for( int i=0; i<Segmentnum; i++)
    {
        color[i] = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
    
    for( int i=0; i<Segmentnum; i++)
    {
        printf("\n********************Setting for object %d ***************\n", i+1);
        
        s[i].modechoose(mode, firstFrame, i, defaultThreshold, defaultseed );
        s[i].drawpoint(firstFrame, s[i].initialseedvektor, color[i]);
        s[i].data.resize(6);
        //printf("\nPlease set the threshold value for region growing\n");
        //cin >> s[i].differencegrow;
        //s[i].differencegrow = 5.0;
    }
    
//---------------------create text for recording the inital seed point before RG
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

    for( int i=0; i<Segmentnum; i++)
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
    
    //outputtxt << "This is a line.\n";
    
//------------------- creating Trackbar
    totalFrame = vc.get(CV_CAP_PROP_FRAME_COUNT);  //获取总帧数
    frameRate = vc.get(CV_CAP_PROP_FPS);   //获取帧率
    double pauseTime=1000/frameRate; // 由帧率计算两幅图像间隔时间
    namedWindow(windowName);
    //在图像窗口上创建控制条
    createTrackbar(trackBarName,windowName,&trackbarValue,trackbarMax,TrackBarFunc);
    //TrackBarFunc(0,0);
    
//------------------------------- Start to apply Segmentation-method in Video
    
    bool stop(false);
    bool all_threshold_notchange(true);
    Mat frame_backup;
    int indexFrame = 0;
    bool bSuccess;

    while(!stop)
    {
        
        Mat frame;//定义一个Mat变量，用于存储每一帧的图像
        Mat MatOut;
        Mat Matfinal;
        //Mat MatOut(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
        
        all_threshold_notchange = true;
        for( int i=0; i<Segmentnum; i++)
        {
            all_threshold_notchange = all_threshold_notchange && s[i].threshold_notchange;
        }
        
        if (all_threshold_notchange){    // if threshold for this frame did not change, read the next frame
            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
            printf("\n----------------------------IndexFrame: %d ----------------------- \n ", indexFrame);
            bSuccess = vc.read(frame); // read a new frame from video
            frame_backup = frame.clone();
            
            for( int i=0; i<Segmentnum; i++)
            {
                s[i].RGThreshold.clear();
                s[i].RGThreshold.push_back(s[i].differencegrow);
                s[i].LoopThreshold = 1;
            }
        }
    
        else{   // if threshold for this frame  changed , ald read the same frame
            printf("\n----------------------------IndexFrame: %d ----------------------- \n ", indexFrame);
            frame = frame_backup.clone();
            bSuccess = true;
        }
        
        //若视频播放完成，退出循环
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
        
        Mat frame_Blur;
        Size kernelsize(5,5);
        GaussianBlur(frame, frame_Blur, kernelsize,0,0);
        //blur( image, out, Size(3, 3));
        
        Regiongrowing R[Segmentnum];
        Counter C[Segmentnum];
        
        Matfinal = frame.clone();
        Mat Matsegment(frame.size(),CV_8UC3,Scalar(0,0,0));
        Mat FramewithCounter = frame.clone();
        
//----------------- add the text(frame index number) to written video frame
        
        vector<string> text;
        vector<string> seedtext;
        Point ptTopLeft(10, 10);
        Point* ptrTopLeft = &ptTopLeft;
        Point ptBottomMiddle(FramewithCounter.cols/2, FramewithCounter.rows);
        Point* ptrBottomMiddle = &ptBottomMiddle;
        Point ptBottomMiddle2(FramewithCounter.cols/2, FramewithCounter.rows);
        Point* ptrBottomMiddle2 = &ptBottomMiddle2;
       
        for( int i=0; i<Segmentnum; i++)
        {
            char seedinfo[50];
            for(size_t j=0; j<s[i].initialseedvektor.size(); j++)
            {
                double B = frame.at<Vec3b>(s[i].initialseedvektor[j])[0];
                double G = frame.at<Vec3b>(s[i].initialseedvektor[j])[1];
                double R = frame.at<Vec3b>(s[i].initialseedvektor[j])[2];
                sprintf(seedinfo, "Obj %d: Seed before Segmentation (%d, %d) intensity: %.2f", i+1, s[i].initialseedvektor[j].y, s[i].initialseedvektor[j].x, (B+G+R)/3);
                seedtext.push_back(seedinfo);
            }
            FramewithCounter = putStats(seedtext,FramewithCounter, color[i], ptrTopLeft, 't');
            seedtext.clear();
        }

        char frameindex[10];
        sprintf( frameindex, "Frame %d",indexFrame);
        text.push_back(frameindex);
        
 //------------------------------------------------------------------
        for( int i=0; i<Segmentnum; i++)
        {
            printf("\n************* Objekt %d Information **********************", i+1);
            printf("\n****** Cyele index for Threshold: %d\n", s[i].LoopThreshold);
            //MatOut = R[i].RegionGrow(frame, frame_Blur , s[i].differencegrow, s[i].initialseedvektor);
            MatOut = R[i].RegionGrow(frame, frame_Blur , s[i].differencegrow, s[i].initialseedvektor);
            
            double intensity =(MatOut.at<Vec3b>(s[i].initialseedvektor.back())[0] + MatOut.at<Vec3b>(s[i].initialseedvektor.back())[1] + MatOut.at<Vec3b>(s[i].initialseedvektor.back())[2])/3.0 ;
            cout<< "intensity: " <<intensity <<endl;
            
//            Mat Mattemp;
//            cvtColor(MatOut, Mattemp, CV_BGR2GRAY);
//            int intensity2 = Mattemp.at<uchar>(s[i].initialseedvektor.back());
//            cout<< "intensity2: " <<intensity2 <<endl;
            
            if (intensity == 0 )
                continue;
            
            FramewithCounter = C[i].FindCounter(MatOut, FramewithCounter, color[i]);
            
            cout<< "Centre: Row " << C[i].cntr.y << " Column: " << C[i].cntr.x << endl;
            
            char Centre[50];
            double B_Centre = frame.at<Vec3b>(C[i].cntr)[0];
            double G_Centre = frame.at<Vec3b>(C[i].cntr)[1];
            double R_Centre = frame.at<Vec3b>(C[i].cntr)[2];
            sprintf(Centre, "Obj %d: Segment Centre(%d, %d) intensity: %.2f", i+1, C[i].cntr.y, C[i].cntr.x, (B_Centre+G_Centre+R_Centre)/3);
            seedtext.push_back(Centre);
            FramewithCounter = putStats(seedtext,FramewithCounter, color[i], ptrTopLeft, 't');
            seedtext.clear();
            
            cout << "EWlong: " << C[i].EWlong<<endl;
            cout << "EWshort: " << C[i].EWshort<<endl;
            cout << "Ratio: "  << C[i].Ratio <<endl;
            cout << "Degree: "  << C[i].Degree <<endl;
            cout<< "Threshold for RegionGrow: " << s[i].differencegrow << endl;
            
            char Thereshold[15];
            sprintf(Thereshold, "Threshold(obj %d): %.8f", i+1, s[i].differencegrow );
            text.push_back(Thereshold);
            
            if (indexFrame == 0){
                s[i].data[3].push_back(1.0); // scale
                s[i].data[4].push_back(0.0); // ScaleDifference
                s[i].data[0].push_back(C[i].EWlong);
                s[i].data[1].push_back(C[i].EWshort);
                s[i].data[2].push_back(C[i].Ratio);
                s[i].data[5].push_back(0.0); // RatioDifference
                s[i].initialseedvektor.clear();
                //s[i].initialseedvektor.push_back(R[i].regioncenter);
                s[i].initialseedvektor.push_back(C[i].cntr);
                s[i].threshold_notchange = true;
            }
           
            //cout<<"s[i].data[0].back(): "<<s[i].data[0].back() <<"  C[i].EWlong: "<<  C[i].EWlong <<endl;
            //double scale = ( (C[i].EWlong/s[i].data[0].back()) + (C[i].EWshort/s[i].data[1].back()) )/2 ;
            double scale = sqrt( (C[i].EWlong* C[i].EWshort)/(s[i].data[0].back() * s[i].data[1].back()) );
            //cout<< "EWlong[indexFrame-1] " << EWlong[indexFrame-1] << " EWlong[indexFrame-2] "<< EWlong[indexFrame-2] << endl;
            //printf("Scale (index %d to %d): %lf \n", indexFrame, indexFrame-1, scale );
            cout<< "Scale (index " << indexFrame << " to " << indexFrame-1<< "): " << scale <<endl;
            
            char scaletext[30];
            sprintf(scaletext, "Scale (obj %d/index %d to %d): %.5f", i+1, indexFrame, indexFrame-1, s[i].data[3].back());
            text.push_back(scaletext);
        
    //---------computeing the average scale and average Scale Difference
            int considerNum = 10;
            int multipleScaleDiff = 15;
            
            double averageScale = averagevalue(considerNum, s[i].data[3]);
            cout <<"Average Scale from last several frames: " << averageScale<< endl;
            double averageScaleDifference = averagedifference(considerNum, s[i].data[4]);
            cout << multipleScaleDiff <<"* Average ScaleDifference from last several frames: " <<multipleScaleDiff * averageScaleDifference<< endl;
            
            //double ScaleDifference = scale -  s[i].data[3].back();
            double ScaleDifference = scale -  averageScale;
            printf("ScaleDifference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, ScaleDifference );

    // ---------- avaerage Ratio and averafe Ratio Difference
            
            double averageRatio = averagevalue(considerNum, s[i].data[2]);
            cout <<"Average Ratio from last several frames: " << averageRatio<< endl;
            double averageRatioDifference = averagedifference(considerNum, s[i].data[5]);
            cout << multipleScaleDiff << "* Average RatioDifference from last several frames: " <<multipleScaleDiff * averageRatioDifference<< endl;
            double RatioDifference = C[i].Ratio - averageRatio;
            printf("Ratio Difference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, RatioDifference );
            
    //--------- update the thereshlod value for region growing if scale varies largely
            if (abs(ScaleDifference) > multipleScaleDiff * averageScaleDifference && abs(ScaleDifference) <= 0.8) {
                
                double newScaleDifference = 0.0;
                if ( ScaleDifference > 0.0){
                    newScaleDifference = s[i].differencegrow - (0.04 / pow(2, s[i].LoopThreshold - 1));
                    printf("!!!!!!!!!!!Update the threshlod value/ Segment (scale difference) is too lagre \n");
                }
                else{
                    newScaleDifference = s[i].differencegrow + (0.04/ pow(2, s[i].LoopThreshold - 1));
                    printf("!!!!!!!!!!!Update the threshlod value/ Segment (scale difference) is too small \n");
                }

                cout << "New RG_Threshlod: " << newScaleDifference  << endl;
    
                vector<double>::iterator iterfind;
                iterfind = find( s[i].RGThreshold.begin(), s[i].RGThreshold.end(), newScaleDifference);
                
                if(iterfind == s[i].RGThreshold.end()){
                    cout << "New RG_Threshlod ist not available in RG_Threshlod vector. Using New RG_Threshlod for next loop" << endl;
                    s[i].RGThreshold.push_back(newScaleDifference);
                    s[i].differencegrow = newScaleDifference;
                }
                else {
                    cout << "New RG_Threshlod ist already available in RG_Threshlod vector. Using old RG_Threshlod for next loop  " << endl;
                    s[i].LoopThreshold = s[i].LoopThreshold + 1;
                    //cout << "s[i].LoopThreshold: " << s[i].LoopThreshold  << endl;
                    
                    if(s[i].LoopThreshold > 20){
                        cout<< "Break the video becasue of infinitv Loop" << endl;
                        stop=true;
                    }
                }
                
                printf("RG_Threshold for next loop %f \n", s[i].differencegrow);

                //vc.set(CV_CAP_PROP_POS_FRAMES, indexFrame);
                s[i].threshold_notchange = false;
            }
        
            else if (abs(ScaleDifference) > 0.8 && abs(ScaleDifference) <= 1.2 ) { /// Object could be found. ScaleDifference  is negativ
                printf("!!!!!!!!!!!Update  threshlod value becasue Object could not be found \n");
                s[i].differencegrow = s[i].differencegrow + 0.05;
                printf("new RG_Threshold: %f \n", s[i].differencegrow);
                s[i].threshold_notchange = false;
                //waitKey(100);
            }
            
            else if (abs(ScaleDifference) > 1.2 ) { /// Object could be found. ScaleDifference  is negativ
                printf("!!!!!!!!!!!Update  threshlod value becasue scale value is too large \n");
                s[i].differencegrow = s[i].differencegrow - 0.05;
                printf("new RG_Threshold: %f \n", s[i].differencegrow);
                s[i].threshold_notchange = false;
                //waitKey(100);
            }
            
            else{
                if (indexFrame > 0){
                    
                s[i].threshold_notchange = true;
                bool test_threshold_notchange = true;
                for( int i=0; i<Segmentnum; i++)
                {
                    test_threshold_notchange = test_threshold_notchange && s[i].threshold_notchange;
                }
                //cout << "test_threshold_notchange: " << test_threshold_notchange <<endl;
                    
                    if(test_threshold_notchange){
                        s[i].data[3].push_back(scale);
                        s[i].data[4].push_back(ScaleDifference);
                        s[i].data[0].push_back(C[i].EWlong);
                        s[i].data[1].push_back(C[i].EWshort);
                        s[i].data[2].push_back(C[i].Ratio);
                        s[i].data[5].push_back(RatioDifference);
                        s[i].initialseedvektor.clear();
                        //s[i].initialseedvektor.push_back(R[i].regioncenter);
                        s[i].initialseedvektor.push_back(C[i].cntr);
                        
                    }
                }
            }
            
//            while(threshold_notchange){
//                char scale[30];-
//                sprintf(scale, "Scale (object %d/index %d to %d): %f", i+1, indexFrame, indexFrame-1, s[i].data[3].back());
//                text.push_back(scale);
//                break;
//            }
            
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

            for(size_t j=0; j<R[i].seedtogether.size(); j++)
            {
                Matfinal.at<Vec3b>(R[i].seedtogether[j]) = color[i];
                Matsegment.at<Vec3b>(R[i].seedtogether[j]) = color[i];
            }
            
            
            Matsegment = putStats(text,Matsegment,color[i], ptrBottomMiddle, 'b' );
            FramewithCounter = putStats(text,FramewithCounter,color[i], ptrBottomMiddle2, 'b' );
            text.clear();
            
        } ////segment big循环在这里截止
        
        imshow ("Matsegment", Matsegment);
        imshow ("segment", Matfinal);
        //imshow("segment counter", FramewithCounter);
        
//----------------- add the text(frame index number) to written video frame
        
//        vector<string> text;
//
//        char frameindex[10];
//        sprintf( frameindex, "Frame %d",indexFrame);
//        text.push_back( frameindex);
//        Point pt(FramewithCounter.cols/2, FramewithCounter.rows);
//        Point* ptr = &pt;
//
//        for( int i=0; i<Segmentnum; i++){
//            char Thereshold[15];
//            sprintf(Thereshold, "Threshold(object %d): %f", i+1, s[i].differencegrow );
//            text.push_back(Thereshold);
//
//            while(threshold_notchange){
//                char scale[30];
//                sprintf(scale, "Scale (object %d/index %d to %d): %f", i+1, indexFrame, indexFrame-1, s[i].data[3].back());
//                text.push_back(scale);
//                break;
//            }
//
//        FramewithCounter = putStats(text,FramewithCounter,color[i], ptr);
//        text.clear();
//        }
        
//        int j;
//        string time_str;
//        char text[50];
//        j = sprintf(text, " Frame %d/ ",indexFrame);
//        for( int i=0; i<Segmentnum; i++)
//        {
//            j += sprintf( text + j, "Threshold(%d): %.2f/ ", i+1, s[i].differencegrow );
//        }
//
//        //time_str=ctime;
//        //text.append(time_str);
//
//        int font_face = FONT_HERSHEY_COMPLEX;
//        double font_scale = 0.7;
//        int thickness = 2;
//        int baseline;
//        //获取文本框的长宽
//        Size text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);
//
//        //将文本框居中绘制
//        Point origin;  //文字在图像中的左下角 坐标 Origin of text ist Bottom-left corner of the text string in the image
//        origin.x = FramewithCounter.cols / 2 - text_size.width / 2;
//        //origin.y = FramewithCounter.rows / 2 + text_size.height / 2;
//        origin.y = FramewithCounter.rows - text_size.height ;
//        Scalar color = CV_RGB(255,0,0);
//        putText(FramewithCounter, text, origin, font_face, font_scale, color, thickness, 8, false); //When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
        
        
//        stringstream s;
//        s << "cv::putText() Demo!" ;
//        s << "cv::putText() Demo!" ;
//        s << "cv::putText() Demo!" ;
//        s << "cv::putText() Demo!" ;
//        putText(FramewithCounter,s.str(),Point(100,100), FONT_HERSHEY_COMPLEX, 1, Scalar(255,0,0), 2, 8 );

        imshow(windowName,FramewithCounter);  //显示图像
       
        controlRate++; // for trackbar
        
//----------------- add the text(frame index number) to written video frame

        
        //vw.write(frame);
        vw << FramewithCounter;
        vwGrow << Matsegment;
        
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
        
        
//-------------define the stop-button and exit-button
        
        int keycode = waitKey(0); // equal to  waitKey(10);  //延时10ms
        if(keycode  == ' ')   //32是空格键的ASCII值
            waitKey(0);
        
        if(keycode  == 27)  // 27 = ASCII ESC
            stop=true;
        //if(keycode  == 32)  // 32 = ASCII Enter
            //stop=true;
        
        //if (keycode > 0 )
           // waitKey(0);
    }
    
    cout << "Video plays over(outside loop)" << endl;
    cout << endl;
    vc.release();
    vw.release();
    vwGrow.release();
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
    double font_scale = 1;
    int thickness = 1;
    int baseline;
    //Scalar color = CV_RGB(255,255,0);
    //Point origin;
    //origin.y = frame.rows ;
    switch (word) {
        case 'b':
            
            for(int i=0; i<stats.size(); i++){
                //获取文本框的长宽
                Size text_size = getTextSize(stats[i].c_str(), font_face, font_scale, thickness, &baseline);
                //将文本框居中绘制
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
                //将文本框居中绘制
                //文字在图像中的左下角 坐标 Origin of text ist Bottom-left corner of the text string in the image
                //origin.x = frame.cols/2 - text_size.width / 2;
                //origin.y = FramewithCounter.rows / 2 + text_size.height / 2;
                //origin.y = origin.y - 1.2*text_size.height ;
                //(*origin).x = (*origin).x- text_size.width / 2;
                (*origin).y = (*origin).y + 1.3*text_size.height ;
                putText(frame, stats[i].c_str(), *origin, font_face , font_scale, color, thickness, 8, false);  //When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
                //(*origin).x = (*origin).x+ text_size.width / 2;
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
        //cout<<"Scale vector size: " << s[i].data[3].size() << endl;
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
    if (array.size() == 1){
        average = 0.01;
    }
    
    else if (array.size()< num && array.size() > 1){
        for(it = array.rbegin(); it!= array.rend(); it++)
        {
            average = (average + abs(*it))/2.0;
        }
    }
    
    else {
        //cout<<"Scale vector size: " << s[i].data[3].size() << endl;
        for(it = array.rbegin(); it!= array.rbegin() + num; it++)
        {
            average = (average + abs(*it))/2.0;
        }
    }
    return average;
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
