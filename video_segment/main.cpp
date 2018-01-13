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
#include <list>

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
int Segmentinitialnum;
//Mat firstFrame;
Size kernelsize(3,3);

int Width ;
int Height ;
int checkThresholditeration ;
bool repeat_thres(false);

vector<Mat> channels;
vector<Mat> channelsMatIn;
vector<Point> relationpointvektor;

clock_t  clockBegin, clockEnd;

//#define WINDOW_NAME " point marking "
//#define windowName String(XXX) //播放窗口名称

//----- global function
Mat putStats(vector<string> stats, Mat frame,Vec3b color,Point* origin, char word);
double averagevalue(int num, vector<double> array);
double averagedifference(int num, vector<double> array);
void on_MouseHandle(int event, int x, int y, int flags, void* param);
double pixeldistance(Mat& img, vector<Point> pv);
void checkThreshold(Mat Frame, Initialseed & seedclass, int& iterationnum, bool& repeat_thres);
//-------------------------------

//setting for trackbar

char const *windowName= "Segment counter "; //播放窗口名称
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

//void PrintInt(const int&nData)
//{
//    cout<<nData<<endl;
//}


int main( )
{
//    vector<int> vecInt;
//    for(int i=0; i<10;++i)
//    {
//        vecInt.push_back(i);
//    }
//    cout<<"向量中的内容为："<<endl;
//    for_each(vecInt.begin(),vecInt.end(),PrintInt);
//    cout<<"vector contains "<<vecInt.size()<<" elements"<<endl;
//    vecInt.pop_back();//删除最后一个元素
//    cout<<"删除最后一个元素后，vector contains "<<vecInt.size()<<" elements"<<endl;
//    vector<int>::iterator k = vecInt.begin()+1;
//    vecInt.erase(k);//删除第一个元素
//    //vecInt.erase(k); //迭代器k已经失效，会出错
//    cout<<"删除第一个元素后，vector contains "<<vecInt.size()<<" elements"<<endl;
//    //vecInt.erase(vecInt.begin(),vecInt.end()); //删除所有元素
//    //cout<<"删除所有元素后，vector contains "<<vecInt.size()<<"elements"<<endl; //输出为0
//    vector<int>::iterator vecNewEnd =remove(vecInt.begin(),vecInt.end(),5); //删除元素
//    cout<<"删除元素后，vector contains "<<vecInt.size()<<" elements"<<endl;
//    cout<<"向量开始到新结束为止的元素："<<endl;
//    for_each(vecInt.begin(),vecNewEnd,PrintInt);
//    cout<<"向量中的元素："<<endl;
//    for_each(vecInt.begin(),vecInt.end(),PrintInt);
    
//--------------some default seed points when i choose 2 during modechoose
// Point (x,y )  x= column  y = row
    vector<vector<Point>> defaultseed(4);
    defaultseed[0].push_back(Point(781,379)); // grass
    defaultseed[1].push_back(Point(215,240)); // light blue roof left
    defaultseed[2].push_back(Point(530,234)); // white boot
    defaultseed[3].push_back(Point(491,356)); // black roof bottem
    double defaultThreshold[] = {4,11, 12 ,8} ;
    
//    defaultseed[0].push_back(Point(531,240)); // white boot
//    double defaultThreshold[] = {33} ;
    
//    cv::Mat image = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
//    //设置蓝色背景
//    image.setTo(cv::Scalar(100, 0, 0));
    
///-------------------------------------- VideoCapture ----------------
    //【1】读入视频
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
    int IndexFrame  = vc.get(CV_CAP_PROP_POS_FRAMES);
    int FPS  = vc.get(CV_CAP_PROP_FPS);
    int FRAME_COUNT = vc.get(CV_CAP_PROP_FRAME_COUNT);
    Width = vc.get(CV_CAP_PROP_FRAME_WIDTH);
    Height = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
    printf("Fourcc: %d / indexFrame: %d / fps: %d / Total Frame: %d / Width * Height : %d * %d / Width*Height/300 : %d / Width*Height/40: %d \n", Fourcc ,IndexFrame, FPS, FRAME_COUNT, Width, Height, Width*Height/250, Width*Height/50);
    
//    string XXX = "Segment counter ";
//    char *windowName = new char[50];
//    strcat(windowName,XXX.c_str());
//    strcat(windowName,videofilename.c_str());
    
//-------------------------------------- VideoWriter ----------------

    char *savePath = new char[50];
    strcpy(savePath,path);
    strcat(savePath,"output/");
    strcat(savePath,videofilename.c_str());
    strcat(savePath,"_output_11");     // savePath::  /Users/yanbo/Desktop/source/output/70_20_descend_output
    cout<< "savePath: " << savePath <<endl;
    cout<<endl;
    
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
    
    cout<<endl;
//---------------- MatGrow Videowriter
    
    char savePathvideoGrow[50] ;
    strcpy(savePathvideoGrow,savePath);
    strcat(savePathvideoGrow,"_Grow");
    strcat(savePathvideoGrow, ".mov");
    
    //char const *savePathvideo = "/Users/yanbo/Desktop/source/output/output2.txt";
    if(remove(savePathvideoGrow)==0)
    {
        cout<<"Old Grow output video was deleted successful"<<endl;
    }
    else
    {
        cout<<"Old Grow output video was deleted failed"<<endl;
    }
    
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
    
    cout<<endl;
    
    Mat firstFrame;
    vc.read(firstFrame);
    
    int initialindex = 0;
    vc.set(CV_CAP_PROP_POS_FRAMES, 1);
    
//---------------------- Setting the relation between real distrance and pixel distance
    
    Mat firstframeBackup; //= firstFrame.clone();
    firstFrame.copyTo(firstframeBackup);

    cout<<"Setting the relation between real distrance and pixel distance"<<endl;
    cout<<"Please mark two point on the image"<<endl;
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

    cout<< "Please input the real distance (m) for this line"<<endl;
    double distance;
    //cin >> distance;

    distance = 10;
    cout<<  distance << endl;

    double pixelrelation = distance/ pixeld;
    cout<< "The relation: " << pixelrelation << " m/pixel \n" << endl;
    

//-----------------------------finding first seed point---------------

    
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    
//    cout<<"Plaese choose method. \n tap 1, choose seeds by logging the threshold value. \n tap 2, choose seeds by clicking in orignal image. \n tap 3, choose seeds by default position of points"<<endl;
//    cin >> mode;
    
    cout<<"How many initial segment do you want: " <<endl;
    cin >> Segmentinitialnum;
    //Segmentnum  = 1;
    
    int arraynum = 10;
    Initialseed s[Segmentinitialnum];
    vector<Initialseed>  vectorS;
    //deque<Initialseed>  vectorS;
    
//    // setting different color for segments
//    RNG rng(time(0));
//    //RNG& rng = theRNG();
//    Vec3b color[arraynum];
//    for( int i=0; i<arraynum; i++)
//    {
//        color[i] = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//        cout<< color[i]  << endl;
//    }

    // ---------check weather input threshold is qualified or not
    
    int i =0;
    while( i < Segmentinitialnum)
    {
        printf("\n********************Setting for object %d ***************\n", i+1);
        cout<<"Plaese choose method. \n tap 1, choose seeds by logging the threshold value. \n tap 2, choose seeds by clicking in orignal image. \n tap 3, choose seeds by default position of points"<<endl;
        cin >> mode;
        s[i] = Initialseed(mode, firstFrame, i, defaultThreshold, defaultseed);
        //Mat drawFrame = firstFrame.clone();
        //s[i].drawpoint(drawFrame, s[i].initialseedvektor);
        
        checkThreshold(firstFrame, s[i], checkThresholditeration, repeat_thres);
        
        if(checkThresholditeration>=1000 || repeat_thres){
            cout<< "this new random initial point can not become a available seed point " << endl;
            continue;
        }
        
        else{
            cout<< "object " << i << " was created successfully "<<endl;
            vectorS.push_back( s[i] );
            i++;
        }
        
    }
    
//    for( int i=0; i<Segmentinitialnum; i++)
//    {
//        printf("\n********************Setting for object %d ***************\n", i+1);
//        s[i] = Initialseed(mode, firstFrame, i, defaultThreshold, defaultseed);
//        //Mat drawFrame = firstFrame.clone();
//        //s[i].drawpoint(drawFrame, s[i].initialseedvektor);
//
//        checkThreshold(firstFrame, s[i], checkThresholditeration, repeat_thres);
//
////        Regiongrowing RTest;
////        Counter CTest;
////        bool threshold_qualified(true);
////
////
////        while (threshold_qualified) {
////            Mat Mattest =  firstFrame.clone();
////            Mat Matcounter =  firstFrame.clone();
////            Mat Mattest_Blur;
////            //Size kernelsize(5,5);
////            GaussianBlur(Mattest, Mattest_Blur, kernelsize, 0, 0);
////
////            Mattest = RTest.RegionGrow(Mattest, Mattest_Blur , s[i].differencegrow, s[i].initialseedvektor);
////            Matcounter = CTest.FindCounter(Mattest, Matcounter, s[i].color);
////            imshow("Contour of Segment", Matcounter);
////            waitKey(1);
////            cout<<"Area: " << CTest.Area << endl;
////            if (CTest.Area < (Width*Height/250)) s[i].differencegrow = s[i].differencegrow + 0.1;  // (Width*Height/250) = 2100   (Width*Height/50 )= 10000
////            else if (CTest.Area <= (Width*Height/50) ) break;
////            else  s[i].differencegrow = s[i].differencegrow - 0.1;
////            cout<< "Threshold: " << s[i].differencegrow <<endl;
////        }
////        waitKey(0);
////        destroyWindow("Contour of Segment");
//        //cout<<"Area: " << CTest.Area << endl;
//        //cout<< "Threshold: " << s[i].differencegrow <<endl;
//    }
    
//    vector<Initialseed>  vectorS;
    
//    for( int i=0; i<Segmentinitialnum; i++)
//    {
//        vectorS.push_back(s[i]);
//    }
    
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
    
    outputtext << "Pixel-real distance relation: " << endl ;
    for(size_t i=0; i< relationpointvektor.size(); i++)
    {
        outputtext << "Row: " << relationpointvektor[i].y << " Column: "  << relationpointvektor[i].x << endl;
    }
    outputtext << "Pixeldistance: " << pixeld << " Real distance: " << distance <<endl;
    outputtext << "The relation: " << pixelrelation << " m/pixel \n" << endl;
    relationpointvektor.clear();
    
    for( int i=0; i<Segmentinitialnum; i++)
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
    namedWindow(windowName, WINDOW_NORMAL);
    //在图像窗口上创建控制条
    createTrackbar(trackBarName,windowName,&trackbarValue,trackbarMax,TrackBarFunc);
    //TrackBarFunc(0,0);
    
//------------------------------- Start to apply Segmentation-method in Video
    
    bool stop(false);
    bool all_threshold_notchange(true);
    Mat frame_backup;
    int indexFrame = 0;
    bool bSuccess;
    double runningtime = 0;
    clock_t start,end;
    start = clock();
    
    while(!stop)
    {
        
        Mat frame;//定义一个Mat变量，用于存储每一帧的图像
        Mat MatOut;
        Mat Matfinal;
        int maxitrarion = 10;
        //Mat MatOut(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
        
        cout <<"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
        
        Segmentnum = vectorS.size();
        cout<<"                             Amount of Segments: " << Segmentnum <<endl;
        
//        all_threshold_notchange = true; //不能放在括号外  如果放在括号外 上一局的的all_threshold_notchange会影响
//        for( int i=0; i<Segmentnum; i++)
//        {
//            all_threshold_notchange = all_threshold_notchange && vectorS[i].threshold_notchange;
//        }
        
        if (all_threshold_notchange){    // if threshold for this frame did not change, read the next frame
            
            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
            printf("\n----------------------------IndexFrame: %d ----------------------- \n ", indexFrame);
            bSuccess = vc.read(frame); // read a new frame from video
            frame_backup = frame.clone();
            
            for( int i=0; i<Segmentnum; i++)
            {
                vectorS[i].RGThreshold.clear();
                vectorS[i].RGThreshold.push_back(vectorS[i].differencegrow);
                vectorS[i].LoopThreshold = 1;
            }
        }
    
        else{   // if threshold for this frame  changed , read the same frame
            //frame = frame_backup.clone();
            
            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
            bool success = vc.set(CV_CAP_PROP_POS_FRAMES, indexFrame-1);
            if (!success) {
                std::cout << "Cannot set frame position from video file at " << indexFrame-1 << std::endl;
                return -1;
            }
            
            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
            printf("\n----------------------------IndexFrame: %d ----------------------- \n ", indexFrame);

            bSuccess = vc.read(frame);

//            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
//            printf("\n----------------------------IndexFrame: %d ----------------------- \n ", indexFrame);
//
//            success = vc.set(CV_CAP_PROP_POS_FRAMES, indexFrame-1);
//            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
//            printf("\n----------------------------IndexFrame: %d ----------------------- \n ", indexFrame);
            
            //bSuccess = true;
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
        //Size kernelsize(5,5);
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
        
        Point ptTopright(FramewithCounter.cols-10, 10);
        Point* ptrTopright = &ptTopright;
        
        Point ptBottomMiddle(FramewithCounter.cols/2, FramewithCounter.rows);
        Point* ptrBottomMiddle = &ptBottomMiddle;
        Point ptBottomMiddle2(FramewithCounter.cols/2, FramewithCounter.rows);
        Point* ptrBottomMiddle2 = &ptBottomMiddle2;
       
        for( int i=0; i<Segmentnum; i++)
        {
            char seedinfo[50];
            for(size_t j=0; j<vectorS[i].initialseedvektor.size(); j++)
            {
                double B = frame.at<Vec3b>(vectorS[i].initialseedvektor[j])[0];
                double G = frame.at<Vec3b>(vectorS[i].initialseedvektor[j])[1];
                double R = frame.at<Vec3b>(vectorS[i].initialseedvektor[j])[2];
                sprintf(seedinfo, "Obj %d: Seed before Segmentation (%d, %d) intensity: %.2f", i+1, vectorS[i].initialseedvektor[j].y, vectorS[i].initialseedvektor[j].x, (B+G+R)/3);
                seedtext.push_back(seedinfo);
            }
            FramewithCounter = putStats(seedtext,FramewithCounter, vectorS[i].color, ptrTopLeft, 't');
            seedtext.clear();
        }

        char frameindex[10];
        sprintf( frameindex, "Frame %d",indexFrame);
        text.push_back(frameindex);
        FramewithCounter = putStats(text,FramewithCounter,Vec3b(255,255,255), ptrBottomMiddle2, 'b' );
        text.clear();
        
 //------------------------------------------------------------------
            for( int i=0; i<Segmentnum; i++)
            {
                printf("\n************* Objekt %d Information **********************", i+1);
                printf("\n****** Cyele index for Threshold: %d\n", vectorS[i].LoopThreshold);
                //MatOut = R[i].RegionGrow(frame, frame_Blur , s[i].differencegrow, s[i].initialseedvektor);
                MatOut = R[i].RegionGrow(frame, frame_Blur , vectorS[i].differencegrow, vectorS[i].initialseedvektor);
                
                double intensity =(MatOut.at<Vec3b>(vectorS[i].initialseedvektor.back())[0] + MatOut.at<Vec3b>(vectorS[i].initialseedvektor.back())[1] + MatOut.at<Vec3b>(vectorS[i].initialseedvektor.back())[2])/3.0 ;

    //            Mat Mattemp;
    //            cvtColor(MatOut, Mattemp, CV_BGR2GRAY);
    //            int intensity2 = Mattemp.at<uchar>(s[i].initialseedvektor.back());
    //            cout<< "intensity2: " <<intensity2 <<endl;
                
                if (intensity == 0 )
                    continue;
                
                FramewithCounter = C[i].FindCounter(MatOut, FramewithCounter, vectorS[i].color);
                
                cout<< "Centre: Row " << C[i].cntr.y << " Column: " << C[i].cntr.x << endl;
                
                char Centre[50];
                double B_Centre = frame.at<Vec3b>(C[i].cntr)[0];
                double G_Centre = frame.at<Vec3b>(C[i].cntr)[1];
                double R_Centre = frame.at<Vec3b>(C[i].cntr)[2];
                sprintf(Centre, "Obj %d: Segment Centre(%d, %d) intensity: %.2f", i+1, C[i].cntr.y, C[i].cntr.x, (B_Centre+G_Centre+R_Centre)/3);
                seedtext.push_back(Centre);
                FramewithCounter = putStats(seedtext,FramewithCounter, vectorS[i].color, ptrTopLeft, 't');
                seedtext.clear();
                
                cout << "EWlong: " << C[i].EWlong<<endl;
                cout << "EWshort: " << C[i].EWshort<<endl;
                cout << "Ratio: "  << C[i].Ratio <<endl;
                cout << "Degree: "  << C[i].Degree <<endl;
                //cout << "Rectangle width: "  << C[i].rectanglewidth <<endl;
                //cout << "Rectangle width * pixelrelation : "  << C[i].rectanglewidth  * pixelrelation <<endl;
                //cout << "Rectangle height: "  << C[i].rectangleheight <<endl;
                cout << "contour area: "  << C[i].Area <<endl;
                printf("Threshold: %.8f \n", vectorS[i].differencegrow );
                
                char Thereshold[15];
                sprintf(Thereshold, "Threshold(obj %d): %.8f", i+1, vectorS[i].differencegrow );
                text.push_back(Thereshold);
                
//                if (indexFrame == initialindex){
//                    vectorS[i].data[3].push_back(1.0); // scale
//                    vectorS[i].data[4].push_back(0.01); // ScaleDifference
//                    vectorS[i].data[0].push_back(C[i].EWlong);    // Green line. long axis
//                    vectorS[i].data[1].push_back(C[i].EWshort);   // lightly blue line . short axis
//                    vectorS[i].data[2].push_back(C[i].Ratio);
//                    vectorS[i].data[5].push_back(0.0); // RatioDifference
//                    vectorS[i].data[6].push_back(C[i].Area);  // Area
//                    vectorS[i].data[7].push_back(1.0);  // scaleArea
//                    vectorS[i].initialseedvektor.clear();
//                    //s[i].initialseedvektor.push_back(R[i].regioncenter);
//                    vectorS[i].initialseedvektor.push_back(C[i].cntr);
//                    vectorS[i].threshold_notchange = true;
//                }
               
                //double scale = ( (C[i].EWlong/vectorS[i].data[0].back()) + (C[i].EWshort/vectorS[i].data[1].back()) )/2 ;
                double scale = sqrt( ( C[i].EWlong* C[i].EWshort)/(vectorS[i].data[0].back() * vectorS[i].data[1].back()) );
                double scaleArea = sqrt( C[i].Area / vectorS[i].data[6].back() );
                //cout<< "EWlong[indexFrame-1] " << EWlong[indexFrame-1] << " EWlong[indexFrame-2] "<< EWlong[indexFrame-2] << endl;
                //printf("Scale (index %d to %d): %lf \n", indexFrame, indexFrame-1, scale );
                cout<< "Scale (index " << indexFrame << " to " << indexFrame-1<< "): " << scale <<endl;
                cout<< endl;
                
    //            char scaletext[30];
    //            sprintf(scaletext, "Scale (obj %d/index %d to %d): %.5f", i+1, indexFrame, indexFrame-1, s[i].data[3].back());
    //            text.push_back(scaletext);
            
        //---------computeing the average scale and average Scale Difference
                int considerNum = 10;
                int multipleScaleDiff = 15;
                
                double averageScale = averagevalue(considerNum, vectorS[i].data[3]);
                
                cout <<"Average Scale from last several frames: " << averageScale<< endl;
                
                            vector<double>::iterator iter;
                            //vector<double> v1 = s[i].data[0];
                            cout<< "ScaleDifference.size(): " << vectorS[i].data[4].size() << endl;
                            cout << "ScaleDifference vector = " ;
                            for (iter= vectorS[i].data[4].begin(); iter != vectorS[i].data[4].end(); iter++)  {cout << *iter << " ";}
                            cout << endl;
  
                double averageScaleDifference = averagedifference(considerNum, vectorS[i].data[4]);
                
                cout << multipleScaleDiff <<"* Average ScaleDifference from last several frames: " <<multipleScaleDiff * averageScaleDifference<< endl;
                
                //double ScaleDifference = scale -  s[i].data[3].back();
                double ScaleDifference = scale -  averageScale;
                printf("ScaleDifference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, ScaleDifference );
                cout<< endl;
        // ---------- average Ratio and average Ratio Difference
                
                double averageRatio = averagevalue(considerNum, vectorS[i].data[2]);
                //cout <<"Average Ratio from last several frames: " << averageRatio<< endl;
                double averageRatioDifference = averagedifference(considerNum, vectorS[i].data[5]);
                //cout << multipleScaleDiff << "* Average RatioDifference from last several frames: " <<multipleScaleDiff * averageRatioDifference<< endl;
                double RatioDifference = C[i].Ratio - averageRatio;
                //printf("Ratio Difference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, RatioDifference );
                cout<< endl;
                
        //------------ Area Difference
                double Areadiffernce = C[i].Area - vectorS[i].data[6].back();
                printf("Area Difference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, Areadiffernce );
                cout<< "0.2*(vectorS[i].data[6].back()): " << 0.2*(vectorS[i].data[6].back())  << endl <<endl;
                
                cout<< "RGThreshold(): " << vectorS[i].RGThreshold.size() << endl;
                cout << "RGThreshold vector = " ;
                for (iter= vectorS[i].RGThreshold.begin(); iter != vectorS[i].RGThreshold.end(); iter++)  {
                    cout << *iter << " ";
                    
                    
                    if(*iter == 5.2)
                    {
                        cout<< "5.1!!!" <<endl;
                    }
                }
                cout << endl;
                
                vector<double>::iterator iterfind;
                iterfind = find( vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), 5.15);

                
        //--------- update the thereshlod value for region growing if scale varies largely
                //cout<< 0.2*vectorS[i].data[6].back() <<endl;
                if ( abs(Areadiffernce) <=  0.2*(vectorS[i].data[6].back()) )
                {
                    printf("the Area of segment is stable\n");
                    
                    if (abs(ScaleDifference) > multipleScaleDiff * averageScaleDifference ) {
                        
                        //double newthreshold = 0.0;
                        double newthreshold;
                        if ( ScaleDifference > 0.0){
                            newthreshold = vectorS[i].differencegrow - (0.03 / pow(2, vectorS[i].LoopThreshold - 1));
                            printf("!!!!!!!!!!!Update/ the threshlod value will be decreased/ current segment (scale difference positive) is too lagre \n");
                        }
                        else{
                            newthreshold = vectorS[i].differencegrow + (0.03/ pow(2, vectorS[i].LoopThreshold - 1));
                            printf("!!!!!!!!!!!Update/ the threshlod value will be increased/Segment (scale difference negative) is too small \n");
                        }

                        //cout << "New RG_Threshlod: " << newthreshold  << endl;
                        printf("New RG_Threshlod: %f \n", newthreshold);
            
                        vector<double>::iterator iterfind;
                        iterfind = find( vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), newthreshold);
                        
                        if(iterfind == vectorS[i].RGThreshold.end()){
                            cout << "New RG_Threshlod ist not available in RG_Threshlod vector. Using New RG_Threshlod for next loop" << endl;
                            vectorS[i].RGThreshold.push_back(newthreshold);
                            vectorS[i].differencegrow = newthreshold;
                        }
                        else {
                            cout << "New RG_Threshlod ist already available in RG_Threshlod vector. Using old RG_Threshlod for next loop  " << endl;
                            vectorS[i].LoopThreshold = vectorS[i].LoopThreshold + 1;
                            //cout << "s[i].LoopThreshold: " << s[i].LoopThreshold  << endl;
                            
                            if(vectorS[i].LoopThreshold > maxitrarion){
                                //cout<< endl <<"######Delete this segment and Break the video becasue of infinitv Loop" << endl;
                                break;
                            }
                        }
                        
                        printf("RG_Threshold for next loop %f \n", vectorS[i].differencegrow);

                        //vc.set(CV_CAP_PROP_POS_FRAMES, indexFrame);
                        vectorS[i].threshold_notchange = false;
                    }
                
//                    else if (abs(ScaleDifference) > 0.8 && abs(ScaleDifference) <= 1.2 ) { /// Object could not be found. ScaleDifference is negativ
//                        printf("!!!!!!!!!!!Update threshlod value becasue Object could not be found \n");
//                        vectorS[i].differencegrow = vectorS[i].differencegrow + 0.05;
//                        printf("new RG_Threshold: %f \n", vectorS[i].differencegrow);
//                        vectorS[i].threshold_notchange = false;
//                        //waitKey(100);
//                    }
//
//                    else if (abs(ScaleDifference) > 1.2 ) { /// Segment grows unregular and is too large . ScaleDifference  is negativ
//                        printf("!!!!!!!!!!!Update threshlod value becasue scale value is too large \n");
//                        vectorS[i].differencegrow = vectorS[i].differencegrow - 0.05;
//                        printf("new RG_Threshold: %f \n", vectorS[i].differencegrow);
//                        vectorS[i].threshold_notchange = false;
//                        //waitKey(100);
//                    }
                
                    else{
                        if (indexFrame > initialindex){
                            
                        vectorS[i].threshold_notchange = true;
                        bool test_threshold_notchange = true;
                        for( int i=0; i<Segmentnum; i++)
                        {
                            test_threshold_notchange = test_threshold_notchange && vectorS[i].threshold_notchange;  //防止 某个object 为ture 另一obj为false.  ture的那个 持续性输入
                        }
                        //cout << "test_threshold_notchange: " << test_threshold_notchange <<endl;
                            
                            if(test_threshold_notchange){
                                vectorS[i].data[3].push_back(scale);
                                vectorS[i].data[4].push_back(ScaleDifference);
                                vectorS[i].data[0].push_back(C[i].EWlong);
                                vectorS[i].data[1].push_back(C[i].EWshort);
                                vectorS[i].data[2].push_back(C[i].Ratio);
                                vectorS[i].data[5].push_back(RatioDifference);
                                vectorS[i].data[6].push_back(C[i].Area);
                                vectorS[i].data[7].push_back(scaleArea);
                                vectorS[i].initialseedvektor.clear();
                                //s[i].initialseedvektor.push_back(R[i].regioncenter);
                                vectorS[i].initialseedvektor.push_back(C[i].cntr);
                                
                            }
                        }
                    }
                    
                }
                
                
                else {
                    
                    cout<<"Area of segment is unstable" << endl;
                    
                    double anothernewthreshold;
                    
                    if (Areadiffernce <0){
                    printf("!!!!!!!!!!!Update / the current segment is too small\n");
                    anothernewthreshold = vectorS[i].differencegrow + (0.05/ pow(2, vectorS[i].LoopThreshold - 1));
                    }
                    
                    else { /// Segment grows unregular and is too large . ScaleDifference  is negativ
                    printf("!!!!!!!!!!!Update threshlod value becasue segment is too large \n");
                    anothernewthreshold = vectorS[i].differencegrow - (0.05/ pow(2, vectorS[i].LoopThreshold - 1));
                    }
                    
                    printf("new RG_Threshold: %f \n", anothernewthreshold);

                    vector<double>::iterator iterfind2;
                    iterfind2 = find( vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), anothernewthreshold);
                    
                    if(iterfind == vectorS[i].RGThreshold.end()){
                        cout << "New RG_Threshlod ist not available in RG_Threshlod vector. Using New RG_Threshlod for next loop" << endl;
                        vectorS[i].RGThreshold.push_back(anothernewthreshold);
                        vectorS[i].differencegrow = anothernewthreshold;
                    }
                    else {
                        cout << "New RG_Threshlod ist already available in RG_Threshlod vector. Using old RG_Threshlod for next loop  " << endl;
                        vectorS[i].LoopThreshold = vectorS[i].LoopThreshold + 1;
                        //cout << "s[i].LoopThreshold: " << s[i].LoopThreshold  << endl;
                        
                        if(vectorS[i].LoopThreshold > maxitrarion){
                            //cout<< "Delete this segment and Break the video becasue of infinitv Loop" << endl;
                            break;
                        }
                    }
                    
                    printf("RG_Threshold for next loop %f \n", vectorS[i].differencegrow);
                    
                    vectorS[i].threshold_notchange = false;
                    
                }
                
    //            vector<double>::iterator iter;
    //            vector<double>::iterator iter2;
    //            //vector<double> v1 = s[i].data[0];
    //            //cout<< "s[i].data[3].size(): " << s[i].data[3].size() << endl;
    //            cout << "Scale vector = " ;
    //            for (iter= s[i].data[3].begin(); iter != s[i].data[3].end(); iter++){
    //                cout << *iter << " ";}
    //            cout << endl;

                for(size_t j=0; j<R[i].seedtogether.size(); j++)
                {
                    Matfinal.at<Vec3b>(R[i].seedtogether[j]) = vectorS[i].color;
                    Matsegment.at<Vec3b>(R[i].seedtogether[j]) = vectorS[i].color;
                }
                
                
                Matsegment = putStats(text,Matsegment, vectorS[i].color, ptrBottomMiddle, 'b' );
                FramewithCounter = putStats(text,FramewithCounter, vectorS[i].color, ptrBottomMiddle2, 'b' );
                text.clear();
                
            }////segment big循环在这里截止
        
        // running time
        end = clock();
        runningtime = (double)(end - start) / CLOCKS_PER_SEC;
        //printf( "%f seconds\n", (double)(end - start) / CLOCKS_PER_SEC); // 在linux系统下，CLOCKS_PER_SEC 是1000000，表示的是微秒。 CLOCKS_PER_SEC，它用来表示一秒钟会有多少个时钟计时单元
        
        char Duration[10];
        sprintf(Duration, "%f s", runningtime );
        text.push_back(Duration);
        
        // update bool all_threshold_notchange
        all_threshold_notchange = true; //不能放在括号外  如果放在括号外 上一局的的all_threshold_notchange会影响
        for( int i=0; i<Segmentnum; i++)
        {
            all_threshold_notchange = all_threshold_notchange && vectorS[i].threshold_notchange;
        }
        
        if (all_threshold_notchange){    // if threshold for this frame did not change,  next frame will be read in next loop
            
            double averageScaleoneFrame = 0.0;
            for( int i=0; i<Segmentnum; i++)
            {
                averageScaleoneFrame += vectorS[i].data[7].back();
            }
            
            cout<< "*************************************************" <<endl;
            averageScaleoneFrame /= Segmentnum;
            cout<< "average Scale of all Segment in same Frame: " << averageScaleoneFrame<<endl;
            pixelrelation /= averageScaleoneFrame;
            cout<< "Pixelrelation: " << pixelrelation << "m/pixel"<<endl;
        }
        
//        for( int i=0; i<Segmentnum; i++)
//        {
//            cout<< "Object " << i+1 << " rectangle diagonallength * pixelrelation: " << C[i].diagonallength * pixelrelation << endl;
//        }

        char relarion[10];
        sprintf(relarion, "%.5fm/p", pixelrelation);
        text.push_back(relarion);
        
        FramewithCounter= putStats(text,FramewithCounter, Vec3b(0,0,210), ptrTopright, 'r' );
        text.clear();
        

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

//        stringstream s;
//        s << "cv::putText() Demo!" ;
//        s << "cv::putText() Demo!" ;
//        s << "cv::putText() Demo!" ;
//        s << "cv::putText() Demo!" ;
//        putText(FramewithCounter,s.str(),Point(100,100), FONT_HERSHEY_COMPLEX, 1, Scalar(255,0,0), 2, 8 );
        
        moveWindow(windowName, 700, 0); // int x = column, int y= row
        imshow(windowName, FramewithCounter);  //显示图像
        //TrackBarFunc(0,0);
        //controlRate++; // for trackbar !!!!!!

        //imshow ("Matsegment", Matsegment);
        //imshow ("segment", Matfinal);
        
        //imshow("segment counter", FramewithCounter);
        
//----------------- add the text(frame index number) to written video frame

        vw << FramewithCounter;
        vwGrow << Matsegment;

        
//-------------define the stop-button and exit-button
        
        int keycode = waitKey(0); // equal to  waitKey(10);  //延时10ms
        
        if(keycode  == ' ')   //32是空格键的ASCII值
            waitKey(0);
        
        if(keycode  == 27)  // 27 = ASCII ESC
            stop=true;
        //if(keycode  == 13)  // 13 = ASCII Enter
            //stop=true;
        
        if(keycode  == 45){  // 45 =  - minus
            cout <<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
            cout<<"Please input the objekt number which you want to delete  "<<endl;
            int deletenum;
            cin >> deletenum;
            vector<Initialseed>::iterator iterdelete = vectorS.begin() + (deletenum - 1);
            //deque<Initialseed>::iterator iterdelete = vectorS.begin() + (deletenum - 1);
            vectorS.erase(iterdelete);
            //waitKey(0);
        }
        
        if(keycode  == 61){  // 43 =  + minus 加号（shift and = ）    等号 = ascii 61
            cout <<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
            cout<<"Please input the number of segments which you want to add "<<endl;
            int addnum;
            cin >> addnum;
            int i =0;
            while(i<addnum){
                cout<<"Setting for New projekt: "<< i+1 <<endl;
                //Initialseed newobject = Initialseed(frame);
                Initialseed newobject = Initialseed(frame, Width, Height);
                checkThreshold(frame, newobject, checkThresholditeration, repeat_thres);
                
                if(checkThresholditeration>=1000 || repeat_thres){
                    cout<< "this new random initial point can not become a available seed point " << endl;
                    continue;
                }
                
                else{
                    cout<< endl << "New segment sucessfully found" << endl;
                    vectorS.push_back(newobject);
                    i++;
                }
                
            }
            waitKey(0);
        }
        
        
        if(keycode  == 114){  // 114 =  r
        Mat FramewithCounterBackup;
        FramewithCounter.copyTo(FramewithCounterBackup);
        
        cout<<"\nTest! real distrance of two pixel length"<<endl;
        cout<<"Please mark two point on the image"<<endl;
        //#define Pixel_realtion_window "Pixel-distance relation"
        namedWindow( Pixel_realtion_window );
        setMouseCallback(Pixel_realtion_window,on_MouseHandle,(void*)&FramewithCounterBackup);
        
        while(1)
        {
            imshow( Pixel_realtion_window, FramewithCounterBackup );
            if( waitKey( 10 ) == 13 ) break;//按下enter键，程序退出
        }
        
        //cout<< "relationpoint vektor size= " << relationpointvektor.size() <<endl;
        if (relationpointvektor.size() != 2){
            cout<< "\n!!!!!!!You did not mark 2 points. Porgramm breaks"  <<endl;
            return 0;
        }
        
        double pixeld = pixeldistance(FramewithCounterBackup, relationpointvektor);
        cout<< "Pixeldistance: " << pixeld <<endl;
       
        cout<< "The real length: " << pixelrelation * pixeld  << " m \n" << endl;
            
        imshow( Pixel_realtion_window, FramewithCounterBackup );
        waitKey(0);
        destroyWindow(Pixel_realtion_window);
        }
        
        
//-------------delete the unuseful segment

        for( int i=0; i<vectorS.size(); i++){
            
            if (vectorS[i].LoopThreshold > maxitrarion) {
                cout<<endl << "####Delete objekt " << i+1 << " because of infinitv loop" << endl;
                    vector<Initialseed>::iterator iterdelete = vectorS.begin() + i ;
                    //deque<Initialseed>::iterator iterdelete = vectorS.begin() + i;
                    vectorS.erase(iterdelete);
            }
        }
    
    }
    
    cout << "Video plays over(outside loop)" << endl;
    
    ofstream outputtext2;
    outputtext2.open(savePathtxt,ios::out|ios::app);
    outputtext2 << "Duration: " << runningtime << "s" << endl;
    outputtext2.close();
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
    double font_scale = 0.9;
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
                putText(frame, stats[i].c_str(), *origin, font_face , font_scale, color, thickness, 8, false);  //When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
                //(*origin).x = (*origin).x+ text_size.width / 2;
            }
            break;
            
        case 'r' :
            for(int i=0; i<stats.size(); i++){
                //获取文本框的长宽
                Size text_size = getTextSize(stats[i].c_str(), font_face, font_scale, thickness, &baseline);
                (*origin).x = (*origin).x- text_size.width;
                (*origin).y = (*origin).y + 1.3*text_size.height ;
                putText(frame, stats[i].c_str(), *origin, font_face , font_scale, color, thickness, 8, false);  //When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
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


void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
    
    Mat& image = *(Mat*) param;
    //Mat *im = reinterpret_cast<Mat*>(param);
    //mouse ist not in window 处理鼠标不在窗口中的情况
    if( x < 0 || x >= image.cols || y < 0 || y >= image.rows )
        return;
    
    if (event == EVENT_LBUTTONDOWN)
    //switch(event)
    {
        //左键按下消息
        Point g_pt = Point(x, y);
        cout<<"Row: "<<y<<", Column: "<< x <<endl;
        line(image, g_pt, g_pt, Scalar(0, 0, 255),4,8,0);
        //DrawLine( image, g_pt );//画线
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

//void DrawLine( Mat& img, vector<Point> pv )
//{
//    line(img, pv[0], pv[1], Scalar(0, 0, 255),5,8,0);//随机颜色
//}

void checkThreshold(Mat Frame, Initialseed &seedclass, int& iterationnum, bool& repeat_thres){
    
    Regiongrowing RTest;
    Counter CTest;
    bool threshold_qualified(true);
    iterationnum = 0;
    repeat_thres = false;
    vector<double> Thresholdstack;
    
    while (threshold_qualified && (iterationnum <1000) && (!repeat_thres)) {
        Mat Mattest =  Frame.clone();
        Mat Matcounter =  Frame.clone();
        Mat Mattest_Blur;
        //Size kernelsize(5,5);
        GaussianBlur(Mattest, Mattest_Blur, Size(3,3), 0, 0);
        
        Mattest = RTest.RegionGrow(Mattest, Mattest_Blur , seedclass.differencegrow, seedclass.initialseedvektor);
        Matcounter = CTest.FindCounter(Mattest, Matcounter, seedclass.color);
        imshow("Contour of Segment", Matcounter);
        waitKey(100);
        cout<<"Area: " << CTest.Area << endl;
        
        Thresholdstack.push_back(seedclass.differencegrow) ;
        
        if (CTest.Area < (Width*Height/300)) seedclass.differencegrow = seedclass.differencegrow + 0.1;  // (Width*Height/250) = 2100   (Width*Height/50 )= 10000
        else if (CTest.Area <= (Width*Height/40) ) break;
        else  seedclass.differencegrow = seedclass.differencegrow - 0.1;
        
        iterationnum++;
        cout<< "iterationnum: " << iterationnum << endl;
        
        vector<double>::iterator iterfind;
        iterfind = find( Thresholdstack.begin(), Thresholdstack.end(), seedclass.differencegrow);
        
        if(iterfind != Thresholdstack.end()){
            cout << "New Threshlod ist already available Threshlod vector. " << endl;
            repeat_thres = true;
            break;
        }
        
        cout<< "Threshold: " << seedclass.differencegrow <<endl;
        
    }
    
    seedclass.RGThreshold.resize(100);
    seedclass.LoopThreshold = 1;
    seedclass.data[3].push_back(1.0); // scale
    seedclass.data[4].push_back(0.01); // ScaleDifference
    seedclass.data[0].push_back(CTest.EWlong);    // Green line. long axis
    seedclass.data[1].push_back(CTest.EWshort);   // lightly blue line . short axis
    seedclass.data[2].push_back(CTest.Ratio);
    seedclass.data[5].push_back(0.0); // RatioDifference
    seedclass.data[6].push_back(CTest.Area);  // Area
    seedclass.data[7].push_back(1.0);  // scaleArea
    seedclass.initialseedvektor.clear();
    //s[i].initialseedvektor.push_back(R[i].regioncenter);
    seedclass.initialseedvektor.push_back(CTest.cntr);
    seedclass.threshold_notchange = true;
    
    waitKey(3000);
    destroyWindow("Contour of Segment");
    
    
}

