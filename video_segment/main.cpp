//
//  main.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 21.09.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <time.h> //time_t time()  clock_t clock()
#include <math.h>
#include <float.h>
#include "fstream"
#include <string>
#include <list>
#include <vector>
#include <limits>  // numeric_limits<double>::epsilon(): 2.22045e-016
#include <numeric>

#include <sys/types.h> // mkdir
#include <sys/stat.h> // mkdir
#include <unistd.h> // 判断文件夹是否存在
#include <dirent.h> //rmdir

//#include <errno.h>
//#include <stdlib.h>
//#include <stdarg.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include "opencv2/features2d/features2d.hpp"
#include <opencv2/ximgproc.hpp>
#include "opencv2/surface_matching/icp.hpp"

#include "INITIALSEED.hpp"
#include "Regiongrowing.hpp"
#include "Opticalflow.hpp"
//#include "COUNTER.hpp"
#include "Function.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc;
using namespace ppf_match_3d;
using namespace cv::flann;
//-------global variable

int FPSfacotr = 7 ;
int pathlength = 100;
int strmPricision = 6;
// region growing
int considerNum = 7;
int multipleScaleDiff = 6;
double Areadifferencefactor = 0.2 ;

double confidenceintervalfactor = 1.04 ;
int Loopiterationmax = 15;
double thresholdstep = 0.1;
double touchbordernum = 10;

Size kernelsize(3,3);
#define EPSILON 1e-13   // arrcuracy value
vector<Point2f> REvector_mark;
clock_t  clockBegin, clockEnd;

// Opical flow
vector<vector<Point2f> > OPpoints(2);    // point0为特征点的原来位置，point1为特征点的新位置  point0:the points which his flow needs to be found.  point1: calculated new positions of input features in the second image.
vector<Point2f> OPinitial;    // 初始化跟踪点的位置

// Ground Truth
vector<vector<Point2f> > GTrelationpairs(2);
int ReGT_fq = 300;

// Superpixel
vector<Vec3b> fillcolor;

//----- global function
Mat putStats(vector<string> stats, Mat frame,Vec3b color,Point* origin, char word);
void on_MouseHandle(int event, int x, int y, int flags, void* param);
double pixeldistance(Mat img, Point2f p1, Point2f p2);

Mat Featurematch(Mat preframe, Mat nextframe, vector<Point2f>& obj_last, vector<Point2f>& obj_next);
Mat Spixel(Mat inputmat, int& number_slic,  vector<Vec3b>& colorSP);
double flann_KDtree(vector<Point> dest_vector, vector<Point> obj_vector);
void createDir(string path);

// window name
char const *windowNameRG = "Region growing ";
char const *windowNameOP = "Optical flow";
char const *windowNameGT = "Ground truth";
char const *Pixel_realtion_window = "Pixel-distance relation";
char const *windowNameSP= "Superpixel" ;
char const *windowName3M= "Three methods" ;
//#define Pixel_realtion_window "Pixel-distance relation"
//#define windowNameRG String(XXX) //播放窗口名称

//Trackbar videocapture
char const *trackBarName="Frame index";    //trackbar控制条名称
double totalFrame=1.0;     //视频总帧数
//double currentFrame=1.0;    //当前播放帧
int trackbarValue=1;    //trackbar控制量
int trackbarMax=200;   //trackbar控制条最大值
double frameRate = 0.1;  //视频帧率
int controlRate = 1;

VideoCapture vc;

void TrackBarFunc(int ,void(*))
{
    controlRate=((double)trackbarValue/trackbarMax)*totalFrame; //trackbar控制条对视频播放进度的控制
    vc.set(CV_CAP_PROP_POS_FRAMES,controlRate);   //设置当前播放帧
}

// Trackbar Resize image
char const *RSBarName="Scale factor";
int const RStrackbarMax = 30;
int RStrackbarValue = 0;
double RSscale = 1.0;

void RSTrackBarFunc(int ,void(*))
{
    RSscale= 1+ (RStrackbarValue * (0.7/RStrackbarMax) ) ;
}

int main(int argc, char *argv[] )
{
    
    //70_20_descend"
    vector<vector<Point>> defaultseed(4);
    defaultseed[0].push_back(Point(515,376)); // black roof 有缺陷 在中间  threshold 5
    defaultseed[3].push_back(Point(781,379)); // grass  threshold 4
    defaultseed[1].push_back(Point(215,240)); // light blue roof left
    defaultseed[2].push_back(Point(530,234)); // white boot
    //defaultseed[3].push_back(Point(491,356)); // black roof bottem threshold: 8
    double defaultThreshold[] = {7, 11, 12 ,4} ;
    
//    //skew_descend
//    vector<vector<Point>> defaultseed(4);
//    defaultseed[0].push_back(Point(449,236)); // swimming pool 有缺陷 在中间  threshold 8
//    //defaultseed[0].push_back(Point(781,379)); // grass  threshold 4
//    defaultseed[1].push_back(Point(215,240)); // light blue roof left
//    defaultseed[2].push_back(Point(530,234)); // white boot
//    defaultseed[3].push_back(Point(491,356)); // black roof bottem
//    double defaultThreshold[] = {8, 11, 12 ,8} ;
    
///--------- VideoCapture ----------------
    
    //VideoCapture vc;
    //vc.open( "/Users/yanbo/Desktop/source/Rotation_50m.mp4");
    const char path[] = "/Users/yanbo/Desktop/source/" ;
    //string videofilename = "skew_descend";
    string videofilename = "70_20_descend";
    //string videofilename = "swiss_rotation";
    
    string videoInputpath;
    videoInputpath.assign(path);  videoInputpath.append(videofilename);   videoInputpath.append(".mp4");
    
    vc.open(videoInputpath);
    
    if (!vc.isOpened()){
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
    
//---------- VideoWriter----------------

    //char *savePath = new char;
    char savePath[pathlength] ;
    strcpy(savePath,path);
    strcat(savePath,"output/");  strcat(savePath, videofilename.c_str());
    
    string outputindex = "_output_8";
    strcat(savePath, outputindex.c_str());     // savePath::  /Users/yanbo/Desktop/source/output/70_20_descend_output
    cout<< "savePath: " << savePath <<endl;
    
    remove_directory(savePath); mkdir(savePath, 0777);
    
    strcat(savePath,"/");strcat(savePath, videofilename.c_str()); strcat(savePath, outputindex.c_str());
    cout<< "savePath: " << savePath <<endl;
    
    // Region growing
    char savePathvideoRG[pathlength] ;
    strcpy(savePathvideoRG,savePath);  strcat(savePathvideoRG, "_RG.mov"); // 这句换前后 path 内的值会变为 NUll 如果用 char *savePath = new char[50] 以及 char savePath[50], 因为char 的长度问题
    cout<< "savePathvideoRG: " << savePathvideoRG <<endl;
    remove(savePathvideoRG);
    
//    if(remove(savePathvideo)==0){ cout<<"Old output video delete successful"<<endl; }
//    else { cout<<"Old output video delete failed"<<endl; }
    
    VideoWriter vwRG; //(filename, fourcc, fps, frameSize[, isColor])
    vwRG.open( savePathvideoRG, // 输出视频文件名
            (int)vc.get( CV_CAP_PROP_FOURCC ),//CV_FOURCC('S', 'V', 'Q', '3'), // //CV_FOURCC('8', 'B', 'P', 'S'), // 也可设为CV_FOURCC_PROMPT，在运行时选取 //fourcc – 4-character code of codec used to compress the frames.
            (double)(vc.get( CV_CAP_PROP_FPS )/FPSfacotr), // 视频帧率
            Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ),
                 (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
            true ); // 是否输出彩色视频
    
    if (!vwRG.isOpened()){
        cout << "Failed to write the video! \n" << endl;
        return 1;
    }
    
    //Three method
    char savePathvideo3m[pathlength];
    //strcat(savePath, outputindex.c_str());
    strcpy(savePathvideo3m,savePath);   strcat(savePathvideo3m,"_3M.mov");
    remove(savePathvideo3m);
    cout<< "savePathvideo3m: " << savePathvideo3m <<endl;
    VideoWriter vw3M;
    vw3M.open( savePathvideo3m, (int)vc.get(CV_CAP_PROP_FOURCC), (double)(vc.get(CV_CAP_PROP_FPS)/FPSfacotr), Size( (int)vc.get(CV_CAP_PROP_FRAME_WIDTH), (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT) ), true );
    
    //optical tracking
    char savePathvideoOP[pathlength];
    strcpy(savePathvideoOP,savePath);   strcat(savePathvideoOP,"_OP.mov");
    remove(savePathvideoOP);
    cout<< "savePathvideoOP: " << savePathvideoOP <<endl;
    VideoWriter vwop;
    vwop.open( savePathvideoOP, (int)vc.get(CV_CAP_PROP_FOURCC), (double)(vc.get(CV_CAP_PROP_FPS)/FPSfacotr), Size( (int)vc.get(CV_CAP_PROP_FRAME_WIDTH), (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT) ), true );
    
    //Superpixel
    char savePathvideoSP[pathlength];
    strcpy(savePathvideoSP,savePath);   strcat(savePathvideoSP,"_SP.mov");
    remove(savePathvideoSP);
    cout<< "savePathvideoSP: " << savePathvideoSP <<endl;
    VideoWriter vwsp;
    vwsp.open( savePathvideoSP, (int)vc.get(CV_CAP_PROP_FOURCC), (double)(vc.get(CV_CAP_PROP_FPS)/FPSfacotr), Size( (int)vc.get(CV_CAP_PROP_FRAME_WIDTH), (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT) ), true );
    
    // ground truth
    char savePathvideoGT[pathlength]  ;
    strcpy(savePathvideoGT,savePath);   strcat(savePathvideoGT,"_GT.mov");
    remove(savePathvideoGT);
    cout<< "savePathvideoGT: " << savePathvideoGT <<endl;
    VideoWriter vwgt;
    vwgt.open( savePathvideoGT, (int)vc.get(CV_CAP_PROP_FOURCC), (double)(vc.get(CV_CAP_PROP_FPS)/FPSfacotr), Size( (int)vc.get(CV_CAP_PROP_FRAME_WIDTH), (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT) ), true );
    
    // Feature match (show FM Point)
    char savePathvideoFM[pathlength]  ;
    strcpy(savePathvideoFM,savePath);   strcat(savePathvideoFM,"_FM.mov");
    remove(savePathvideoFM);
    cout<< "savePathvideoFM: " << savePathvideoFM <<endl;
    VideoWriter vwfm;
    vwfm.open( savePathvideoFM, (int)vc.get(CV_CAP_PROP_FOURCC), (double)(vc.get(CV_CAP_PROP_FPS)/FPSfacotr), Size( (int)vc.get(CV_CAP_PROP_FRAME_WIDTH), (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT) ), true );
 
    
//------------Proprocessing-----------
    
    Mat firstFrame;
    int fitstframeindex = vc.get(CV_CAP_PROP_POS_FRAMES);
    printf("\n---------------------------IndexFrame: %d -----------------------------------\n\n", fitstframeindex);
    vc.read(firstFrame);
    Mat frame = firstFrame.clone();
    Mat firstFrame_blur = firstFrame.clone(); GaussianBlur(firstFrame, firstFrame_blur, kernelsize,0,0);
    Mat preFrame ;//= firstFrame.clone();
    
    vector<Point> Allcontourvector_Pre;
    //Mat preallsegment(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
        //vc.set(CV_CAP_PROP_POS_FRAMES, 1);
    
//----------Setting the relation between real distrance and pixel distance
    Mat firstFrameResize = firstFrame.clone();
    //firstFrame.copyTo(firstframeBackup);
    
    cout<<"Setting the relation between real distrance and pixel distance"<<endl << "Please mark points on the image"<<endl;
    cout<<"1: using default points, 2: clicking points manually" <<endl;
    int RelationMode;
    cin >> RelationMode;
    
    switch (RelationMode) {
        case 1:
            // 70_20_descend
            //    REvector_mark.push_back(Point(272,437));   REvector_mark.push_back(Point(329,335)); // left small dark roof
            REvector_mark.push_back(Point(557,417));  REvector_mark.push_back(Point(462,323)); // middle big dark roof
            //    REvector_mark.push_back(Point(584,222));   REvector_mark.push_back(Point(497,268)); // ??
          break;
            
        case 2:
            while(true){
                static double RSscale_pre;
                if (RSscale != RSscale_pre){
                    resize(firstFrame, firstFrameResize, Size(), RSscale, RSscale, INTER_CUBIC);
                    RSscale_pre = RSscale;
                }
                
                //int A  = getTrackbarPos(RSBarName, Pixel_realtion_window);
                createTrackbar(RSBarName, Pixel_realtion_window, &RStrackbarValue, RStrackbarMax, RSTrackBarFunc);
                RSTrackBarFunc(RStrackbarValue, 0);
                namedWindow( Pixel_realtion_window );
                setMouseCallback(Pixel_realtion_window,on_MouseHandle,(void*)&firstFrameResize);
                imshow( Pixel_realtion_window, firstFrameResize );
                if( waitKey( 10 ) == 13 ){
                    destroyWindow(Pixel_realtion_window);
                    break;
                }
            }
            
            if ( REvector_mark.size() == 0 || REvector_mark.size()%2 != 0 ){
                cout<< "\n!!!!!!!You did not mark Mulitiple of 2 points. Porgramm breaks"  <<endl;
                return 0;
            }
            
                //    namedWindow( Pixel_realtion_window );
                //    setMouseCallback(Pixel_realtion_window,on_MouseHandle,(void*)&firstframeBackup);
                //
                //    while(true){
                //        imshow( Pixel_realtion_window, firstframeBackup );
                //        if( waitKey( 10 ) == 13 ) break;//按下enter键，程序退出
                //    }
                //
                //    if ( REvector_mark.size() == 0 || REvector_mark.size()%2 != 0 ){
                //        cout<< "\n!!!!!!!You did not mark Mulitiple of 2 points. Porgramm breaks"  <<endl;
                //        return 0;
                //    }
            break;
            
        default:
            break;
    }

    
    GTrelationpairs[0].assign(REvector_mark.begin(), REvector_mark.end()); // for Ground truth
    
    int relationnumber  = (int)REvector_mark.size()/2;
    //double*pixeld=(double*)malloc(relationnumber*sizeof(double));
    double distance[relationnumber]; double pixeld[relationnumber]; // 这两个 GT 中还要用到
    double initialpixelrelation = 0.0 ;
    Mat firstFrame_showline = firstFrame.clone();
    for (int i = 0; i < relationnumber; i++){
        cout<< "Line Number: " << i+1 << endl;
        pixeld[i] = pixeldistance(firstFrame_showline, REvector_mark[2*i], REvector_mark[2*i+1]);
        imshow( Pixel_realtion_window , firstFrame_showline );
        waitKey(1000);
        cout<< "Pixeldistance: " << pixeld[i] <<endl;
        cout<< "Please input the real distance (m) for this line"<<endl;
        //cin >> distance[i];
        distance[i] = 10;
        initialpixelrelation += ( distance[i]/ pixeld[i] );
    }
    initialpixelrelation /= relationnumber;
    cout<< "The relation: " << initialpixelrelation << " m/pixel \n" <<endl;
    
    destroyWindow(Pixel_realtion_window); //destroyAllWindows();

    //--------find seed point for RG
    cout<<"How many initial segment do you want: " <<endl;
    int Segmentinitialnum;
    //cin >> Segmentinitialnum;
    Segmentinitialnum  = 2;
    
    Initialseed s[Segmentinitialnum];
    vector<Initialseed>  vectorS;
    //deque<Initialseed>  vectorS;
    
    // check weather input threshold is qualified or not
    int i =0;
    while( i < Segmentinitialnum)
    {
        printf("\n********************Setting for object %d **********\n", i+1);
        cout<<"Plaese choose method. \n tap 1, choose seeds by clicking in orignal image. \n tap 2, choose seeds by default position of points"<<endl;
        int mode;
        //cin >> mode;
        mode =2;
        s[i] = Initialseed(mode, firstFrame, i, defaultThreshold, defaultseed);
        
        if(s[i].checkThreshold(firstFrame_blur, initialpixelrelation)){
            vectorS.push_back( s[i] );
            Allcontourvector_Pre.insert(Allcontourvector_Pre.end(), s[i].Contourvector.begin(), s[i].Contourvector.end());
            //cout<< "Allcontourvector_Pre.size: " <<Allcontourvector_Pre.size() << endl;
            i++;
        }
    }

//------------TXT generation
    // savePathtxt = "/Users/yanbo/Desktop/source/txt/~.txt";
    string Pathtxt;
    Pathtxt.assign(path);  Pathtxt.append("txt/");  Pathtxt.append(videofilename); Pathtxt.append(outputindex); //Pathtxt.append("/");
    remove_directory(Pathtxt.c_str());  mkdir(Pathtxt.c_str(), 0777);

    Pathtxt.append("/"); Pathtxt.append(videofilename); Pathtxt.append(outputindex);
    cout<< "Pathtxt: " << Pathtxt<<endl;
    
    //----txt: the inital seed point before RG
    string saveInitaltxt;
    saveInitaltxt.assign(Pathtxt);  saveInitaltxt.append("_initial.txt");
    
    ofstream initialstream;
    initialstream.open(saveInitaltxt,ios::out|ios::trunc);

    if (!initialstream.is_open()){
        cout << "Failed to open outputtext file! \n" << endl;
        return 1;
    }
    
    initialstream << "Pixel-real distance relation: " << endl ;
    for(size_t i=0; i< REvector_mark.size(); i++){
        initialstream << "Row: " << REvector_mark[i].y << " Column: "  << REvector_mark[i].x << endl;
        
        if ( (i+1)%2 == 0){
             initialstream << "Pixeldistance: " << pixeld[i] << " Real distance: " << distance[i] <<endl;
        }
    }
    
    initialstream << "The relation: " << initialpixelrelation << " m/pixel \n" << endl;
    REvector_mark.clear();
    
    for( int i=0; i< Segmentinitialnum; i++){
        initialstream << "Objekt: " << i+1 << endl << "Initial Threshold: " << s[i].differencegrow << endl << "Row  Column " << endl;
        for(size_t j=0;j<s[i].initialseedvektor.size();j++){
            initialstream << s[i].initialseedvektor[j].y << "  " << s[i].initialseedvektor[j].x << endl ;
        }
        initialstream << "\n";
    }
    
    initialstream.close(); //initialstream << flush;

    //----txt:  Scale value per frame
    string saveScaletxt;
    saveScaletxt.assign(Pathtxt);  saveScaletxt.append("_SingleScale.txt");
    
    ofstream Scalestream;  // Scalestream 在 ！stop 循环中 第一个循环还可以打开， 第二个循环这个ofstream文件就不能打开了
    Scalestream.open(saveScaletxt,ios::out|ios::trunc);Scalestream.setf(ios::fixed, ios::floatfield); Scalestream.precision(strmPricision);
    if (!Scalestream.is_open()){
        cout << "Failed to open Scale txt file! \n" << endl;
        return 1;
    }
    Scalestream << videofilename << outputindex << endl << "Scale per frame" << endl;  Scalestream << "Frameindex  FM  OP  RG  ICP  GT" << endl;
    Scalestream << fitstframeindex << " " <<1.0 << " " << 1.0 << " " << 1.0 << " " << 1.0 << " " << 1.0 <<endl;
    
    //----txt:  accumulated Scale value
    string saveACScaletxt;
    saveACScaletxt.assign(Pathtxt);  saveACScaletxt.append("_AccumScale.txt");
    
    ofstream AccumScalestream;
    AccumScalestream.open(saveACScaletxt,ios::out|ios::trunc); AccumScalestream.setf(ios::fixed, ios::floatfield); AccumScalestream.precision(strmPricision);
    AccumScalestream << videofilename << outputindex << endl << "Accumulated scale value" << endl << "Frameindex  FM  OP  RG  ICP  GT" << endl;
    AccumScalestream << fitstframeindex << " " <<1.0 << " " << 1.0 << " " << 1.0 << " " << 1.0 << " " << 1.0 <<endl;
    //---txt:  Total time
    string saveACTimetxt;
    saveACTimetxt.assign(Pathtxt);  saveACTimetxt.append("_TotalTime.txt");
    
    ofstream TotalTimestream;
    TotalTimestream.open(saveACTimetxt,ios::out|ios::trunc);
    TotalTimestream << videofilename << outputindex << endl << "Total execution time" << endl << "Frameindex  FM  OP  RG" << endl;
    
    //---txt:  time per frame
    string saveSingleTimetxt;
    saveSingleTimetxt.assign(Pathtxt);  saveSingleTimetxt.append("_SingleTime.txt");
    
    ofstream SingleTimestream;
    SingleTimestream.open(saveSingleTimetxt,ios::out|ios::trunc);
    SingleTimestream << videofilename << outputindex << endl << "Execution time per frame" << endl << "Frameindex  FM  OP  RG" << endl;
    
    //----txt: pixel-realdistance relation
    string saverealtiontxt;
    saverealtiontxt.assign(Pathtxt);  saverealtiontxt.append("_relation.txt");
    
    ofstream Relationstream;
    Relationstream.open(saverealtiontxt,ios::out|ios::trunc);  Relationstream.setf(ios::fixed, ios::floatfield); Relationstream.precision(strmPricision);
    Relationstream << videofilename << outputindex << endl << "Pixel-real Distance-Porprotion per frame" << endl << "Frameindex  FM  OP  RG  ICP  GT" << endl;
    Relationstream << fitstframeindex << " " << initialpixelrelation << " " << initialpixelrelation << " " << initialpixelrelation << " " << initialpixelrelation << " "<< initialpixelrelation <<endl;
    
    //----txt: Ground Truth info
    string saveGTtxt;
    saveGTtxt.assign(Pathtxt);  saveGTtxt.append("_GT.txt");
    
    ofstream GTstream;
    GTstream.open(saveGTtxt,ios::out|ios::trunc);  GTstream.setf(ios::fixed, ios::floatfield); GTstream.precision(strmPricision);
    GTstream << videofilename << outputindex << endl << "Ground truth" << endl << "Frameindex  SingleScale  AccumScale  Pixel-Relation" << endl;
    GTstream << fitstframeindex << " " << 1.0 << " " << 1.0 << " " << initialpixelrelation <<endl;
    
//-------- mkdir for pictures in ICP
    string Pathcounter;
    Pathcounter.assign(path); Pathcounter.append("picture/ICPcounter/"); Pathcounter.append(videofilename); Pathcounter.append(outputindex);
    remove_directory(Pathcounter.c_str());  mkdir(Pathcounter.c_str(), 0777);
    Pathcounter.append("/");
    
//-------- creating Trackbar
    totalFrame = vc.get(CV_CAP_PROP_FRAME_COUNT);  //获取总帧数
    frameRate = vc.get(CV_CAP_PROP_FPS);   //获取帧率
    double pauseTime=1000/frameRate; // 由帧率计算两幅图像间隔时间
    namedWindow(windowNameRG, WINDOW_NORMAL);
    //在图像窗口上创建控制条
    createTrackbar(trackBarName,windowNameRG,&trackbarValue,trackbarMax,TrackBarFunc);
    //TrackBarFunc(0,0);
    
//--------------video run
    
    bool stop(false);
    bool all_threshold_notchange(true);
    Mat frame_backup;
    int indexFrame;
    bool bSuccess;
    
    double totoalRGtime = 0, totoalOPtime = 0, totoalFMtime = 0;
    double AccumscaleRG = 1.0, AccumscaleOP = 1.0, AccumscaleFM = 1.0, AccumscaleICP = 1.0, AccumscaleGT = 1.0;
    clock_t start_RG, end_RG, start_OP, end_OP, start_FM, end_FM;
    
    double relationRG = initialpixelrelation , relationOP = initialpixelrelation, relationFM = initialpixelrelation,  relationICP = initialpixelrelation, relationGT_pre = initialpixelrelation, relationGT = 0.0;
    double scaleGT = 1.0;
    
    while(!stop)
    {
        //Mat frame;

        int Segmentnum = (int)vectorS.size();
        
        if (all_threshold_notchange){    // if threshold for this frame did not change, read the next frame
            
            preFrame = frame.clone();
            indexFrame = vc.get(CV_CAP_PROP_POS_FRAMES);
            printf("\n---------------------------IndexFrame: %d -----------------------------------\n\n", indexFrame);
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
            printf("\n---------------------------IndexFrame: %d -----------------------------------\n\n", indexFrame);

            bSuccess = vc.read(frame);
        }
        
        if (frame.empty()){
            cout << "Video play over/ in loop" <<endl;
            //waitKey(0);
            break;
        }

        if (!bSuccess){ //if not success, break loop
            cout << "ERROR: Cannot read a frame from video" << endl;
            break;
        }

//----------------- show the text to written video frame
        
        vector<string> text; // RG: frameindex totaltime relation
        vector<string> seedtext;  // RG per frame: center/intensity/Sclae/real area
        vector<string> Thresholdtext; // RG
        vector<string> scaletext; // three methods: frame index/ scale/ accumlate scale  || Scalechar Acscalechar
        vector<string> timetext; // three methods: single time/ total time/ pixel-relation  || Timechar Totoaltimechar pixelrelachar
        vector<string> optext; // Optical flow:  Scale/ frame index / time
        vector<string> FMtext; // Feature matching:  Scale/ frame index / time
        vector<string> GTrelationtext; // Ground truth: pixel relation / frame index
        vector<string> SPtext; // Superpixel: frame index/ superpixel number
        
        char Buffer[60];
        
        //以下都为了3M
        char Scalechar[70];
        char Timechar[70];
        char Totoaltimechar[70];
        char Acscalechar[70];
        char pixelrelachar[70];
        
        Point ptTopLeft(10, 10); // RG: seedtext
        Point* ptrTopLeft = &ptTopLeft;
        
        Point ptTopright(frame.cols-10, 10); // RG: text
        Point* ptrTopright = &ptTopright;
        
        Point ptBottomMiddle_RG(frame.cols/2, frame.rows); //  RG: Thresholdtext
        Point* ptrBottomMiddle_RG = &ptBottomMiddle_RG;
        
        Point ptTopLeft2(10, 10);     // 3m: scaletext
        Point* ptrTopLeft2 = &ptTopLeft2;
        
        Point ptBottomMiddle_3M(frame.cols/2, frame.rows); // 3M:  timetext
        Point* ptrBottomMiddle_3M = &ptBottomMiddle_3M;
        
        Point ptBottomMiddle_GT(frame.cols/2, frame.rows);  // GT: GTrelationtext
        Point* ptrBottomMiddle_GT = &ptBottomMiddle_GT;
        
        Point ptBottomMiddle_OP(frame.cols/2, frame.rows);  // OP: optext
        Point* ptrBottomMiddle_OP = &ptBottomMiddle_OP;
        
        Point ptBottomMiddle_SP(frame.cols/2, frame.rows); // SP: SPtext
        Point* ptrBottomMiddle_SP = &ptBottomMiddle_SP;
        
        Point ptBottomMiddle_FM(frame.cols/2, frame.rows); // FM: FMtext
        Point* ptrBottomMiddle_FM = &ptBottomMiddle_FM;
        
// --------  Ground turth for pixel-distance relation
        cout<< "$$ Ground turth for pixel-distance relation" <<endl;
        Mat GToutput = frame.clone();
        Opticalflow GT;
        
        if(all_threshold_notchange){
            relationGT = 0.0;
            scaleGT = 0.0;
            
                if (indexFrame!=0 &&  indexFrame%ReGT_fq == 0){
                //if (indexFrame!=0){
                    Mat frameBackup;
                    frame.copyTo(frameBackup);
                    cout<< "Input points again for update pixel-distance relation" <<endl;
                    
                    double RSscale_pre =0;
                    while(true)
                    {
                        //static double RSscale_pre;
                        if (RSscale != RSscale_pre){
                            resize(frame, frameBackup, Size(), RSscale, RSscale, INTER_CUBIC);
                            RSscale_pre = RSscale;
                        }
                        
                        createTrackbar(RSBarName, Pixel_realtion_window, &RStrackbarValue, RStrackbarMax, RSTrackBarFunc);
                        RSTrackBarFunc(RStrackbarValue, 0);
                        namedWindow( Pixel_realtion_window );
                        setMouseCallback(Pixel_realtion_window, on_MouseHandle, (void*)&frameBackup);
                        imshow( Pixel_realtion_window, frameBackup );
                        if( waitKey( 10 ) == 13 ) break;//按下enter键，程序退出

                    }
                    
                    if ( REvector_mark.size() == 0 || REvector_mark.size()%2 != 0 ){
                        cout<< "\n!!!!!!!You did not mark Mulitiple of 2 points. Porgramm breaks"  <<endl;
                        return 0;
                    }
                    
                    GTrelationpairs[1].assign(REvector_mark.begin(), REvector_mark.end()); // for Ground truth
                    destroyWindow(Pixel_realtion_window);
                    REvector_mark.clear();
                }
            
                else GT.relationtrack(preFrame, frame, GTrelationpairs) ;
      
            // update the pixel relation value
            int newrelationnumber  = (int)GTrelationpairs[1].size()/2;
            double newpixeld[newrelationnumber];
            
            for (int i = 0; i < newrelationnumber; i++){
                newpixeld[i] = pixeldistance(GToutput, GTrelationpairs[1][2*i], GTrelationpairs[1][2*i +1]);
                relationGT += (distance[i]/ newpixeld[i]) ;
            }
            
            relationGT /= newrelationnumber;
            GTrelationpairs[0].assign(GTrelationpairs[1].begin(), GTrelationpairs[1].end());
            cout<< "GT relation: " << relationGT << "m/pixel" << endl;
            
            scaleGT = relationGT_pre/relationGT;  cout<< "scaleGT: " << scaleGT  << endl;

            sprintf( Buffer, "Frame: %d/ GT relation: %.5fm/p / Scale: %.5f", indexFrame, relationGT, scaleGT);  GTrelationtext.push_back(Buffer);
            GToutput = putStats(GTrelationtext, GToutput, Vec3b(0,0,190), ptrBottomMiddle_GT, 'b' );
            
            moveWindow(windowNameGT, 100, 450);
            //imshow(windowNameGT, GToutput);  //显示图像
            vwgt << GToutput;
        }
        
        cout<<endl;

// --------  Feature detection and classic matching method
        start_FM = clock();
        cout<< "$$ Feature detection and matching" <<endl;
        
        Mat FMoutput;
        vector<Point2f> obj_last;   vector<Point2f> obj_next;
        FMoutput = Featurematch(preFrame, frame, obj_last, obj_next);
        
        Mat H_Featurematch = findHomography( obj_last, obj_next ,CV_RANSAC );
        //cout<<"Homography matrix:" <<endl << H_Featurematch <<endl << endl;
        double scaleFM = decomposematrix(H_Featurematch);
        cout<< "Scale (Freaturematch):" << scaleFM << endl;
        end_FM = clock();
        
        if(all_threshold_notchange){
            
            sprintf( Buffer, "Frame: %d/Scale: %.7f/Interframe Time: %.5f", indexFrame, scaleFM, (double)(end_FM - start_FM) / CLOCKS_PER_SEC);
            FMtext.push_back(Buffer);
            FMoutput = putStats(FMtext, FMoutput, Vec3b(0,0,170), ptrBottomMiddle_FM, 'b' );
            
            //imshow("Feature matchig", FMoutput);  //显示图像
            vwfm << FMoutput;
        }
        
        cout<<endl;
        
// --------  optical flow
        start_OP = clock();
        
        cout<< "$$ Optical flow" <<endl;
        Mat OPoutput = frame.clone();
        
        Opticalflow OP;
        
        vector<vector<Point2f> > matchingpairs(2);
        OP.matchedpairs(preFrame, frame, matchingpairs);
        
        Mat H_OP = findHomography( matchingpairs[0], matchingpairs[1] ,CV_RANSAC );
        double scaleOP = decomposematrix(H_OP);
        cout<< "Scale (optical flow):" << scaleOP << endl;
        
        end_OP = clock();
        
        if(all_threshold_notchange){
            OP.trackpath(preFrame , frame, OPoutput, OPpoints, OPinitial);
            swap(OPpoints[1], OPpoints[0]);
            
            sprintf( Buffer, "Frame: %d/Scale: %.7f/Interframe Time: %.5f", indexFrame, scaleOP, (double)(end_OP - start_OP) / CLOCKS_PER_SEC);
            optext.push_back(Buffer);
            OPoutput = putStats(optext, OPoutput, Vec3b(0,0,170), ptrBottomMiddle_OP, 'b' );
            
            moveWindow(windowNameOP, 400, 500); // int x = column, int y= row
            //imshow(windowNameOP, OPoutput);  //显示图像
            vwop << OPoutput;
        }
        
        cout<<endl;
        
// -----------------Superpixel SLIC
        cout<< "$$ Superpixel" <<endl;
        
        if(all_threshold_notchange){
            int SP_num;
            Mat SPoutput = Spixel(frame, SP_num ,fillcolor);
            
            sprintf( Buffer, "Frame: %d/ Superpixel number %d", indexFrame, SP_num);
            SPtext.push_back(Buffer);
            SPoutput = putStats(SPtext, SPoutput, Vec3b(0,0,200), ptrBottomMiddle_SP, 'b' );
            
            imshow( windowNameSP, SPoutput);
            vwsp << SPoutput;
        }
        
        cout<<endl;
// -----------------Region growing--------------
        start_RG = clock();
        cout<< "$$ Region growing" ;
        vector<double> templateScaleSameframe;
        
        Mat frame_Blur = frame.clone();
        GaussianBlur(frame, frame_Blur, kernelsize,0,0);
        //blur( image, out, Size(3, 3));
        
        Regiongrowing R[Segmentnum];
        Mat Matsegment= frame.clone(); // 原图+segment mask with spiecal color
        Mat FramewithCounter = frame.clone(); // segment 的 counter 输出
        Mat framethreemethode = frame.clone();
        Mat MatSegment; // Black image + segment with orignla color 
        Mat MatCounter;
        Mat Matallsegment(frame.size(),CV_8UC3,Scalar(0,0,0));
        
            for( int i=0; i<Segmentnum; i++)
            {

                printf("\n*** Objekt %d Information ****************", i+1);
                printf("\n*** Cyele index for Threshold: %d\n", vectorS[i].LoopThreshold);
                MatSegment = R[i].RegionGrow(frame, frame_Blur , vectorS[i].differencegrow, vectorS[i].initialseedvektor);
                
                MatCounter  = R[i].Matcounter(MatSegment, vectorS[i].color);
                
//                ICP icp(100, 0.05f, 2.5f, 8);
//                double residual; Matx44d pose;
//                vector<Pose3DPtr> resultsSub;
//                icp.registerModelToScene(vectorS[i].preCounter, counter, resultsSub);
//                //icp.registerModelToScene(vectorS[i].preCounter, counter, residual, pose);
//                cout<< resultsSub.size()<< endl;
//                for (size_t i=0; i<resultsSub.size(); i++)
//                {
//                    Pose3DPtr result = resultsSub[i];
//                    cout << "Pose Result " << i << endl;
//                }
                //cout << pose << endl;
                
             // If true, the function finds an optimal affine transformation with no additional restrictions (6 degrees of freedom). Otherwise, the class of transformations to choose from is limited to combinations of translation, rotation, and uniform scaling (5 degrees of freedom).
//                Mat Affine = estimateRigidTransform(vectorS[i].preSegment,MatSegment,false);  // estimateRigidTransform 有可能得到的 matrix 为0矩阵
//                //cout<<"Affine:" << endl << Affine<<endl;
//                double scaleRG_Affine = decomposematrix(Affine);
//                cout<< "scale RG Affine:" << scaleRG_Affine << endl;
                
                addWeighted(Matallsegment, 1, MatSegment, 1, 0.0, Matallsegment);

                FramewithCounter = R[i].FindCounter(MatSegment, FramewithCounter, vectorS[i].color);
            
                //double scale = ( (C[i].EWlong/vectorS[i].data[0].back()) + (C[i].EWshort/vectorS[i].data[1].back()) )/2 ;
                double scale = sqrt( ( R[i].EWlong* R[i].EWshort)/(vectorS[i].data[0].back() * vectorS[i].data[1].back()) );
                //double scale = sqrt( R[i].Area / vectorS[i].data[6].back() );

                double Realarea = R[i].Area * relationRG * relationRG;
                double scaleRealarea =  Realarea / vectorS[i].data[5].back() ;
                
                //cout << "EWlong: " << R[i].EWlong<< endl;
                //cout << "EWshort: " << R[i].EWshort<< endl;
                //cout << "Ratio: "  << C[i].Ratio <<endl;
                cout << "centre: "  << R[i].cntr <<endl;
                //cout << "Rectangle width: "  << C[i].rectanglewidth <<endl;
                //cout << "Rectangle width * pixelrelation : "  << C[i].rectanglewidth  * pixelrelation <<endl;
                //cout << "Rectangle height: "  << C[i].rectangleheight <<endl;
                printf("Threshold: %.8f \n", vectorS[i].differencegrow );
                cout << "Object area: "  << R[i].Area << endl;
                cout << "Real area: "  << Realarea << endl;
                cout<< "Scale (index " << indexFrame << " to " << indexFrame-1<< "): " << scale <<endl;
                cout<< "ScaleRealarea (index " << indexFrame << " to " << indexFrame-1<< "): " << scaleRealarea <<endl;
                cout<< endl;
                
                templateScaleSameframe.push_back(scale);
                
                // text infomation show in frame
                double B_Centre = frame.at<Vec3b>(R[i].cntr)[0];
                double G_Centre = frame.at<Vec3b>(R[i].cntr)[1];
                double R_Centre = frame.at<Vec3b>(R[i].cntr)[2];
                
                sprintf(Buffer, "%d: (%d,%d)/intensity: %.2f/Scale: %.5f / Real Area: %.2f", i+1, R[i].cntr.y, R[i].cntr.x, (B_Centre + G_Centre + R_Centre) /3.0 , scale, Realarea );
                seedtext.push_back(Buffer);
                FramewithCounter = putStats(seedtext, FramewithCounter, vectorS[i].color, ptrTopLeft, 't');
                seedtext.clear();
                
                sprintf(Buffer, "Threshold(obj %d): %.8f", i+1, vectorS[i].differencegrow );
                Thresholdtext.push_back(Buffer);
            
        //---------computeing the average scale and average Scale Difference
                
                double averageScale = averagevalue(considerNum, vectorS[i].data[2]);
                double averageScaleDifference = averagedifference(considerNum, vectorS[i].data[3]);
                double ScaleDifference = scale -  averageScale;
                //                vector<double>::iterator iter;
                //                cout<< "ScaleDifference.size(): " << vectorS[i].data[4].size() << endl;
                //                cout << "ScaleDifference vector = " << endl;;
                //                for (iter= vectorS[i].data[4].begin(); iter != vectorS[i].data[4].end(); iter++)  {cout << *iter << " ";}
                //                cout << endl;
                
                cout << multipleScaleDiff <<"* Average ScaleDifference of previous "<< considerNum <<" frames: " <<multipleScaleDiff * averageScaleDifference<< endl;
                printf("ScaleDifference (index %d to %d): %.8f = （%.8f-%.8f）\n", indexFrame, indexFrame-1, ScaleDifference, scale, averageScale);
                cout<< endl;
                
        // ---------- average Ratio and average Ratio Difference
                
//                double averageRatio = averagevalue(considerNum, vectorS[i].data[2]);
//                double averageRatioDifference = averagedifference(considerNum, vectorS[i].data[5]);
//                double RatioDifference = R[i].Ratio - averageRatio;
//                printf("Ratio Difference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, RatioDifference );
        
        // ----------  real Area
        
                double averageScaleRA = averagevalue(considerNum, vectorS[i].data[6]);
                double averageScaleRADifference = averagedifference(considerNum, vectorS[i].data[7]);
                double ScaleRADifference = scaleRealarea - averageScaleRA;
                
                cout << multipleScaleDiff <<"* Average RealArea Scale Difference of previous "<< considerNum <<" frames: " <<multipleScaleDiff * averageScaleRADifference<< endl;
                printf("Scale RealArea Difference (index %d to %d): %.8f = （%.8f-%.8f）\n", indexFrame, indexFrame-1, ScaleRADifference, scaleRealarea, averageScaleRA);
                cout<< endl;
                
        //------------ Area Difference
//                double Areadiffernce = R[i].Area - vectorS[i].data[4].back();
//                printf("Area Difference (index %d to %d): %.8lf \n", indexFrame, indexFrame-1, Areadiffernce );
//                cout<< Areadifferencefactor << "* Area in last frame : " <<  Areadifferencefactor *(vectorS[i].data[4].back())  << endl ;
                
        //--------- update the thereshlod value for region growing if scale varies largely
        
                double newthreshold;
                //if ( abs(Areadiffernce) <=  Areadifferencefactor *(vectorS[i].data[4].back()) )
                if ( abs(ScaleRADifference) <=  multipleScaleDiff * averageScaleRADifference )
                {
                    //printf("the Area of segment %d is stable (Area difference is smaller 0.2* Area in last frame) \n", i+1);
                    
                    if (abs(ScaleDifference) > multipleScaleDiff * averageScaleDifference ) {
                        
                        if ( ScaleDifference > 0.0 ){
                            newthreshold = vectorS[i].differencegrow - (thresholdstep / pow(2, vectorS[i].LoopThreshold - 1));
                            printf("!!!!!Update/ the threshlod value will be decreased/ Segment is too lagre (scale difference positive) \n");
                        }
                        else{
                            newthreshold = vectorS[i].differencegrow + (thresholdstep / pow(2, vectorS[i].LoopThreshold - 1));
                            printf("!!!!!Update/ the threshlod value will be increased/ Segment is too small (scale difference negative) \n");
                        }

                        printf("New RG_Threshlod: %f \n", newthreshold);
            
                        vector<double>::iterator iterfind;
                        //iterfind = find( vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), newthreshold);
                        iterfind  = find_if(vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), [newthreshold](double b) { return fabs(newthreshold - b) < EPSILON; });
                        
                        if(iterfind == vectorS[i].RGThreshold.end()){
                            vectorS[i].RGThreshold.push_back(newthreshold);
                            vectorS[i].differencegrow = newthreshold;
                            cout << "New RG_Threshlod ist not available in RG_Threshlod vector. Using New RG_Threshlod " << vectorS[i].differencegrow <<" for next loop" << endl;
                        }
                        else {
                            cout << "New RG_Threshlod ist already available in RG_Threshlod vector. Using old RG_Threshlod " << vectorS[i].differencegrow <<" for next loop  " << endl;
                            vectorS[i].LoopThreshold = vectorS[i].LoopThreshold + 1;
                            //cout << "s[i].LoopThreshold: " << s[i].LoopThreshold  << endl;
                            
                            if(vectorS[i].LoopThreshold > Loopiterationmax){
                                //cout<< endl <<"######Delete this segment and Break the video becasue of infinitv Loop" << endl;
                                break;
                            }
                        }
                    
                        //vc.set(CV_CAP_PROP_POS_FRAMES, indexFrame);  // 此处set frame index 会引起死机
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
                                
                                vectorS[i].data[2].push_back(scale);
                                vectorS[i].data[3].push_back(ScaleDifference);
                                
                                vectorS[i].data[4].push_back(R[i].Area);
                                vectorS[i].data[5].push_back(Realarea);
                                
                                vectorS[i].data[6].push_back(scaleRealarea);
                                vectorS[i].data[7].push_back(ScaleRADifference);
                                
                                vectorS[i].preCounter  = MatCounter.clone();
                                vectorS[i].preSegment  = MatSegment.clone();
                                
                                vectorS[i].initialseedvektor.clear();
                                //s[i].initialseedvektor.push_back(R[i].regioncenter);
                                vectorS[i].initialseedvektor.push_back(R[i].cntr);
                                break;
                            }
                    }
                    
                }
                
                
                else { // Area of segment is unstable / abs(Areadiffernce) >  0.2*(vectorS[i].data[6].back()
                    
                    //if (Areadiffernce <0)
                    if (ScaleRADifference <0)
                    {
                    printf("!!!!!Update /Area(negetive) varies too much / segment is too small\n");
                    newthreshold = vectorS[i].differencegrow + (thresholdstep / pow(2, vectorS[i].LoopThreshold - 1));
                    }
                    
                    else {
                    printf("!!!!!Update / Area(positive) varies too much / segment is too large \n");
                    newthreshold = vectorS[i].differencegrow - (thresholdstep / pow(2, vectorS[i].LoopThreshold - 1));
                    }
                    
                    cout<<"New RG_Threshold: " <<  newthreshold << endl;
                  
                    vector<double>::iterator iterfind2;
                    iterfind2  = find_if(vectorS[i].RGThreshold.begin(), vectorS[i].RGThreshold.end(), [newthreshold](double b) { return fabs(newthreshold - b) < EPSILON; });
                    //iterfind2  = find_if(v.begin(),v.end(),dbl_cmp(10.5, 1E-8));
                    
                    if(iterfind2 == vectorS[i].RGThreshold.end()){
                        vectorS[i].RGThreshold.push_back(newthreshold);
                        vectorS[i].differencegrow  = newthreshold;
                        cout << "New RG_Threshlod ist not available in RG_Threshlod vector. Using New RG_Threshlod " << vectorS[i].differencegrow << " for next loop" << endl;
                    }
                    
                    else {
                        cout << "New RG_Threshlod ist already available in RG_Threshlod vector. Using old RG_Threshlod " << vectorS[i].differencegrow << " for next loop  " << endl;
                        vectorS[i].LoopThreshold = vectorS[i].LoopThreshold + 1;
                        
                        if(vectorS[i].LoopThreshold > Loopiterationmax){
                            //cout<< "Delete this segment and Break the video becasue of infinitv Loop" << endl;
                            break;
                        }
                    }
                    
                    //printf("RG_Threshold for next loop %f \n", vectorS[i].differencegrow);
                    
                    vectorS[i].threshold_notchange = false;
                }
                
                cout<<endl;
                
                for(size_t j=0; j<R[i].seedtogether.size(); j++){
                    Matsegment.at<Vec3b>(R[i].seedtogether[j]) = vectorS[i].color;
                    Matallsegment.at<Vec3b>(R[i].seedtogether[j]) = frame.at<Vec3b>(R[i].seedtogether[j]);
                }
                
                cout<< "objekt:" << i+1 << " /Point number which touch border window: " << R[i].touchbordernum << endl;
                
                FramewithCounter = putStats(Thresholdtext, FramewithCounter, vectorS[i].color, ptrBottomMiddle_RG, 'b' );
                Thresholdtext.clear();
                
            }//segment big循环在这里截止
        
        end_RG = clock();
        
        char PathcounterBackup[pathlength];
        strcpy(PathcounterBackup, Pathcounter.c_str());  sprintf(PathcounterBackup + strlen(PathcounterBackup),"Idx_%d.png", indexFrame);
        imwrite(PathcounterBackup,Matallsegment);
        
   //---------- Scale of different Segment in the same Frame to build gaussian model

        cout<< endl << "***Build gaussian model"  << endl;
       
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
        cout<< "mean: " << mean <<  "/ stdev: " << stdev << "/ Confidence Interval: " << mean - confidenceintervalfactor *stdev << " ~ " << mean + confidenceintervalfactor *stdev << endl;
        for (int i = 0 ; i< templateScaleSameframe.size(); i++){
            if (templateScaleSameframe[i] > (mean + confidenceintervalfactor *stdev) || templateScaleSameframe[i] < (mean - confidenceintervalfactor *stdev)) cout<< "the Scale of Segment " << i+1 << " is not qualified"<< endl;
            else cout<< "the Scale of Segment " << i+1 << " is qualified"<< endl;
        }
        cout<<endl;
        
// ---------------ICP
        cout<< "$$ ICP" <<endl;
        vector<Point> Allcontourvector_cur;
        cout<< Allcontourvector_cur.size() << endl;
        for( int i=0; i<Segmentnum; i++){
            Allcontourvector_cur.insert(Allcontourvector_cur.end(), R[i].Contourvector.begin(), R[i].Contourvector.end());
        }
        
        cout<< "Allcontourvector_cur: " << Allcontourvector_cur.size() << " Allcontourvector_Pre: "  << Allcontourvector_Pre.size() << endl;
    
        double scaleICP = flann_KDtree(Allcontourvector_cur, Allcontourvector_Pre);
        
        cout<< "ScaleICP: " << scaleICP <<endl;
//--------------------------
        
        // RG
        sprintf( Buffer, "Frame %d",indexFrame);
        text.push_back(Buffer);
        totoalRGtime +=(double)(end_RG - start_RG) / CLOCKS_PER_SEC; // 在linux系统下，CLOCKS_PER_SEC 是1000000，表示的是微秒。 CLOCKS_PER_SEC，它用来表示一秒钟会有多少个时钟计时单元
        sprintf(Buffer, "%f s", totoalRGtime );
        text.push_back(Buffer);
        
        // 3M:
        sprintf( Buffer, "Frame %d", indexFrame);
        scaletext.push_back(Buffer);

        sprintf(Timechar, "Time of frame %d/ FM: %.5f/ OP: %.5f/ RG: %.5f",indexFrame, (double)(end_FM - start_FM) / CLOCKS_PER_SEC, (double)(end_OP - start_OP) / CLOCKS_PER_SEC, (double)(end_RG - start_RG) / CLOCKS_PER_SEC);
        timetext.push_back(Timechar);
        
        
// -------------update any infomation
        // update bool all_threshold_notchange
        all_threshold_notchange = true; //不能放在括号外  如果放在括号外 上一局的的all_threshold_notchange会影响
        for( int i=0; i<Segmentnum; i++){
            all_threshold_notchange = all_threshold_notchange && vectorS[i].threshold_notchange;
        }
        
        if (all_threshold_notchange){    // if threshold for this frame did not change,  next frame will be read in next loop
            
            //for ICP
            Allcontourvector_Pre.swap(Allcontourvector_cur);
            
            // for RG averageScale
            double averageScaleoneFrame = 0.0;
            for( int i=0; i<Segmentnum; i++)
            {
                averageScaleoneFrame += vectorS[i].data[2].back();
            }
            
            cout<< endl << "****************" <<endl;
            averageScaleoneFrame /= Segmentnum;
            cout<< "Average Scale of all Segment in same Frame: " << averageScaleoneFrame<<endl;
            
            // single Scale for  op/fm /rg
            sprintf(Scalechar, "Scale (index %d to %d)/ FM: %.7f/ OP: %.7f/ RG: %.7f / GT: %.7f", indexFrame, indexFrame-1, scaleFM, scaleOP, averageScaleoneFrame, scaleGT);
            scaletext.push_back(Scalechar);
            
            // accumulated Scale for FM /OP/RG/GT
            AccumscaleFM *= scaleFM;  AccumscaleOP *= scaleOP;   AccumscaleRG *= averageScaleoneFrame; AccumscaleICP *= scaleICP; AccumscaleGT *= scaleGT;
            sprintf(Acscalechar, "Accumulated Sclae/ FM: %.7f/ ", AccumscaleFM);
            sprintf(Acscalechar+strlen(Acscalechar), "OP: %.7f/ ", AccumscaleOP); sprintf(Acscalechar+strlen(Acscalechar), "RG: %.7f/ ", AccumscaleRG); sprintf(Acscalechar+strlen(Acscalechar), "GT: %.7f/ ", AccumscaleGT);
            scaletext.push_back(Acscalechar);
            
            // total running  time of OP FM and RG
            totoalFMtime +=(double)(end_FM - start_FM) / CLOCKS_PER_SEC;   totoalOPtime +=(double)(end_OP - start_OP) / CLOCKS_PER_SEC;
            sprintf(Totoaltimechar, "Totaltime/ FM: %.5f/ OP: %.5f/ RG: %.5f ", totoalFMtime, totoalOPtime, totoalRGtime);
            timetext.push_back(Totoaltimechar);
            
            // pixelrelation
            relationRG /= averageScaleoneFrame;  relationOP /= scaleOP;  relationFM /= scaleFM; relationICP /= scaleICP;
            sprintf(pixelrelachar, "Pixel relation/ FM: %.6f/ OP: %.6f/ RG: %.6f/ GT: %.6f", relationFM, relationOP,  relationRG, relationGT);
            timetext.push_back(pixelrelachar);
            
            cout<< "Pixelrelation: " << relationRG << "m/pixel"<<endl;
            
            //ofstream
            ofstream Scalestream;
            Scalestream.open(saveScaletxt,ios::out|ios::app);  Scalestream.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
            Scalestream.precision(strmPricision);
            Scalestream << indexFrame << " " <<scaleFM <<" "<< scaleOP<< " " << averageScaleoneFrame << " " <<scaleICP << " " << scaleGT << endl;
            Scalestream.close();
            
            ofstream AccumScalestream;
            AccumScalestream.open(saveACScaletxt,ios::out|ios::app);  AccumScalestream.setf(ios::fixed, ios::floatfield);  AccumScalestream.precision(strmPricision);
            AccumScalestream << indexFrame << " " <<AccumscaleFM <<" "<< AccumscaleOP<< " " <<  AccumscaleRG << " " <<AccumscaleICP << " " <<  AccumscaleGT <<endl;
            AccumScalestream.close();
            
            ofstream TotalTimestream;
            TotalTimestream.open(saveACTimetxt,ios::out|ios::app);  TotalTimestream.setf(ios::fixed, ios::floatfield);  TotalTimestream.precision(strmPricision);
            TotalTimestream << indexFrame << " " <<totoalFMtime <<" "<< totoalOPtime<< " " <<  totoalRGtime <<endl;
            TotalTimestream.close();
            
            ofstream SingleTimestream;
            SingleTimestream.open(saveSingleTimetxt,ios::out|ios::app);  SingleTimestream.setf(ios::fixed, ios::floatfield); SingleTimestream.precision(strmPricision);
            SingleTimestream << indexFrame << " " << (double)(end_FM - start_FM) / CLOCKS_PER_SEC <<" "<< (double)(end_OP - start_OP) / CLOCKS_PER_SEC << " " <<  (double)(end_RG - start_RG) / CLOCKS_PER_SEC <<endl;
            SingleTimestream.close();
            
            ofstream Relationstream;
            Relationstream.open(saverealtiontxt,ios::out|ios::app);  Relationstream.setf(ios::fixed, ios::floatfield);  Relationstream.precision(strmPricision);
            Relationstream << indexFrame << " " << relationFM << " " << relationOP << " " << relationRG << " " << relationICP << " " << relationGT <<endl;
            Relationstream.close();
            
            ofstream GTstream;
            GTstream.open(saveGTtxt,ios::out|ios::app);  GTstream.setf(ios::fixed, ios::floatfield); GTstream.precision(strmPricision);
            GTstream << indexFrame << " " << scaleGT << " " << AccumscaleGT << " " << relationGT <<endl;
            GTstream.close();
            relationGT_pre = relationGT; //for next frame
        }
        
        framethreemethode = putStats(timetext,framethreemethode, Vec3b(0,0,200), ptrBottomMiddle_3M, 'b' );
        framethreemethode = putStats(scaletext,framethreemethode, Vec3b(0,230,230), ptrTopLeft2, 't' );
        
        sprintf(Buffer, "%.5fm/p", relationRG);
        text.push_back(Buffer);
        
        FramewithCounter= putStats(text,FramewithCounter, Vec3b(0,0,170), ptrTopright, 'r' );
        text.clear();
        
        moveWindow(windowNameRG, 700, 0);
        imshow(windowNameRG, FramewithCounter);
        
        //moveWindow(windowName3M, 700, 450); // int x = column, int y= row
        //imshow(windowName3M, framethreemethode);
        
        // ---  Videocapture Trackbar activate
        //TrackBarFunc(0,0);
        //controlRate++; //
        
        vwRG << FramewithCounter;
        vw3M << framethreemethode;

//-------------delete the unuseful segment
        
        for( int i=0; i<vectorS.size(); i++){
            if (vectorS[i].LoopThreshold > Loopiterationmax || R[i].touchbordernum >= 1) {
                cout<<endl << "!!!! Delete objekt " << i+1 << " because of infinitv loop or object touch border of window" << endl << endl;
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
        
            if(newobject.checkThreshold(frame_Blur, relationRG)){
                vectorS.push_back(newobject);
            }
        }

        
//-------------define the button
        
        int keycode = waitKey(0); // equal to  waitKey(10);  //延时10ms
        
        if(keycode  == ' ')   waitKey(0); //32是空格键的ASCII值 暂停
        
        if(keycode  == 27)  stop=true; // 27 = ASCII ESC
        
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

                if (newobject.checkThreshold(frame_Blur, relationRG)){
                    vectorS.push_back(newobject);
                    i++;
                }
            }
        }
        
//        if(keycode  == 109){  // 109 =  m
//            Mat FramewithCounterBackup;
//            FramewithCounter.copyTo(FramewithCounterBackup);
//            cout <<endl<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
//            cout<<"Test! real distrance of two pixel length"<<endl;
//            cout<<"Please mark two point on the image"<<endl;
//            //#define Pixel_realtion_window "Pixel-distance relation"
//            namedWindow( Pixel_realtion_window );
//            setMouseCallback(Pixel_realtion_window,on_MouseHandle,(void*)&FramewithCounterBackup);
//
//            while(1)
//            {
//                imshow( Pixel_realtion_window, FramewithCounterBackup );
//                waitKey( 10 );
//                //if( waitKey( 10 ) == 13 ) break;//按下enter键，程序退出
//                if (REvector_mark.size() == 2) break;
//            }
//
////            if (REvector_mark.size() != 2){
////                cout<< "\n!!!!!!!You did not mark 2 points. Porgramm breaks"  <<endl;
////                return 0;
////            }
//
//            double pixeld = pixeldistance(FramewithCounterBackup, REvector_mark);
//            cout<< "Pixeldistance: " << pixeld <<endl;
//            cout<< "The real length: " << relationRG * pixeld  << " m \n" << endl;
//
//            imshow( Pixel_realtion_window, FramewithCounterBackup );
//            waitKey(0);
//            destroyWindow(Pixel_realtion_window);
//            REvector_mark.clear();
//        }
        
        
    }

    cout << "Video plays over(outside loop)" << endl;
    
    vc.release();
    vwRG.release();  vw3M.release();  vwop.release();  vwsp.release();  vwgt.release();  vwfm.release();
    
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
                //Origin of text ist Bottom-left corner of the text string
                (*origin).x = (*origin).x- text_size.width / 2;
                (*origin).y = (*origin).y - 1.2*text_size.height ;
                putText(frame, stats[i].c_str(), *origin, font_face , font_scale, color, thickness, 8, false);  //When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
                (*origin).x = (*origin).x+ text_size.width / 2;
            }
            break;
        
        case 't' :
            for(int i=0; i<stats.size(); i++){
                Size text_size = getTextSize(stats[i].c_str(), font_face, font_scale, thickness, &baseline);
                (*origin).y = (*origin).y + 1.3*text_size.height ;
                putText(frame, stats[i].c_str(), *origin, font_face , font_scale, color, thickness, 8, false);
            }
            break;
            
        case 'r' :
            for(int i=0; i<stats.size(); i++){
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

void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
    Scalar clr_end= Scalar(255,255,0); // 青色
    Scalar clr_start = Scalar(0,255,255) ; //yellow
    Scalar clr_text = Scalar(255,0,255);// proper
    Scalar clr_delete = Scalar(170,178,32);//
    
    int Radius = 2;
    int font_face = FONT_HERSHEY_COMPLEX_SMALL;
    double font_scale = 0.7;
    int thickness = 1;
    int baseline;
    
    Mat& image = *(Mat*) param;
    //Mat *im = reinterpret_cast<Mat*>(param);
    Mat Matmause =image.clone();   char temp[30];   static Point2f pt_pre, pt_next;
    
    if( x < 0 || x >= image.cols || y < 0 || y >= image.rows )   return ;
        
    switch(event)
    {
        case EVENT_MOUSEMOVE:{
            
            Point pt = Point(x,y);
            Point2f pt_temp = Point2f(x/RSscale , y/RSscale );
            
            sprintf(temp,"(Row:%.2f, Col:%.2f)",pt_temp.y, pt_temp.x);
            putText(Matmause,temp, pt, font_face, font_scale,  clr_text, thickness, 8, false);
            circle( Matmause, pt, Radius, Scalar(255,0,0,0) ,-1, 8, 0 );
            imshow(Pixel_realtion_window, Matmause);
            waitKey(700);
            break;
        }
            
        case EVENT_LBUTTONDOWN:{
            pt_pre = Point(x, y);
            Point2f pt_pre_temp = Point2f(x/RSscale , y/RSscale );
            
            cout<<"Start: Row: "<<pt_pre_temp.y<<", Column: "<< pt_pre_temp.x <<endl;
            sprintf(temp,"(%.2f,%.2f)",pt_pre_temp.y, pt_pre_temp.x);
            
            putText(image,temp, pt_pre, font_face, font_scale,  clr_text, thickness, 8, false);
            circle( image, pt_pre, Radius, clr_start ,-1, 8, 0 );
            REvector_mark.push_back(pt_pre_temp);
            break;
        }
            
        case EVENT_RBUTTONDOWN:{
            if(! (flags & CV_EVENT_FLAG_CTRLKEY)){
                pt_next = Point(x, y);
                Point2f pt_next_temp = Point2f(x/RSscale, y/RSscale);
                
                cout<<"End: Row: "<<pt_next_temp.y<<", Column: "<< pt_next_temp.x <<endl;
                sprintf(temp,"(%.2f,%.2f)",pt_next_temp.y, pt_next_temp.x);
                
                putText(image,temp, pt_next, font_face, font_scale,  clr_text, thickness, 8, false);
                line(image, pt_pre, pt_next, Scalar(0,0,230),2,8,0);
                circle( image, pt_next, Radius, clr_end ,-1, 8, 0 );
                REvector_mark.push_back(pt_next_temp);
            }
            
            if( !REvector_mark.empty() && (flags & CV_EVENT_FLAG_CTRLKEY)){  // ctrl+ rightmouse  = delete
                Point2f g_pt = REvector_mark.back();
                Point2f g_pt_temp = Point2f(g_pt.x *RSscale , g_pt.y *RSscale );
                
                cout<<"Delete: Row: "<<g_pt.y<<", Column: "<< g_pt.x <<endl;
                
                sprintf(temp," Delete");
                Size text_size = getTextSize(temp, font_face, font_scale, thickness, &baseline);
                Point orgin(g_pt_temp.x, g_pt_temp.y ); orgin.y += 1.2*text_size.height;
                
                putText(image,temp, orgin, font_face, font_scale, clr_delete, thickness, 8, false);
                circle( image, g_pt_temp, Radius, clr_delete ,-1, 8, 0 );
                
                REvector_mark.pop_back();
            }
            
            break;
            
//        case CV_EVENT_RBUTTONDBLCLK:{
//            if(!REvector_mark.empty() && (flags & CV_EVENT_FLAG_CTRLKEY)){
//                Point g_pt = REvector_mark.back();
//                cout<<"Delete: Row: "<<g_pt.y<<", Column: "<< g_pt.x <<endl;
//                sprintf(temp,"Delete");
//
//                Size text_size = getTextSize(temp, font_face, font_scale, thickness, &baseline);
//                g_pt.y = g_pt.y - 1.2*text_size.height;
//                putText(image,temp, g_pt, font_face, font_scale, clr_text, thickness, 8, false);
//                circle( image, g_pt, 2, clr_end ,-1, 8, 0 );
//                REvector_mark.pop_back();
//            }
//            break;
            
        
//            if(!REvector_mark.empty()){
//                 g_pt = REvector_mark.back();
//                cout<<"Delete: Row: "<<g_pt.y<<", Column: "<< g_pt.x <<endl;
//                Scalar addcolor = Matmause.at<Vec3b>(g_pt);
//                cout<< addcolor << endl;
//                line(image, g_pt, g_pt, addcolor,4,8,0);
//                REvector_mark.pop_back();
//            }
//            break;
        }
            
        default:
            break;
    }
        
//        if( event == EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))
//        {
//
//            Point g_pt = Point(x,y);
//
//            char temp[16];
//            sprintf(temp,"(Row:%d, Col:%d)",g_pt.y, g_pt.x);
//
//            putText(Matmause,temp, g_pt, font_face, font_scale,  clr, thickness, 8, false);
//            circle( Matmause, g_pt, 3, Scalar(255,0,0,0) ,-1, 8, 0 );
//            imshow(Pixel_realtion_window, Matmause);
//            waitKey(1000);
//        }
//
//        else if (event == EVENT_LBUTTONDOWN){
//
//            Point g_pt = Point(x, y);
//            cout<<"Row: "<<y<<", Column: "<< x <<endl;
//            line(image, g_pt, g_pt, clr,4,8,0);
//            REvector_mark.push_back(g_pt);
//        }
//
//        else if (event == EVENT_RBUTTONDOWN){
//
//            if(!REvector_mark.empty()){
//                Point g_pt = REvector_mark.back();
//                cout<<"Delete: Row: "<<g_pt.y<<", Column: "<< g_pt.x <<endl;
//                Scalar addcolor = Matmause.at<Vec3b>(g_pt);
//                cout<< addcolor << endl;
//                line(image, g_pt, g_pt, addcolor,4,8,0);
//                REvector_mark.pop_back();
//            }
//
//        }
    
}

Mat Featurematch(Mat preframe , Mat nextframe,  vector<Point2f>& obj_last, vector<Point2f>& obj_next){
    
    int minHessian = 500;//SURF算法中的hessian阈值    the higher the minHessian, the fewer keypoints you will obtain, but you expect them to be more repetitive
    const int GOOD_PTS_MAX = 150;
    const float GOOD_PORTION = 0.1;
    
    //----- define Feature Detector and extractor（SURF） 特征检测类对象
    Ptr<SURF> f2d = xfeatures2d::SURF::create(minHessian);
    //Ptr<ORB> f2d = ORB::create();
    //Ptr<SIFT> f2d = xfeatures2d::SIFT::create();

    vector<KeyPoint> keypoints_pre, keypoints_next;//vector模板类，存放任意类型的动态数组
    Mat descriptors_pre, descriptors_next;
    
    //Ptr<SURF> detector = xfeatures2d::SURF::create(minHessian);
//    f2d->detect( preframe, keypoints_pre );
//    f2d->detect( nextframe, keypoints_next );
    
//    SurfDescriptorExtractor extractor;
//    Ptr<SURF> extractor = SURF::create();
//    Ptr<SIFT> extractor = xfeatures2d::SIFT::create();
//    f2d->compute( preframe, keypoints_pre, descriptors_pre );
//    f2d->compute( nextframe, keypoints_next, descriptors_next );
    
    f2d->detectAndCompute(preframe, Mat(), keypoints_pre, descriptors_pre);
    f2d->detectAndCompute(nextframe, Mat(), keypoints_next, descriptors_next);
    
    //------- Matching descriptor vectors using FLANN/BruteForce/ DescriptorMatcher matcher
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
    for( int i = 0; i < ptsPairs; i++ ){
        good_matches.push_back( matches[i]);
    }
    cout << "Calculating homography matrice using " << ptsPairs << " point pairs." << endl;
    
    //绘制出匹配到的关键点   draw the Good Match pairs
    Mat img_matches;
    drawMatches( preframe, keypoints_pre, nextframe, keypoints_next,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    resize(img_matches, img_matches, Size(), 0.8, 1.2, INTER_CUBIC);
    
    //---- 从匹配成功的匹配对中获取关键点  -- Get the keypoints from the good matches
    for( unsigned int i = 0; i < good_matches.size(); i++ ){
        obj_last.push_back( keypoints_pre[ good_matches[i].queryIdx ].pt);
        obj_next.push_back( keypoints_next[ good_matches[i].trainIdx ].pt);
    }
    
    // draw output Matimage
    vector<KeyPoint> keypoints_output;
    for( unsigned int i = 0; i < good_matches.size(); i++ )
    {
        keypoints_output.push_back( keypoints_next[ good_matches[i].trainIdx]);
    }
    
    Mat output;
    drawKeypoints(nextframe, keypoints_output, output, Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
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
    
    return output;
}

Mat Spixel(Mat inputmat, int& number_slic,  vector<Vec3b>& colorSP){
    Mat mask, labels, frameSP_Blur;
    
    Mat frameSP = inputmat.clone();
    GaussianBlur(frameSP, frameSP_Blur, kernelsize,0,0);
    cvtColor(frameSP_Blur, frameSP_Blur, COLOR_RGB2Lab);
    
    Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(frameSP_Blur, SLICO, 30, 10.0);  // region_size = sqrt(area) , number_slic*region_size^2 = window area
    
    slic->iterate();//迭代次数，默认为10
    slic->enforceLabelConnectivity(25); //default is 25
    slic->getLabelContourMask(mask);//获取超像素的边界
    slic->getLabels(labels);//获取labels
    number_slic = slic->getNumberOfSuperpixels();//获取超像素的数量
    //cout<< "number_slic: "<< number_slic << endl;
    
    //Ptr<SuperpixelSEEDS> seeds = createSuperpixelSEEDS(frame.cols, frame.rows, frame.channels(), 1000, 15, 2, 5, true);
    //seeds->iterate(frame);//迭代次数，默认为4
    //seeds->getLabels(labels);//获取labels
    //seeds->getLabelContourMask(mask);;//获取超像素的边界
    //int number_seeds = seeds->getNumberOfSuperpixels();//获取超像素的数量
    
    //Ptr<SuperpixelLSC> lsc = createSuperpixelLSC(frame);
    //        lsc->iterate();//迭代次数，默认为4
    //        lsc->enforceLabelConnectivity();
    //        lsc->getLabels(labels);//获取labels
    //        lsc->getLabelContourMask(mask);;//获取超像素的边界
    //        int number_lsc = lsc->getNumberOfSuperpixels();//获取超像素的数量
    
    
    //frameSP.setTo(Scalar(0, 0 , 230), mask);
    //imshow("Superpixel", frameSP);
    
    Mat PerspectiveImage = Mat::zeros(frameSP.size(),CV_8UC3);
    
    
    //Vec3b fillcolor[number_slic];
    if ( colorSP.size() < number_slic ){
        //cout<< "loop" << endl;
        RNG rng;
        for (int k = (int)colorSP.size() ; k < number_slic ; k++){
            //waitKey(100);
            //int colornum  = floor((k+1) *(255.0/ number_slic));
            //Vec3b fillcolor = Vec3b(rng.uniform(0,colornum), rng.uniform(0,colornum), rng.uniform(0,colornum));
            //Vec3b fillcolor = Vec3b(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
            colorSP.push_back( Vec3b(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)) );
        }
    }
    
    //cout<< colorSP[1] << endl;
    
//    for (int k = 0 ; k < number_slic ; k++){
//
//        for(int i=0;i<labels.rows;i++)
//        {
//            for(int j=0;j<labels.cols;j++)
//            {
//                if(labels.at<int>(i,j) == k)
//                {
//
//                    PerspectiveImage.at<Vec3b>(i,j)= colorSP[k];
//                }
//            }
//        }
//    }
    
//    imshow("After ColorFill",PerspectiveImage);
//    waitKey(1);
    
    //Mat together;
    //addWeighted(frameSP,0.6,PerspectiveImage,0.4,0,together);
    frameSP.setTo(Scalar(255, 255 ,255), mask);
    
    //cout<< PerspectiveImage<< endl;
    //Mat markers(markerMask.size(), CV_32S);
    //mask.convertTo(mask, CV_32SC1);
    //        imshow("mask_before", mask);
    //        watershed( frameSP, mask );
    //        Mat afterWatershed;
    //        convertScaleAbs(mask,afterWatershed);
    //        imshow("After Watershed",afterWatershed);
    //        //imshow("Superpixel_watershed", frameSP);
    //        imshow("mask", mask);
    
    //cout<< labels<< endl;
    //labels.convertTo(labels, CV_8UC1, 255/ (number_slic - 1));
    //labels.convertTo(labels, CV_64FC1, 1.0/(number_slic-1));
    //imshow("Labels", labels);
    
    return frameSP;
}


double flann_KDtree(vector<Point> dest_vector, vector<Point> obj_vector){
    // find nearest neighbors using FLANN
    
    Mat Matdest = Mat(dest_vector).reshape(1); Matdest.convertTo(Matdest,CV_32F);
    Mat Matobj = Mat(obj_vector).reshape(1);  Matobj.convertTo(Matobj,CV_32F);
    Mat Matindice;
    Mat Matdist;
    //Mat Matalign;
    vector<Point> align_vector;
    vector<Point2f>obj_vector_pr, align_vector_pr;
    
    Index flann_index(Matdest, KDTreeIndexParams(2));  // using 2 randomized kdtrees
    flann_index.knnSearch(Matobj, Matindice, Matdist, 1, SearchParams(64) );
    int error  = sum(Matdist)[0];
    
    cout<< "error: " <<error << endl;
    
    // matching pairs for scale computation
    //vector<Point> obj_vector_pr, align_vector_pr;
    for(int i = 0; i< obj_vector.size(); i++){
        int row  = Matindice.at<int>(i,0);
        align_vector.push_back(dest_vector[row]);
        
        if (Matdist.at<float>(i,0) <= 10){
            obj_vector_pr.push_back(obj_vector[i]);
            align_vector_pr.push_back(dest_vector[row]);
        }
    }
    
    Mat H_ICP = findHomography( obj_vector_pr, align_vector_pr ,CV_RANSAC);
    double scale  = decomposematrix(H_ICP);
    
    return scale;
}







