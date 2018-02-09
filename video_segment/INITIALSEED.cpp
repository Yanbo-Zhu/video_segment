//
//  INITIALSEED.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 26.09.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#include "INITIALSEED.hpp"
#include <time.h>


Initialseed :: Initialseed(){// constructor function
}

Initialseed :: Initialseed(Mat Frame ){
    int width = Frame.cols;
    int height = Frame.rows;
    this->randomseed(Frame, width, height);
    //this->newseed(Frame);
    waitKey(1000);
    RNG rng(time(0));
    color = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    data.resize(8);
}

Initialseed :: Initialseed(int x, Mat firstFrame, int objektindex,  double defaultTH[], vector<vector<Point>> defaultSD){
    this->modechoose(x, firstFrame, objektindex, defaultTH, defaultSD);
    waitKey(1000);
    RNG rng(time(0));
    color = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    data.resize(8);
}

void Initialseed :: randomseed(Mat firstFrame, int width, int height)
{
    
    Mat MatInBackup = firstFrame.clone();
    RNG rng(time(0));
    Point newrandomseed = Point(rng.uniform(0, width), rng.uniform(0, height));
    
    initialseedvektor.push_back(newrandomseed);
    
    printf( "New random seed : (Row: %d, Column: %d) / ",   newrandomseed.y, newrandomseed.x );
    
    differencegrow = 8;
}

void Initialseed :: modechoose(int x, Mat firstFrame, int objektindex,  double defaultTH[], vector<vector<Point>> defaultSD)
{
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    Mat MatInBackup = firstFrame.clone();
    printf("Plaese select initial seeds \n");
    
    switch (x) {
            
            //tap 1, choose seeds by entrying the threshold value
        case 1:
            
            cout<< "plaese give the value for threshold value (seed condition):" <<endl;
            cin >> thresholdvalue;
            
            //initialize the seeds 初始化原始种子点
            for(int i=0;i<firstFrame.rows;i++)
            {
                for(int j=0;j<firstFrame.cols;j++)
                {
                    double averageonepoint = (firstFrame.at<Vec3b>(i,j)[0]+ firstFrame.at<Vec3b>(i,j)[1] + firstFrame.at<Vec3b>(i,j)[2])/3.0;
                    
                    if(averageonepoint >= thresholdvalue)//选取种子点，自己更改
                    {   // mark the growing seeds
                        
                        g_pt = Point(j, i);
                        initialseedvektor.push_back(g_pt);
                        //MatGrowCur.at<Vec3b>(i,j)= firstFrame.at<Vec3b>(i,j);  //255: white
                    }
                }
            }
            printf("\nPlease set the threshold value for region growing\n");
            cin >> differencegrow;
            
            break;
            
        case 2:
            // tap 2, choose seeds by clicking in orignal image
            
            // setting foe mouse
            namedWindow( Point_mark_window );
            //imshow( WINDOW_NAME, MatInBackup);
            //setMouseCallback(WINDOW_NAME, Initalseed :: on_MouseHandle,(void*)&MatInBackup);
            //setMouseCallback(WINDOW_NAME, Initalseed :: on_MouseHandle, &MatInBackup);
            setMouseCallback(Point_mark_window, Initialseed ::on_MouseHandle,this);
            
            while(1)
            {
                imshow( Point_mark_window, MatInBackup);
                //setMouseCallback(WINDOW_NAME, Initalseed :: on_MouseHandle,this);
                //if( waitKey( 10 ) == 27 ) break;//按下ESC键，程序退出
                if( waitKey( 10 ) == 13 ) break; // 按下 enter 键
            }
            
            destroyWindow(Point_mark_window);
    
            printf("\nPlease set the threshold value for region growing\n");
            cin >> differencegrow;
            
            break;
            
        case 3:
            //tap 3 , set seeds by using default position of points
            
            //Point (x,y )  x= column  y = row

            initialseedvektor = defaultSD[objektindex];
            
            //initialize the seeds
            for(size_t i=0; i<initialseedvektor.size();i++)
            {
                printf("Seed %d: (Row: %d, Column: %d)\n",  int(i)+1,  initialseedvektor[i].y, initialseedvektor[i].x  );
                //MatGrowCur.at<Vec3b>(initialseedvektor[i]) = firstFrame.at<Vec3b>(initialseedvektor[i]);
            }
            printf("\nPlease set the threshold value for region growing \n");
            
            differencegrow = defaultTH[objektindex];
            cout << differencegrow << endl;
            break;
            

        default:
            cout<< "Wrong number input during choosing Method" << endl;
            exit(0);
            //break;
            
    }
}

void Initialseed :: newseed(Mat firstFrame)
{
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    Mat MatInBackup = firstFrame.clone();
    printf("Plaese select initial seeds \n");
    
    //choose seeds by clicking in orignal image
            
    // setting for mouse
    namedWindow( Point_mark_window );
    setMouseCallback(Point_mark_window, Initialseed ::on_MouseHandle,this);
    
    while(1)
    {
        imshow( Point_mark_window, MatInBackup);
        //setMouseCallback(WINDOW_NAME, Initalseed :: on_MouseHandle,this);
        //if( waitKey( 10 ) == 27 ) break;//按下ESC键，程序退出
        if( waitKey( 10 ) == 13 ) break; // 按下 enter 键
    }
    
    destroyWindow(Point_mark_window);
    
    //initialize the seeds
    for(size_t i=0;i<initialseedvektor.size();i++)
    {
        //printf("Seed %d: (Row: %d, Column: %d)\n",  int(i)+1,  initialseedvektor[i].y, initialseedvektor[i].x  );
        //cout << initialseedvektor[i] <<endl;
    }
    
    printf("\nPlease set the threshold value for region growing\n");
    cin >> differencegrow;
            
}

void Initialseed :: drawpoint(Mat Frame, vector<Point> initialseedvektor)
{
    for(size_t i=0; i<initialseedvektor.size();i++)
    {
        
        Scalar colorvalue = Frame.at<Vec3b>(initialseedvektor[i]);
        double intensity = (Frame.at<Vec3b>(initialseedvektor[i])[0] + Frame.at<Vec3b>(initialseedvektor[i])[1] + Frame.at<Vec3b>(initialseedvektor[i])[2]) / 3.0;
        //Vec3b colorvalue = image.at<Vec3b>(Point(x, y));
        printf("Seed %d: (Row: %d, Column: %d) / ",  int(i)+1,  initialseedvektor[i].y, initialseedvektor[i].x  );
        cout << " Scalar value: " << colorvalue << " / Intensity: " << intensity <<endl;
        //调用函数进行绘制
        DrawLine( Frame, initialseedvektor[i], color); //画线
    }
    
    imshow ("Frame with initial seeds" , Frame);
    waitKey(1);
}

void Initialseed :: on_MouseHandle(int event, int x, int y, int flags, void* param)
{
    //cout<<"in loop" <<endl;
    //Mat & image = *(Mat*) param;
    //if( x < 0 || x >= image.cols || y < 0 || y >= image.rows ){
    //    return;
    //}
    // Check for null pointer in userdata and handle the error
    Initialseed* temp = reinterpret_cast<Initialseed*>(param);
    temp->on_Mouse(event, x, y, flags);
    
}

void Initialseed :: on_Mouse(int event, int x, int y, int flags)
{
    //Mat & image = *(Mat*) param;
    //Mat *im = reinterpret_cast<Mat*>(param);
    
    //mouse ist not in window 处理鼠标不在窗口中的情况
    //    if( x < 0 || x >= image.cols || y < 0 || y >= image.rows ){
    //        return;
    //    }
    
    if (event == EVENT_LBUTTONDOWN)
        
    {
        initialseedvektor.push_back(Point(x, y));
        cout<<"at( row: "<< y <<", column: "<< x <<" )"<<endl;
    }
}

void Initialseed :: DrawLine( Mat &img, Point pt, Vec3b color )
{
    //RNG rng(time(0));
    //line(img, pt, pt, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),6,8,0); //随机颜色
    int thickness = 6;
    int lineType = 8;
    line(img, pt, pt, color,thickness,lineType,0);
    //line(img, pt, pt, Scalar(0,0,255),thickness,lineType,0); //随机颜色
}

bool Initialseed :: checkThreshold(Mat frame){

    Mat Frame = frame.clone();
    int width = Frame.cols;
    int height = Frame.rows;
    Regiongrowing RTest;
    //Counter CTest;
    bool threshold_qualified(true);
    int iterationnum = 0;
    bool repeat_thres = false;
    vector<double> Thresholdstack;
    Mat Mattest;
    Mat Matcounter;

    while ((iterationnum <Threshoditerationmax) && (!repeat_thres)) {
        
        
        Mat Mattest_Blur = Frame.clone();
        Mattest = RTest.RegionGrow(Frame, Mattest_Blur , this->differencegrow, this->initialseedvektor);
        
        Matcounter = RTest.FindCounter(Mattest, Frame, this->color);
        imshow("Contour of Segment", Matcounter);
        waitKey(100);
        
        cout<< "iterationnum: " << iterationnum <<"/ Area: " << RTest.Area <<"/ Threshold: "<< this->differencegrow << endl;
        
        Thresholdstack.push_back(this->differencegrow) ;

        if (RTest.Area < (width*height/300)) this->differencegrow = (this->differencegrow + 0.1);  // (Width*Height/300) = 2000   (Width*Height/50 )= 10000
        else if (RTest.Area <= (width*height/40) ) break;
        else  this->differencegrow = (this->differencegrow - 0.1);

        iterationnum++;

        vector<double>::iterator iterfind2;
        iterfind2 = find( Thresholdstack.begin(), Thresholdstack.end(), this->differencegrow);

        if(iterfind2 != Thresholdstack.end()){
            cout << "New Threshlod ist already available Threshlod vector. " << endl;
            repeat_thres = true;
        }

        //cout<< "Threshold: " << this->differencegrow <<endl;
    }

    if(iterationnum >=Threshoditerationmax || repeat_thres){
        cout<< "This new random initial point can not become a available seed point " << endl;
        threshold_qualified = false;
    }

    else{
        // create new object sucessfully
        this->LoopThreshold = 1;

        this->data[0].push_back(RTest.EWlong);    // Green line. long axis
        this->data[1].push_back(RTest.EWshort);   // lightly blue line . short axis
        this->data[2].push_back(RTest.Ratio);
        this->data[3].push_back(1.0); // scale
        this->data[4].push_back(initialScalediff); // ScaleDifference
        this->data[5].push_back(0.0); // RatioDifference
        this->data[6].push_back(RTest.Area);  // Area
        this->data[7].push_back(1.0);  // scaleArea
        
        this->preSegment  = Mattest.clone();
        this->initialseedvektor.clear();
        this->initialseedvektor.push_back(RTest.cntr);
        this->threshold_notchange = true;

        //waitKey(0);
    }

    destroyWindow("Contour of Segment");
    
    return threshold_qualified;
}


//void MyClass::on_Mouse(int event, int x, int y)
//{
//    switch (event)
//    {
//        case CV_EVENT_LBUTTONDOWN:
//            //your code here
//            break;
//        case CV_EVENT_MOUSEMOVE:
//            //your code here
//            break;
//        case CV_EVENT_LBUTTONUP:
//            //your code here
//            break;
//    }
//}


