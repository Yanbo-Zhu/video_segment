//
//  INITIALSEED.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 26.09.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#include "INITIALSEED.hpp"

//Initalseed(Mat x, Mat y) constructor function
//{
//    MatInBackup = x;
//    MatGrowCur = y;
//}


void Initalseed :: on_MouseHandle(int event, int x, int y, int flags, void* param)
{
    
    //cout<<"in loop" <<endl;
    //Mat & image = *(Mat*) param;
    //if( x < 0 || x >= image.cols || y < 0 || y >= image.rows ){
    //    return;
    //}
    // Check for null pointer in userdata and handle the error
    Initalseed* temp = reinterpret_cast<Initalseed*>(param);
    temp->on_Mouse(event, x, y, flags);
    
    //Scalar colorvalue = image.at<Vec3b>(Point(x, y));
    //Vec3b colorvalue = image.at<Vec3b>(Point(x, y));
    //cout<<"at("<<x<<","<<y<<") pixel value: " << colorvalue <<endl;
    
    //调用函数进行绘制
    //DrawLine( image, Point(x, y));//画线
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

void Initalseed :: on_Mouse(int event, int x, int y, int flags)
{
    
    //Mat & image = *(Mat*) param;
    //Mat *im = reinterpret_cast<Mat*>(param);
    
    //mouse ist not in window 处理鼠标不在窗口中的情况
    //    if( x < 0 || x >= image.cols || y < 0 || y >= image.rows ){
    //        return;
    //    }
    
    if (event == EVENT_LBUTTONDOWN)
        
    {
        //g_pt = Point(x, y);
        initialseedvektor.push_back(Point(x, y));
        cout<<"at( row: "<< y <<", column: "<< x <<" )"<<endl;
    }
}

void Initalseed :: drawpoint(Mat firstFrame, vector<Point> initialseedvektor, Vec3b color)
{
    for(size_t i=0; i<initialseedvektor.size();i++)
    {
        
        Scalar colorvalue = firstFrame.at<Vec3b>(initialseedvektor[i]);
        //Vec3b colorvalue = image.at<Vec3b>(Point(x, y));
        printf("Seed %d: (Row: %d, Column: %d) / ",  int(i)+1,  initialseedvektor[i].y, initialseedvektor[i].x  );
        cout << " pixel value: " << colorvalue <<endl;
        //调用函数进行绘制
        DrawLine( firstFrame, initialseedvektor[i], color); //画线

    }
    
    imshow ("firstFrame with initialseedvektor" , firstFrame);
    waitKey(10);
}

void Initalseed :: DrawLine( Mat &img, Point pt, Vec3b color )
{
    //RNG rng(time(0));
    //line(img, pt, pt, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),6,8,0); //随机颜色
    int thickness = 6;
    int lineType = 8;
    line(img, pt, pt, color,thickness,lineType,0);
    //line(img, pt, pt, Scalar(0,0,255),thickness,lineType,0); //随机颜色
    
}

void Initalseed :: modechoose(int x, Mat firstFrame)
{
    //Mat MatGrowCur(firstFrame.size(),CV_8UC3,Scalar(0,0,0));
    Mat MatInBackup = firstFrame.clone();
    
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
                    double averageonepoint = (firstFrame.at<Vec3b>(i,j)[0]+ firstFrame.at<Vec3b>(i,j)[1] + firstFrame.at<Vec3b>(i,j)[2])*1.0/3;
                    
                    //printf("averageonepoint = %f\n",averageonepoint);
                    
                    if(averageonepoint >= thresholdvalue)//选取种子点，自己更改
                    {   // mark the growing seeds
                        
                        g_pt = Point(j, i);
                        initialseedvektor.push_back(g_pt);
                        //MatGrowCur.at<Vec3b>(i,j)= firstFrame.at<Vec3b>(i,j);  //255: white
                    }
                }
            }
            
            break;
            
        case 2:
            
            // tap 2, choose seeds by clicking in orignal image
            #define WINDOW_NAME " point marking "
            
            // setting foe mouse
            namedWindow( WINDOW_NAME );
            //imshow( WINDOW_NAME, MatInBackup);
            //setMouseCallback(WINDOW_NAME, Initalseed :: on_MouseHandle,(void*)&MatInBackup);
            //setMouseCallback(WINDOW_NAME, Initalseed :: on_MouseHandle, &MatInBackup);
            setMouseCallback(WINDOW_NAME, Initalseed ::on_MouseHandle,this);
            
            while(1)
            {
                imshow( WINDOW_NAME, MatInBackup);
                //setMouseCallback(WINDOW_NAME, Initalseed :: on_MouseHandle,this);
                //if( waitKey( 10 ) == 27 ) break;//按下ESC键，程序退出
                if( waitKey( 10 ) == 13 ) break; // 按下 enter 键
            }
            
            destroyWindow(WINDOW_NAME);
            
            //initialize the seeds
            for(size_t i=0;i<initialseedvektor.size();i++)
            {
                printf("Seed %d: (Row: %d, Column: %d)\n",  int(i)+1,  initialseedvektor[i].y, initialseedvektor[i].x  );
                //cout << initialseedvektor[i] <<endl;
               
            }
            
            break;
            
        case 3:
            //tap 3 , set seeds by using default position of points
            
            // descend video Rotation_descend_20_10m_small
            //seedvektor.push_back(Point(677, 280)); // white boot
            //seedvektor.push_back(Point(439,221)); // white window
            
            // ascend video ascend_5-50m
            initialseedvektor.push_back(Point(581,33)); // white window
            
            for(size_t i=0; i<initialseedvektor.size();i++)
            {
                printf("Seed %d: (Row: %d, Column: %d)\n",  int(i)+1,  initialseedvektor[i].y, initialseedvektor[i].x  );
                //MatGrowCur.at<Vec3b>(initialseedvektor[i]) = firstFrame.at<Vec3b>(initialseedvektor[i]);
            }
            
            break;
            
        default:
            break;
    }
    
    
}
