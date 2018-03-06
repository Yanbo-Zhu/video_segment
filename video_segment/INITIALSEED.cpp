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
    
    differencegrow = 4;
}

void Initialseed :: modechoose(int x, Mat firstFrame, int objektindex,  double defaultTH[], vector<vector<Point>> defaultSD)
{
    Mat MatInBackup = firstFrame.clone();
    printf("Plaese select initial seeds \n");
    
    switch (x) {
            
//        //tap 1, choose seeds by entrying the threshold value
//        case 1:
//
//            cout<< "plaese give the value for threshold value (seed condition):" <<endl;
//            cin >> thresholdvalue;
//
//            //initialize the seeds 初始化原始种子点
//            for(int i=0;i<firstFrame.rows;i++)
//            {
//                for(int j=0;j<firstFrame.cols;j++)
//                {
//                    double averageonepoint = (firstFrame.at<Vec3b>(i,j)[0]+ firstFrame.at<Vec3b>(i,j)[1] + firstFrame.at<Vec3b>(i,j)[2])/3.0;
//
//                    if(averageonepoint >= thresholdvalue)//选取种子点，自己更改
//                    {   // mark the growing seeds
//
//                        Point g_pt = Point(j, i);
//                        initialseedvektor.push_back(g_pt);
//                        //MatGrowCur.at<Vec3b>(i,j)= firstFrame.at<Vec3b>(i,j);  //255: white
//                    }
//                }
//            }
//            printf("\nPlease set the threshold value for region growing\n");
//            cin >> differencegrow;
//
//            break;
            
        case 1:
            // tap 1, choose seeds by clicking in orignal image
            
            namedWindow( Point_mark_window );
            //setMouseCallback(Point_mark_window, Initialseed :: on_MouseHandle, &MatInBackup);
            setMouseCallback(Point_mark_window, Initialseed ::on_MouseHandle,this);
            
            while(1)
            {
                imshow( Point_mark_window, MatInBackup);
                //setMouseCallback(WINDOW_NAME, Initalseed :: on_MouseHandle,this);
                if( waitKey( 10 ) == 13 ) break; // 按下 enter 键
            }
            
            destroyWindow(Point_mark_window);
    
            printf("\nPlease set the threshold value for region growing\n");
            cin >> differencegrow;
            break;
            
        case 2:
            //tap 2, set seeds by using default position of points
            //Point (x,y )  x= column  y = row

            initialseedvektor = defaultSD[objektindex];
            
            for(size_t i=0; i<initialseedvektor.size();i++)
            {
                printf("Seed %d: (Row: %d, Column: %d)\n",  int(i)+1,  initialseedvektor[i].y, initialseedvektor[i].x  );
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

void Initialseed :: on_MouseHandle(int event, int x, int y, int flags, void* param)
{
//    Mat& image = *(Mat*) param;
//    if( x < 0 || x > image.cols -1 || y < 0 || y > image.rows -1  ){
//        return;
//    }
    
    Initialseed* Imgtemp = reinterpret_cast<Initialseed*>(param);
    Imgtemp->on_Mouse(event, x, y, flags);
}

void Initialseed :: on_Mouse(int event, int x, int y, int flags)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        initialseedvektor.push_back(Point(x, y));
        cout<<"at( row: "<< y <<", column: "<< x <<" )"<<endl;
    }
}

void Initialseed :: drawpoint(Mat Frame2, vector<Point> initialseedvektor)
{
    Mat Frame = Frame2.clone();
    int thickness = 2;
    int lineType = 8;
    
    for(size_t i=0; i<initialseedvektor.size();i++)
    {
        Scalar colorvalue = Frame.at<Vec3b>(initialseedvektor[i]);
        double intensity = (Frame.at<Vec3b>(initialseedvektor[i])[0] + Frame.at<Vec3b>(initialseedvektor[i])[1] + Frame.at<Vec3b>(initialseedvektor[i])[2]) / 3.0;
        printf("Seed %d: (Row: %d, Column: %d) / ",  int(i)+1,  initialseedvektor[i].y, initialseedvektor[i].x  );
        cout << " Scalar value: " << colorvalue << " / Intensity: " << intensity <<endl;
        
        line(Frame, initialseedvektor[i], initialseedvektor[i], color,thickness, lineType,0);
    }
    
    //imshow ("Frame with initial seeds" , Frame);
    waitKey(1);
}

bool Initialseed :: checkThreshold(Mat frame, double relation){

    Mat Frame = frame.clone();
    int width = Frame.cols;
    int height = Frame.rows;
    Regiongrowing RTest;
    //Counter CTest;
    bool threshold_qualified(true);
    int iterationnum = 0;
    bool repeat_thres = false;
    vector<double> Thresholdstack;
    Mat Mattest, Framewithcounter, Matcounter;
    

    while ((iterationnum <Threshoditerationmax) && (!repeat_thres)) {
        
        
        Mat Mattest_Blur = Frame.clone();
        Mattest = RTest.RegionGrow(Frame, Mattest_Blur , this->differencegrow, this->initialseedvektor);
        Matcounter  = RTest.Matcounter(Mattest, this->color);
        Framewithcounter = RTest.FindCounter(Mattest, Frame, this->color);
        imshow("New object", Framewithcounter);
        waitKey(100);
        
        //cout<< "iterationnum: " << iterationnum <<"/ Area: " << RTest.Area <<"/ Threshold: "<< this->differencegrow << endl;
        
        Thresholdstack.push_back(this->differencegrow) ;

        if (RTest.Area < (width*height/300)) this->differencegrow = (this->differencegrow + thresholdstep);  // (Width*Height/300) = 2000   (Width*Height/50 )= 10000
        else if (RTest.Area <= (width*height/40) ) break;
        else  this->differencegrow = (this->differencegrow - thresholdstep);

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
        cout<< endl << "This new random point can not become a available seed point " << endl;
        threshold_qualified = false;
    }

    else{
        // create new object sucessfully
        threshold_qualified = true;
        cout<< endl << "New object sucessfully found" << endl;
        cout<< "iterationnum: " << iterationnum <<"/ pixel Area: " << RTest.Area << "/ Real Area: " <<  RTest.Area * relation * relation <<"/ Threshold: "<< this->differencegrow << endl;
        this->LoopThreshold = 1;

        data[0].push_back(RTest.EWlong);    // Green line. long axis
        this->data[1].push_back(RTest.EWshort);   // lightly blue line . short axis
     
        this->data[2].push_back(1.0); // scale
        this->data[3].push_back(initialScalediff); // ScaleDifference
        
        this->data[4].push_back(RTest.Area);  // Area
        this->data[5].push_back(RTest.Area * relation * relation);  // Realarea
        
        this->data[6].push_back(1.0); // Scale Realarea
        this->data[7].push_back(initialScalediff); // Realarea ScaleDifference ?? 刚开始设为多少
        
        this->preCounter  = Matcounter.clone();
        this->preSegment  = Mattest.clone();
        
        this->initialseedvektor.clear();
        this->initialseedvektor.push_back(RTest.cntr);
        this->threshold_notchange = true;
        
        this->Contourvector.assign(RTest.Contourvector.begin(), RTest.Contourvector.end());
        cout<< RTest.Contourvector.size()<<endl;
    }
    //destroyWindow("Contour of Segment");
    
    return threshold_qualified;
}




