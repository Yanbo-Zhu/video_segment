//
//  Regiongrowing.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 09.10.17.
//  Copyright © 2017 Zhu. All rights reserved.
//

#include "Regiongrowing.hpp"


Mat Regiongrowing:: RegionGrow(Mat MatIn, Mat MatBlur , double iGrowJudge, vector<Point> seedset) //iGrowPoint: seeds 为种子点的判断条件，iGrowJudge: growing condition 为生长条件
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
        //seedtogether.push_back(seedset[i])
    }
    
    seedtogether.clear();
    seedtogether = seedset;
    
    //生长方向顺序数据
    //int DIR[8][2]={{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
    int DIR[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
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
            if (nextseed.x < 0 || nextseed.y < 0 || nextseed.x > (MatIn.cols-2) || (nextseed.y > MatIn.rows-2))
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
        //vw << MatGrownow;
    }

    regioncenter  = centerpoint(seedtogether);
    //cout<<"regioncenter: " << regioncenter <<endl;
    
    //seedtogether.clear();
    //vw.release();
    //imshow("Segment", Segment);
    
    //Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    //dilate(Segment,Segment, element);
    //imshow("Segment after dalite", Segment);
    //erode(Segment,Segment, element);
    //imshow("Segment after erode", Segment);
    //morphologyEx(Segment,Segment, MORPH_CLOSE, element);
    //waitKey(0);
    
    return Segment;
}

//--------------------------  difference value duction ---------
double Regiongrowing:: differenceValue(Mat MatIn, Point oneseed, Point nextseed, int DIR[][2], double rowofDIR, double B, double G, double R)
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
Point  Regiongrowing:: centerpoint(vector<Point> seedtogetherBackup){
    
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

Regiongrowing:: ~Regiongrowing(){
//    //cout << endl;
//    cout << "Klasse Regiongrowing " << regioncenter <<" wurde geloescht." << endl;
}



