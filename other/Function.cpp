//
//  Function.cpp
//  video_segment
//
//  Created by Yanbo Zhu on 2018/2/5.
//  Copyright © 2018年 Zhu. All rights reserved.
//

#include "Function.hpp"

double decomposematrix(Mat H){
    double averageScale = 1.0;
    if(!H.empty()){
        //        Mat Traslation = H.col(2);
        //        cout<<"Traslation:" <<endl << Traslation <<endl<< endl;;
        
        Mat A = Mat::zeros(2, 2, CV_64F);
        for (int j = 0; j< A.rows; j++)
        {
            for (int i = 0; i< A.cols; i++)
            {
                A.at<double>(j, i) = H.at<double>(j, i);
            } // end of line
        }
        
        double p =  sqrt( ( H.at<double>(0,0) * H.at<double>(0,0) ) + ( H.at<double>(0,1)*H.at<double>(0,1) ) );
        double r = determinant(A)/p;
        
        Mat Scale = Mat::zeros(2, 2, CV_64F);
        Scale.at<double>(0, 0) = p;
        Scale.at<double>(1, 1) = r;
        //cout<<"Scale:" <<endl << Scale <<endl << endl;
        
        averageScale = (p+r)/2;
        //cout<<"averageScale:" <<endl << averageScale <<endl << endl;
        
        //        double angle = atan2(H.at<double>(1,0),H.at<double>(0,0));
        //        cout<<"angle:" <<endl << angle * 180/M_PI <<endl << endl;
        
        //        double shear = ( H.at<double>(0,0) * H.at<double>(1,0)  + H.at<double>(0,1)* H.at<double>(1,1) ) / determinant(A) ;
        //        Mat Shearmatrix = Mat::eye(2, 2, CV_64F);
        //        Shearmatrix.at<double>(0, 1) = shear;
        //        cout<<"Shearmatrix:" <<endl << Shearmatrix <<endl << endl;
        
    }
    
    else{
        cout<<"Homography matrix is empty" << endl;
    }
    
    return averageScale;
}
