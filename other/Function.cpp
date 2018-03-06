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


double averagevalue(int num, vector<double> array){
    double average = array.back();
    vector<double>::reverse_iterator it;
    
    if (array.size()< num){
        for(it = array.rbegin(); it!= array.rend(); it++){
            average = (average + *it)/2.0;
        }
    }
    
    else {
        for(it = array.rbegin(); it!= array.rbegin() + num; it++){
            average = (average + *it)/2.0;
        }
    }
    return average;
}


double averagedifference(int num, vector<double> array){
    double average = abs(array.back());
    vector<double>::reverse_iterator it;
    
    if (array.size()< num ){
        for(it = array.rbegin(); it!= array.rend(); it++){
            average = (average + abs(*it))/2.0;  // 此处有abs
        }
    }
    
    else {
        for(it = array.rbegin(); it!= array.rbegin() + num; it++){
            average = (average + abs(*it))/2.0;
        }
    }
    return average;
}

double pixeldistance(Mat img, Point2f p1, Point2f p2)
{
    //    int num = (int)(pv.size()/2);
    //    double pixeldistacne[num];
    //    for (int i=0; i< num ; i++){
    //        double a = pv[2*i].x - pv[2*i +1].x;
    //        double b = pv[2*i].y - pv[2*i +1].y;
    //        pixeldistacne[i] = sqrt(a*a+b*b);
    //    }
    //
    //    return pixeldistacne;
    
    //    if(pv.size() == 2){
    //        double a = pv[0].x - pv[1].x;
    //        double b = pv[0].y - pv[1].y;
    //        line(img, pv[0], pv[1], Scalar(0, 0, 210),2,8,0); //随机颜色
    //        return(sqrt(a*a+b*b)); // a^2 not equal to a*a. a^2 has differnt meaning in Opencv
    //    }
    //
    //    else{
    //        cout<< "vector (pixel-relation) size is not 2" <<endl;
    //        return 0 ;
    //    }
    
    double a = p1.x - p2.x;
    double b = p1.y - p2.y;
    line(img, p1, p2, Scalar(200, 0, 0),2,8,0);
    return(sqrt(a*a+b*b)); // a^2 not equal to a*a. a^2 has differnt meaning in Opencv
}


int remove_directory(const char *path)
{
    DIR *d = opendir(path);
    size_t path_len = strlen(path);
    int r = -1;
    
    if (d)
    {
        struct dirent *p;
        r = 0;
        
        while (!r && (p=readdir(d)))
        {
            int r2 = -1;
            char *buf;
            size_t len;
            
            /* Skip the names "." and ".." as we don't want to recurse on them. */
            if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")){
                continue;
            }
            
            len = path_len + strlen(p->d_name) + 2;
            buf = static_cast<char*>(malloc(len));
            
            if (buf)
            {
                struct stat statbuf;
                snprintf(buf, len, "%s/%s", path, p->d_name);
                
                if (!stat(buf, &statbuf))
                {
                    if (S_ISDIR(statbuf.st_mode)) {  r2 = remove_directory(buf);  }
                    else { r2 = unlink(buf); }
                }
                
                free(buf);
            }
            
            r = r2;
        }
        
        closedir(d);
    }
    
    if (!r){ r = rmdir(path); }
    
    return r;
}
