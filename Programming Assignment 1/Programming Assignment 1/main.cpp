//
//  main.cpp
//  Programming Assignment 1
//
//  Created by Shantanu Bobhate on 1/26/16.
//  Collaborated with Inna
//  Copyright © 2016 Shantanu Bobhate. All rights reserved.
//
//

#include <iostream>
#include <opencv2/opencv.hpp>

/* 
 * Objectives to accomplish:
 *  - delineate hand shapes (fist, thumbs up, thumbs down, pointing)
 *  - gestures (waving one or both hands, swinging, drawing something)
 *  - create a graphical display that responds to the recognition
 */

#define WINDOW_HEIGHT 240
#define WINDOW_WIDTH 320

// Function to detect skin in a frame
void mySkinDetect ( cv::Mat& src, cv::Mat& dst );
// Function that returns the maximum of 3 integers
int myMax ( int a, int b, int c );
// Function that returns the minimum of 3 integers
int myMin ( int a, int b, int c );
// Function to find the difference between 2 frames
void myFrameDifferencing ( cv::Mat& first, cv::Mat& second, cv::Mat& dst, int threshold );
// Functiont to find the motion energy given a set of frames
void myMotionEnergy ( std::vector < cv::Mat > frames, cv::Mat &dst, int threshold );
// Function to find the area of a blob
int findArea (cv::Mat& src);
// Function to find the centroid of an object
cv::Point findCenter (int area, cv::Mat& src);
//Finding projections
int verticalProjection(cv::Mat& src, cv::Mat& dst);
int horizontalProjection(cv::Mat& src, cv::Mat& dst);

cv::Mat fgMaskMOG2;
cv::Ptr<cv::BackgroundSubtractor> pMOG2;

int main(int argc, const char * argv[]) {
    
    // Create BackgroundSubtractor objects
    pMOG2 = cv::createBackgroundSubtractorMOG2();
    
    // Begin a video capture
    cv::VideoCapture cap(0);
    // Make sure the camera opened
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open camera.\n";
        return -1;
    }
    
    // Get the height and width of the frame
    int cap_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int cap_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    std::cout << "Width: " << cap_width << " ; Height: " << cap_height << std::endl;
    
    // Resize the image so it doesn't take up the whole frame
    cap.set(CV_CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT);
    
    // Create a Mat to hold a frame from the video capture
    cv::Mat frame;
    
//    sleep(0.5);
//    cv::Mat frame0;
//    cv::Mat base = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
//    cap.read(frame0);
//    cv::cvtColor(frame0, frame0, CV_BGR2GRAY);
//    for (int ii = 0; ii < WINDOW_HEIGHT; ++ii) {
//        uchar* pixel = frame0.ptr<uchar>(ii);
//        uchar* pixel_base = base.ptr<uchar>(ii);
//        for (int jj = 0; jj < WINDOW_WIDTH; ++jj) {
//            if (pixel[jj] > 128) {
//                pixel_base[jj] = 255;
//            } else {
//                pixel_base[jj] = 0;
//            }
//        }
//    }
    
    // Exit when user clicks 'esc'
    while (cv::waitKey(50) != 27) {
        // Grab a frame
        bool success = cap.read(frame);
        
        // Make sure the image is not empty
        if (!success) {
            std::cout << "Error: Could not read frame.\n";
            return -1;
        }
        
        /*
         *
         * Add code to implement objectives
         *
         */
        
        /*
         * Background Subtraction
         */
        // Update the background model
        pMOG2 -> apply(frame, fgMaskMOG2);
        // Get the frame number and write it on the current frame
        std::stringstream ss;
        cv::rectangle(frame, cv::Point(10, 2), cv::Point(100, 20), cv::Scalar(255, 255, 255));
        ss << cap.get(CV_CAP_PROP_POS_FRAMES);
        std::string frameNumberString = ss.str();
        cv::putText(frame, frameNumberString.c_str(), cv::Point(15, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        // Show the current frame and the fg Mask
//        cv::imshow("Frame", frame);
//        cv::imshow("Mask", fgMaskMOG2);
        
        /*
         * Compute Area
         */
        int area = findArea(fgMaskMOG2);
        std::cout << area << std::endl;
        cv::Point center = findCenter(area, fgMaskMOG2);
        cv::circle(frame, center, 1, cv::Scalar(0, 0, 255));
        
        /*
         * Skin Detection
         */
        cv::Mat skin_detected_frame = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        mySkinDetect(frame, skin_detected_frame);
        cv::imshow("Skin Detection", skin_detected_frame);
        cv::Mat frame_test;
        cap.read(frame_test);
        cv::Mat frame2 = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        mySkinDetect(frame_test, frame2);
        cv::Mat diff = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        myFrameDifferencing(skin_detected_frame, frame2, diff, 128);
        cv::imshow("Diff", diff);
        
        
        /*
         * Detecting Waving
         */
        
        int totalHorizontalPixels;
        int totalVerticalPixels;
        bool check;
        
        cv::Mat horizontalProj = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        totalHorizontalPixels = horizontalProjection(diff, horizontalProj);
        imshow("HorizontalProjection",horizontalProj);
        cv::Mat verticalProj = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        totalVerticalPixels = verticalProjection(diff, verticalProj);
        imshow("VerticalProjection",verticalProj);
        
        if (totalHorizontalPixels > 3000000 && totalVerticalPixels > 3000000)
            check = 1;
        
        if (check)
            cv::putText(frame, "DONE", cv::Point2f(50,230), CV_FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0,225,0), 2, cv::LINE_AA);
        
        /*
         * Hand Gestures
         */
        
        
        /*
         * Motion Energy
         */

        
        // Show the images
        cv::imshow("Original", frame);
    }
    
    //  Release all resources used by the video capture
    cap.release();
    // Release all resources used by all windows
    cv::destroyAllWindows();
    
    return 0;
}

// Function to detect skin in a frame
void mySkinDetect(cv::Mat& src, cv::Mat& dst) {
    // Variables to store values
    uchar r, g, b;
    // Iterate through the pixels sequentially
    for (int ii = 0; ii < src.rows; ++ii) {
        cv::Vec3b* pixel = src.ptr<cv::Vec3b>(ii);
        uchar* pixel_dst = dst.ptr<uchar>(ii);
        for (int jj = 0; jj < src.cols; ++jj) {
            r = pixel[jj][2];
            g = pixel[jj][1];
            b = pixel[jj][0];
            // Conduct the tests for Skin Detection
            if (r > 95 &&
                b > 20 &&
                g > 40 &&
                (myMax(r, g, b) - myMin(r, g, b)) > 15 &&
                abs(r - g) > 15 &&
                r > g &&
                r > b) {
                // Skin is set to white
                pixel_dst[jj] = 255;
            } else {
                pixel_dst[jj] = 0;
            }
            
        }
    }
}

//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
    return std::max(std::max(a, b), c);
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
    return std::min(std::min(a, b), c);
}

// Function to find the difference between 2 frames
void myFrameDifferencing (cv::Mat& first, cv::Mat& second, cv::Mat& dst, int threshold) {
    // Iterate through each pixel sequentially
    for (int ii = 0; ii < WINDOW_HEIGHT; ++ii) {
        uchar* pixel_1 = first.ptr<uchar>(ii);
        uchar* pixel_2 = second.ptr<uchar>(ii);
        uchar* pixel_dst = dst.ptr<uchar>(ii);
        for (int jj = 0; jj < WINDOW_WIDTH; ++jj) {
            // Compute the intensities
            int intensity_1 = pixel_1[jj];
            int intensity_2 = pixel_2[jj];
            // Compute the absolute difference
            int intensity_difference = abs(intensity_2 - intensity_1);
            // Threshold the image to convert it to binary
            if (intensity_difference > threshold) {
                pixel_dst[jj] = 255;
            } else {
                pixel_dst[jj] = 0;
            }
        }
    }
}

// Functiont to find the motion energy given a set of frames
void myMotionEnergy (std::vector < cv::Mat > frames, cv::Mat &dst, int threshold) {
    // Create a reverse iterator to traverse the vector
    std::vector < cv::Mat >::reverse_iterator rev_it = frames.rbegin();
    // Traverse the vector backwards in time
    for (; rev_it != frames.rend() - 1; ++rev_it) {
        cv::Mat second = *rev_it;
        cv::Mat first = *(rev_it + 1);
        // Create a Mat to hold the result
        cv::Mat diff = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        // Compute the difference
        myFrameDifferencing(first, second, diff, threshold);
        // Compute the sum of all the differences
        for (int xx = 0; xx < WINDOW_HEIGHT; ++xx) {
            uchar* pixel_diff = diff.ptr<uchar>(xx);
            uchar* pixel_dst = dst.ptr<uchar>(xx);
            for (int yy = 0; yy < WINDOW_WIDTH; ++yy) {
                if (pixel_dst[yy] == 0 && pixel_diff[yy] == 255) {
                    pixel_dst[yy] = 255;
                }
            }
        }
    }
}

int findArea (cv::Mat& src) {
    int area = 0;
    for (int ii = 0; ii < WINDOW_HEIGHT; ++ii) {
        uchar* pixel = src.ptr<uchar>(ii);
        for (int jj = 0; jj < WINDOW_WIDTH; ++jj) {
            if (pixel[jj] == 255) {
                area += 1;
            }
        }
    }
    return area;
}

int horizontalProjection(cv::Mat& src, cv::Mat& dst) {
    int count;
    int total;
    for (int i = 0; i<WINDOW_HEIGHT; i++) {
        count = 0;
        for (int j = 0; j<WINDOW_WIDTH; j++) {
            if (src.at<uchar>(i,j) == 255) {
                dst.at<uchar>(i,count) = 255;
                count++;
            }
            total += count;
        }
    }
    
    return total;
}

int verticalProjection(cv::Mat& src, cv::Mat& dst) {
    int count;
    int total;
    for (int i = 0; i<WINDOW_WIDTH; i++) {
        count = 0;
        for (int j = 0; j<WINDOW_HEIGHT; j++) {
            if (src.at<uchar>(i,j) == 255) {
                dst.at<uchar>(count,i) = 255;
                count++;
            }
            total += count;
        }
    }
    return total;
}

cv::Point findCenter (int area, cv::Mat& src) {
    int sum_x = 0, sum_y = 0;
    for (int ii = 0; ii< WINDOW_HEIGHT; ii++)
    {
        uchar* p = src.ptr<uchar>(ii);
        for (int jj = 0; jj<WINDOW_WIDTH; jj++)
        {
            if (p[jj] == 255) {
                sum_x += jj;
                sum_y += ii;
            }
        }
    }
    int x_bar = 0, y_bar = 0;
    if (area != 0) {
        x_bar = sum_x / area;
        y_bar = sum_y / area;
    }
    return cv::Point (x_bar, y_bar);
}

/*
 * References:
 *  - http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
 *  - http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
 *  - http://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html#gsc.tab=0
 *  - Hand Gesture Recognition using Convex Hull: http://anikettatipamula.blogspot.ro/2012/02/hand-gesture-using-opencv.html
 */
