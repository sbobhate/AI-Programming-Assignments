//
//  main.cpp
//  Programming Assignment 1
//
//  Created by Shantanu Bobhate on 1/26/16.
//  Copyright © 2016 Shantanu Bobhate. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

/* 
 * Objectives to accomplish:
 *  - delineate hand shapes (fist, thumbs up, thumbs down, pointing)
 *  - gestures (waving one or both hands, swinging, drawing something)
 *  - create a graphical display that responds to the recognition
 */

int main(int argc, const char * argv[]) {
    
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
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    
    // Create a Mat to hold the first frame (base image)
    cv::Mat first_frame;
    // Create a Mat to hold a frame from the video capture
    cv::Mat frame;
    // Create a window to show the output
    cv::namedWindow("Output", CV_WINDOW_NORMAL);
    
    // Exit when user clicks 'esc'
    while (cv::waitKey(50) != 27) {
        // Grab a frame
        cap >> frame;
        
        // Make sure the image is not empty
        if (frame.empty()) {
            std::cout << "Error: Could not find a frame.\n";
            return -1;
        }
        
        // Convert the image from color to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, CV_BGR2GRAY);
        // Apply Gaussian smoothing to average pixel intensities accross an 11 x 11 region
        cv::GaussianBlur(gray, gray, cv::Size(21, 21), 0);
        
        // Capture a base image
        if (first_frame.empty()) {
            first_frame = gray;
        }
        
        // Compute the absolute difference between the 2 frames
        cv::Mat delta;
        cv::absdiff(first_frame, gray, delta);
        // Convert the absolute difference to a binary image
        cv::Mat threshold_image;
        cv::threshold(delta, threshold_image, 25, 255, CV_THRESH_BINARY);
        
        // Extract the contours for the moving object
        // NOTE: Uncomment for contour extraction
        /*
        std::vector < std::vector < cv::Point > > contours;
        cv::findContours(threshold_image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        
        for (int ii = 0; ii < contours.size(); ++ii) {
            std::vector<cv::Point> c = contours[ii];
            cv::Rect rectangle = boundingRect(c);
            cv::rectangle(frame, rectangle.tl(), rectangle.br(), cv::Scalar(0, 255, 0));
        }
         */
        
        /*
         *
         * Add code to implement objectives
         *
         */
        
        /*
         * Hand Gestures
         */
        
        /* 
         * Skin Detection 
         */
        
        // Show the images
        cv::imshow("Original", frame);
        cv::imshow("Grayscale", gray);
        cv::imshow("Difference", delta);
        cv::imshow("Threshold", threshold_image);
    }
    
    //  Release all resources used by the video capture
    cap.release();
    // Release all resources used by all windows
    cv::destroyAllWindows();
    
    return 0;
}

/*
 * References:
 *  - http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
 */
