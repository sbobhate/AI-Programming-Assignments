//
//  main.cpp
//  Programming Assignment 1
//
//  Created by Shantanu Bobhate on 1/26/16.
//  Copyright © 2016 Shantanu Bobhate. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, const char * argv[]) {
    
    // Begin a video capture
    cv::VideoCapture cap(0);
    // Make sure the camera opened
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open camera.\n";
        return -1;
    }
    
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
        
        cv::imshow("Output", frame);
    }
    
    // Release all resources used by this window
    cv::destroyWindow("Output");
    
    return 0;
}
