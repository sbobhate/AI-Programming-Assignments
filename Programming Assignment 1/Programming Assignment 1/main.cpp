//
//  main.cpp
//  Programming Assignment 1
//
//  Created by Shantanu Bobhate on 1/26/16.
//  Collaborated with Inna
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

/*
 * Define Macros
 */
#define WINDOW_HEIGHT 240
#define WINDOW_WIDTH 320
#define THRESHOLD 128
#define MAX_THRESHOLD 255

/*
 * Declare Functions
 */
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
// Function to find the orientation of an object
int findOrientation (cv::Mat& src, int x_bar, int y_bar);
void putCircle (cv::Mat& src);




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
    cap.set(CV_CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT);
    
    // Create a Mat to hold a frame from the video capture
    cv::Mat frame;
    
    sleep(1);
    
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
     * Put Circle
     */
//        putCircle(frame);
        
    /*
     * Skin Detection
     */
        cv::Mat skin_detected_frame = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        mySkinDetect(frame, skin_detected_frame);
        cv::imshow("Skin Detection", skin_detected_frame);

    /*
     * Denoising
     */
        // Method 1
        cv::Mat filtered_image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        cv::blur(skin_detected_frame, filtered_image, cv::Size(10, 10));
        cv::threshold(filtered_image, filtered_image, 200, MAX_THRESHOLD, 0);
        cv::dilate(filtered_image, filtered_image, cv::Mat());
        cv::imshow("Filter", filtered_image);

    /*
     * Draw Contours on Skin Detected Image
     */
        // Vector to hold contours
        std::vector<std::vector<cv::Point>> contours;
        // Vector to hold hierarchy
        std::vector<cv::Vec4i> hierarchy;
        // Clone original frame
        cv::Mat skin2 = skin_detected_frame.clone();
        // Find contours in clone
        cv::findContours(skin2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        // Print out the number of contours detected
        std::cout << "The number of contours detected is: " << contours.size() << std::endl;
        // Create destination frame to show the contours
        cv::Mat contour_output = cv::Mat::zeros(skin_detected_frame.size(), CV_8UC3);
        // Find largest contour
        int max_size = 0;
        int idx = 0;
        cv::Rect bounding_rect;
        for (int i = 0; i < contours.size(); i++)
        {
            // Compute the area of the current contour
            double area = contourArea(contours[i]);
            // Find the contour with the largest area
            if (area > max_size) {
                max_size = area;
                idx = i;
                bounding_rect = cv::boundingRect(contours[i]);
            }
        }
        // Draw contours
        drawContours(contour_output, contours, idx, cv::Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
        cv::drawContours(contour_output, contours, idx, cv::Scalar(0, 0, 255));
        rectangle(frame,bounding_rect, cv::Scalar(0, 255, 0), 1, 8, 0);
        
        cv::namedWindow("Contours", CV_WINDOW_AUTOSIZE);
        imshow("Contours", contour_output);
        
        /*
         * Compute Area
         */
        
//        int area = findArea(contour_output);
//        std::cout << area << std::endl;
//        cv::Point center = findCenter(area, contour_output);
//        cv::circle(frame, center, 1, cv::Scalar(0, 0, 255));
//        std::cout << findOrientation(contour_output, center.x, center.y) << std::endl;
        
//        int sum_x = 0, sum_y = 0;
//        int area = 1;
//        if (bounding_rect.width != 0 && bounding_rect.height != 0) {
//            cv::Mat roi = cv::Mat::zeros(bounding_rect.height, bounding_rect.width, CV_8UC1);
//            roi = skin_detected_frame(cv::Rect(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height));
//            cv::imshow("ROI", roi);
//            area = findArea(roi);
//        }
//
//        for (int ii = bounding_rect.y; ii < bounding_rect.y + bounding_rect.height; ii++)
//        {
//            uchar* p = skin_detected_frame.ptr<uchar>(ii);
//            for (int jj = bounding_rect.x; jj<bounding_rect.x + bounding_rect.width; jj++)
//            {
//                if (p[jj] == 255) {
//                    sum_x += jj;
//                    sum_y += ii;
//                }
//            }
//        }
//        int x_bar = 0, y_bar = 0;
//        if (area != 0) {
//            x_bar = sum_x / area;
//            y_bar = sum_y / area;
//        }
//
//        int top_dist = y_bar - bounding_rect.y;
//        int bot_dist = bounding_rect.y + bounding_rect.height - y_bar;
//        
//        std::cout << "Top Dist: " << top_dist << "; Bot Dist: " << bot_dist << std::endl;
//        std::cout << "Area: " << area << std::endl;
//        if (area < 100) cv::putText(frame, "No Hand", cv::Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
//        else if (top_dist > bot_dist && top_dist - bot_dist > 25) cv::putText(frame, "Thumbs Up", cv::Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
//        else if (top_dist < bot_dist && bot_dist - top_dist > 25) cv::putText(frame, "Thumbs Down", cv::Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
//        else  cv::putText(frame, "Fist", cv::Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
//        
//        cv::circle(frame, cv::Point(x_bar, y_bar), 2, cv::Scalar(0, 0, 255));
        
    /*
     * Convex Hull and Convexity Defects
     */
        // Find the convex hull object for each contour
        /* METHOD 1*/
        /*
        std::vector < std::vector < cv::Point > > hull (contours.size());
        std::vector < std::vector < int > > hullI (contours.size());
        std::vector < std::vector < cv::Vec4i > > defects (contours.size());
        
        for (int ii = 0; ii < contours.size(); ++ii) {
            cv::convexHull(cv::Mat(contours[ii]), hull[ii]);
            cv::convexHull(cv::Mat(contours[ii]), hullI[ii]);
            // More than 3 indicies are needed
            if (hullI.size() > 3) {
                cv::convexityDefects(contours[ii], hullI[ii], defects[ii]);
            }
        }
        cv::drawContours(frame, hull, 0, cv::Scalar(255, 0, 0));
        cv::imshow("Hull", contour_output);
        */
        /* METHOD 2 */
        std::vector < std::vector < cv::Point > > hull (1);
        std::vector < int > hullI;
        std::vector < cv::Vec4i > defects;
        if (!contours.empty()) {
            cv::convexHull(cv::Mat(contours[idx]), hull[0]);
            cv::convexHull(cv::Mat(contours[idx]), hullI);
        }
        // More than 3 indicies are needed
        if (hullI.size() > 3) {
            cv::convexityDefects(contours[idx], hullI, defects);
        }
        cv::drawContours(frame, hull, 0, cv::Scalar(255, 0, 0));
        cv::imshow("Hull", contour_output);
        
    /*
     * Draw Convexity Defects
     */
        /* METHOD 1 */
        /*
        for (int ii = 0; ii < contours.size(); ++ii)  {
            for (const cv::Vec4i& v : defects[ii]) {
                float depth = v[3] / 256;
                if (depth > 10) {
                    int start_idx = v[0];
                    cv::Point start = contours[ii][start_idx];
                    int end_idx = v[1];
                    cv::Point end = contours[ii][end_idx];
                    int far_idx = v[2];
                    cv::Point far = contours[ii][far_idx];
                    
                    cv::line(frame, start, end, cv::Scalar(0, 150, 0));
                    cv::line(frame, start, far, cv::Scalar(0, 150, 0));
                    cv::line(frame, end, far, cv::Scalar(0, 150, 0));
                    cv::circle(frame, far, 4, cv::Scalar(0, 150, 0));
                }
            }
        }
        */
        /* METHOD 2 */
        int finger_count = 0;
        for (const cv::Vec4i& v : defects) {
            float depth = v[3] / 256;
            if (depth > 20 && depth < 80) {
                int start_idx = v[0];
                cv::Point start = contours[idx][start_idx];
                int end_idx = v[1];
                cv::Point end = contours[idx][end_idx];
                int far_idx = v[2];
                cv::Point far = contours[idx][far_idx];
                    
//                cv::line(frame, start, end, cv::Scalar(0, 255, 0), 2);
                cv::line(frame, start, far, cv::Scalar(0, 255, 0), 2);
                cv::line(frame, end, far, cv::Scalar(0, 255, 0), 2);
                cv::circle(frame, start, 4, cv::Scalar(100, 0, 255), 2);
                finger_count++;
            }
        }
        std::cout << finger_count << std::endl;
        std::stringstream ss;
        ss << finger_count;
        cv::putText(frame, ss.str(), cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        
//        std::vector < int > h;
//        std::vector < cv::Vec4i > defects;
//        cv::convexHull(contours[idx], h);
//        cv::convexityDefects(contours[idx], h, defects);
//        
//        for (int ii = 0; ii < defects.size(); ++ii) {
//            cv::Vec4i value = defects[ii];
//            int s = value[0];
//            int e = value[1];
//            cv::line(frame, contours[s][0], contours[e][0], cv::Scalar(150, 0, 150));
//        }
        
    /*
     * Hand Gestures
     */
//        int fingerCount = 1;
//        for (int i = 0; i< defects.size(); i++){
//            int start_index = defects[i][0];
//            CvPoint start_point = contours[idx][start_index];
//            int end_index = defects[i][1];
//            CvPoint end_point = contours[idx][end_index];
//            double d1 = (end_point.x - start_point.x);
//            double d2 = (end_point.y - start_point.y);
//            double distance = sqrt((d1*d1)+(d2*d2));
//            int depth_index = defects[i][2];
//            int depth =  defects[i][3]/1000;
//            
//            if (depth > 10 && distance > 2.0 && distance < 200.0){
//                fingerCount ++;
//            }
//        }
//        
//        std::cout << fingerCount << std::endl;
        
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
    cv::Mat gray_first = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
    cv::Mat gray_second = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
    cv::cvtColor(first, gray_first, CV_BGR2GRAY);
    cv::cvtColor(second, gray_second, CV_BGR2GRAY);
    // Iterate through each pixel sequentially
    for (int ii = 0; ii < WINDOW_HEIGHT; ++ii) {
        uchar* pixel_1 = gray_first.ptr<uchar>(ii);
        uchar* pixel_2 = gray_second.ptr<uchar>(ii);
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

// Function to find the area of an object
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

// Function to find the center of an object
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

// Function to find the orientation of an object
int findOrientation (cv::Mat& src, int x_bar, int y_bar) {
    int alpha = 0, beta = 0, gamma = 0;
    for (int ii = 0; ii < WINDOW_HEIGHT; ++ii) {
        for (int jj = 0; jj < WINDOW_WIDTH; ++jj) {
            int s = ii - x_bar;
            int t = jj - y_bar;
            alpha += s*s;
            beta += s*t;
            gamma = t*t;
        }
    }
    return atan(beta / (alpha - gamma));
}

void putCircle (cv::Mat& src) {
    int rand_x = rand() % WINDOW_WIDTH;
    int rand_y = rand() % WINDOW_HEIGHT;
    std::cout << "(" << rand_x << ", " << rand_y << ")" << std::endl;
    cv::circle(src, cv::Point(rand_x, rand_y), 30, cv::Scalar(255, 255, 150));
}

/*
 * References:
 *  - Motion Detection and Tracking: http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
 *  - Common OpenCV Functions: http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
 *  - OpenCV Background Subtraction Methods: http://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html#gsc.tab=0
 *  - Hand Gesture Recognition using Convex Hull: http://anikettatipamula.blogspot.ro/2012/02/hand-gesture-using-opencv.html
 *  - How to Draw Convexity Defects: http://stackoverflow.com/questions/31354150/opencv-convexity-defects-drawing
 */
