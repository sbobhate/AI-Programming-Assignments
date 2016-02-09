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
// Function to find the area of a blob
int findArea ( cv::Mat& src );
// Function to find the centroid of an object
cv::Point findCenter ( int area, cv::Mat& src, cv::Rect loc );
// Function to set the background of an image
void setBackground ( cv::Mat& src, int color[] );
// Function to draw circles in an image
void drawCircles ( cv::Mat& src, int number );


int number_of_fingers;


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
     * Skin Detection
     */
        cv::Mat skin_detected_frame = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        mySkinDetect(frame, skin_detected_frame);
        cv::imshow("Skin Detection", skin_detected_frame);

    /*
     * Denoising
     */
        /*
        cv::Mat filtered_image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
        cv::blur(skin_detected_frame, filtered_image, cv::Size(10, 10));
        cv::threshold(filtered_image, filtered_image, 200, MAX_THRESHOLD, 0);
        cv::dilate(filtered_image, filtered_image, cv::Mat());
        cv::imshow("Filter", filtered_image);
         */
        
    /*
     * Draw Contours on Skin Detected Image
     */
        // Vector to hold contours
        std::vector < std::vector< cv::Point > > contours;
        // Vector to hold hierarchy
        std::vector < cv::Vec4i > hierarchy;
        // Clone original frame
        cv::Mat skin_cpy = skin_detected_frame.clone();
        // Find contours in clone
        cv::findContours(skin_cpy, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        // Create destination frame to show the contours
        cv::Mat contour_output = cv::Mat::zeros(skin_cpy.size(), CV_8UC3);
        // Variable to hold the largest area
        int max_size = 0;
        // Variable to hold the index of the contour with the largest area
        int idx = 0;
        // Bounding rectangle for the largest contour
        cv::Rect bounding_rect;
        // Find the largest contour
        for (int i = 0; i < contours.size(); i++)
        {
            // Compute the area of the current contour
            double area = contourArea(contours[i]);
            // Find the contour with the largest area
            if (area > max_size) {
                // Set the new max area
                max_size = area;
                // Store the index of the contour
                idx = i;
                // Create the bounding rectangle for this contour
                bounding_rect = cv::boundingRect(contours[i]);
            }
        }
        // Draw contours filled in (Blue)
        drawContours(contour_output, contours, idx, cv::Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
        // Draw the contour outline (Red)
        cv::drawContours(contour_output, contours, idx, cv::Scalar(0, 0, 255));
        // Draw the bounding rectangle for the contour (Green)
        rectangle(frame, bounding_rect, cv::Scalar(0, 255, 0), 1, 8, 0);
        // Make sure there exist contours
        if (!contours.empty()) {
            // Compute the min rectangle to bound the hand
            cv::RotatedRect min_rect = cv::minAreaRect(contours[idx]);
            // Get the points for this rectangle
            cv::Point2f rect_points[4];
            min_rect.points(rect_points);
            // Draw the rotated rectangle
            for (int ii = 0; ii < 4; ++ii) {
                cv::line(frame, rect_points[ii], rect_points[(ii + 1) % 4], cv::Scalar(0, 255, 255), 2, 8);
            }
        }
        
    /*
     * Find the center of the largest contour
     */
        int gesture = 0;
        // Make sure there exists a contour
        if (bounding_rect.width != 0 && bounding_rect.height != 0) {
            // Find the region of interest using the bounding rectangle
            cv::Mat roi = skin_detected_frame(bounding_rect);
            // Show this region of interest
            cv::imshow("Region Of Interest", roi);
            // Find the area of the object in the region of interest
            int area = findArea(roi);
            // Find the centroid of the object
            cv::Point center = findCenter(area, skin_detected_frame, bounding_rect);
            int x_bar = center.x;
            int y_bar = center.y;
            // Find the distance between the top of the bounding rectangle to the center
            int top_dist = y_bar - bounding_rect.y;
            // Find the distance between the bottom of the bounding rectangle to the center
            int bot_dist = bounding_rect.y + bounding_rect.height - y_bar;
            // Print out the top_dist and the bot_dist
            std::cout << "Top Dist: " << top_dist << "; Bot Dist: " << bot_dist << std::endl;
            // Print out the area
            std::cout << "Area: " << area << std::endl;
            // Print the dimensions
            std::cout << "Width: " << bounding_rect.width << "; Height: " << bounding_rect.height << std::endl;
            // Check for hand gestures
            if (area > 100 && top_dist > bot_dist && top_dist - bot_dist > 25 && bounding_rect.width < 190) gesture = 1;
            else if (area > 100 && top_dist < bot_dist && bot_dist - top_dist > 25 && bounding_rect.width < 190) gesture = -1;
            else gesture = 0;
            // Represent the center of the object with a centroid
            cv::circle(frame, cv::Point(x_bar, y_bar), 3, cv::Scalar(0, 0, 255));
        }
        
    /*
     * Convex Hull and Convexity Defects
     */
        // Vector to hold the hull points
        std::vector < std::vector < cv::Point > > hull (1);
        // Vector to hold the hull integer values
        std::vector < int > hullI;
        // Vector to hold the points of defect
        std::vector < cv::Vec4i > defects;
        if (!contours.empty()) {
            cv::convexHull(cv::Mat(contours[idx]), hull[0]);
            cv::convexHull(cv::Mat(contours[idx]), hullI);
        }
        // More than 3 indicies are needed
        if (hullI.size() > 3) {
            cv::convexityDefects(contours[idx], hullI, defects);
        }
        // Draw the convex hull
        cv::drawContours(frame, hull, 0, cv::Scalar(0, 255, 0));
        // Show the output
        cv::imshow("Contours", contour_output);
        
    /*
     * Draw Convexity Defects
     */
        int finger_count = 0;
        for (const cv::Vec4i& v : defects) {
            float depth = v[3] / 256;
            if (depth > 10 && depth < 80) {
                int start_idx = v[0];
                cv::Point start = contours[idx][start_idx];
                int end_idx = v[1];
                cv::Point end = contours[idx][end_idx];
                int far_idx = v[2];
                cv::Point far = contours[idx][far_idx];

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

    /*
     * Hand Gestures
     */
        cv::Mat data = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
        int color[3];
        if (gesture == 0) {
            color[0] = 255; color[1] = 255; color[2] = 255; // White
        } else if (gesture == 1) {
            color[0] = 0; color[1] = 255; color[2] = 0;     // Green
        } else if (gesture == -1) {
            color[0] = 0; color[1] = 0; color[2] = 255;     // Red
        } else {
            color[0] = 0; color[1] = 255; color[2] = 255;   // Yellow
        }
        setBackground(data, color);
        if (number_of_fingers != finger_count) {
            drawCircles(data, finger_count);
            number_of_fingers = finger_count;
        }
        cv::imshow("Output", data);
        
        // Show the images
        cv::imshow("Original", frame);
    }
    
    //  Release all resources used by the video capture
    cap.release();
    // Release all resources used by all windows
    cv::destroyAllWindows();
    
    return 0;
}

///////////////
/*           //
 * Functions //
 */          //
///////////////

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
cv::Point findCenter (int area, cv::Mat& src, cv::Rect loc) {
    int rect_x = loc.x, rect_y = loc.y, rect_width = loc.width, rect_height = loc.height;
    int sum_x = 0, sum_y = 0;
    for (int ii = rect_y; ii < rect_y + rect_height; ii++)
    {
        uchar* p = src.ptr<uchar>(ii);
        for (int jj = rect_x; jj < rect_x + rect_width; jj++)
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

void setBackground (cv::Mat& src, int color[]) {
    for (int ii = 0; ii < WINDOW_HEIGHT; ++ii) {
        cv::Vec3b* pixel = src.ptr<cv::Vec3b>(ii);
        for (int jj = 0; jj < WINDOW_WIDTH; ++jj) {
            pixel[jj][0] = (uchar) color[0];
            pixel[jj][1] = (uchar) color[1];
            pixel[jj][2] = (uchar) color[2];
        }
    }
}

void drawCircles (cv::Mat& src, int number) {
    for (int ii = 0; ii < number; ++ii) {
        int x = std::rand() % WINDOW_WIDTH;
        int y = std::rand() % WINDOW_HEIGHT;
        cv::circle(src, cv::Point(x, y), 10, cv::Scalar(0, 0, 0), CV_FILLED, 8, 0);
    }
}

/*
 * References:
 *  - Motion Detection and Tracking: http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
 *  - Common OpenCV Functions: http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
 *  - OpenCV Background Subtraction Methods: http://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html#gsc.tab=0
 *  - Hand Gesture Recognition using Convex Hull: http://anikettatipamula.blogspot.ro/2012/02/hand-gesture-using-opencv.html
 *  - How to Draw Convexity Defects: http://stackoverflow.com/questions/31354150/opencv-convexity-defects-drawing
 *  - Counting Finger Tips using Convexity Defects: http://www.codeproject.com/Articles/782602/Beginners-guide-to-understand-Fingertips-counting
 */
