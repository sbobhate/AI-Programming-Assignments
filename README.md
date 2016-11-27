# AI-Programming-Assignments (Hand Gesture Recognition)
Programming Assignments for the Course CAS CS 440 Artificial Intelligence (Boston University)

## Problem Definition

The focus of our assignment was to recognize hand gestures based on the number of fingers detected. Gesture recognition is a valuable part of Computer Vision and provides a medium for interaction between a user and a computer vision system. The techniques that we learned can be transformed into developing games for children, teaching them math by counting fingers or connecting numbers by moving their hands. It can also be used for controlling the computer remotely, for instance thumbs up and thumbs down gestures could be used to reply to requests from the computer (Y/N). The assumptions that we made include controlled background (the background needs to be of a color that is not close to the skin color), the main focus should be the hand and not the whole body, so the hand should be located close to the camera.

## Method and Implementation

### Basic Setup:

Our algorithm starts of by looking for skin the camera feed. We tested in front of a blue background to cancel out any background noise. We did look at a couple of background subtraction methods but did not find any that worked well for our purposes. The result of skin detection yields a binary image with skin colored to white.

The next step is for the algorithm to extract contours from the image with the skin detected. During contour extract the algorithm keeps track of the largest contour detected. This step helped us eliminate irregularities in the contour extraction.

### Shape Delineation:

We decided to look for thumbs up and thumbs down shapes in a frame. We started by using the contour to get a bounding box for the region of interest. We then calculated the area of the object in this region of interest by using our function called ‘findArea’ that used the formula for the zeroth moment to compute the result.

We then created a function called ‘findCenter’ to compute the centroid of this object. We used the formula for the first moment to compute x_bar and y_bar. We then computed the distance between the top of the bounding rectangle and the centroid and the bottom of the bounding rectangle and the centroid. We tested the values and found a threshold that worked the best to detect the gestures. We used another image to display a green background for a thumbs up, a red background for a thumbs down and a white background for everything else.Fig 1 and Fig 2 display the results.

We then tested various other gestures to determine any false positives our system hadn’t dealt with. One of the examples is shown in Fig 3. To deal with such false positives we used the width of the bounding rectangle as a test since the width of a closed fist was much smaller that of an open palm.

### Gesture Recognition:

We decided to use waving as a gesture. To determine if a person is waving, we used a difference image of a skin detected frame. We then found both vertical and horizontal projections and calculated the total number of motion pixels. When waving vigorously, the program would detect that a certain threshold is reached and output a result. While testing, a person’s hand needs to be far enough from the camera so that it could fit the screen. The threshold could be increased to prompt the user for faster waving. 

### Counting Fingers:

We decided to include a method to count fingers since it provided a good idea of separating different types of hand gestures and also because it is one the most popular ways of recognizing hands.

We used the obtained contour to extract a convex hull, which is a convex set enclosing the contour, and the convexity defect points, which are points of defects in the convex hull.

The algorithm then looked for the defects with a specified depth: < 80 and 15 > in our case (We got these values after performing some tests). The resulting number of defect points told us how many fingers were in the frame. There were of course some errors with some hand shapes where the count would be wrong.

## Experiments

For testing the hand gesture recognition and finger counting we looked at how our system reacted to random hand gestures captured by a camera. We ran a number of experiments to determine the true positive and false positive rates. The true positive rates turned out to be perfect in our case. The false positive rates, however, were not perfect, since some gestures looked similar to the ones we were looking for. The method could be improved to exclude those cases by multiple conditions for one gesture.

### Confusion Matrix for Thumbs Up

|| p | n |
| ------------- |:-------------:| -----:|
| Y      | 10 | 3 |
| N      | 0      |   7 |
