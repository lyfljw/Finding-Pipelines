## **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_image_output/solidWhiteRight.jpg 

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.


My pipeline consisted of 8 steps. All steps are clearly commented in the code. Here is a brief introduction of the main idea.
1. Define some important parameters for the functions and adjust them later.
2. Convert the image to gray image.
3. Apply GaussianBlur on the gray image to reduce some noise
4. Apply Canny to the image 
5. Define the polygon. I used 
`[(0, imshape[0]), (imshape[1]*0.5, imshape[0]*0.6),(imshape[1]*0.5, imshape[0]*0.6),(imshape[1], imshape[0])]` 
as my vertices instead of numbers to fit all resolution of pictures.
6. Get a masked image with the polygon area defined in step 4.
7. Apply Hough transformation on that masked image to find the lines, and draw the lines on an empty canvas.
8. Merge the canvas with lines and the original image to get a "pipeline highlighted" image.

In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function.  #### Algorithm:
	1. Add up all the reasonable slopes and biases of each pipeline in the picture with weights.
	2. regularzation the slopes and biases to get the weight mean. This step will help reduce the influence of some short line segments.
	3. Using the slope and bias terms found in 2nd step, calculate the bottom and top points' coordinates of both the left and right lines.
	4. cv2.line() to draw the line.

In order to implement the algorithm, I did
##### For step 1:
	1. Set up some useful variables to save the sums of the slopes, biases and total weight number.
	2. Calculate the length of each segments.
	3. 



If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
