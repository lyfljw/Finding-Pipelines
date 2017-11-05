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

In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function. 
#### Algorithm for draw_lines():
	1. Add up all the reasonable slopes and biases of each segment in the picture with weights.
	2. regularzation the slopes and biases to get the weight mean. This step will help reduce the influence of some short line segments.
	3. Using the slope and bias terms found in 2nd step, calculate the bottom and top points' coordinates of both the left and right lines.
	4. deal with bad image (didn't detect any segments, or only one side lane) and draw the lines.

In order to implement the algorithm, I did
##### For step 1 and 2:
	1. Set up some useful variables to save the sums of the slopes, biases and total weight number.
	2. A slope with its absolute value between 0.5 to 0.8 will consider reasonable.
	3. A negative slope is from left_line and a positive slope is from right_line.
	4. I used np.polyfit(p1,p2,1) to fit each segments to get the slope and bias.
	5. Calculate the length of each segments.
	6. Multiply the length with a scaler to get the weight of that segment.
	7. Add all the weighted slopes and biases up, and divide the whole weights.
	
##### For step 3 and 4:
	1. Defined a useful function: `get_points(slope, bias, y)` to return a tuple as a point.
	2. I set up the initial total weight 0.001 in case miss detect a pipeline.
	3. If one side is detected, I will do a brief estimatation of the other line.(the two lines are likely to be ymmetrical)
	4. After getting the slopes and biases, I used `get_points(slope, bias, y)` to find the 4 points (2 per each side)
	5. Use cv2.line(img, p1, p2, color, thickless) twice to draw the two lines.



![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

I tried a lot different combinations of the variables but the program still had a hard time to detect the pipelines in the Challenging video. I found that when mis-painting the pipeline, the pipelines have the same slopes with the real lanes. It seemed that it mis-calculted the bias terms but I couldn't figure out how to fix it. 



### 3. Suggest possible improvements to your pipeline

1. Finish the challenging video.

2. Make my program run simultaneously instead of processing the a existed video.