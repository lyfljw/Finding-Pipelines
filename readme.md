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


My pipeline consisted of 5 steps. All steps are clearly commented in the code. Here is a brief introduction of the main idea.
0. Define some important parameters for the functions and adjust them later.
1. Convert the image to gray image.
2. Apply GaussianBlur on the gray image to reduce some noise
3. Apply Canny to the image 
4. Define the polygon. I used 
[(0, imshape[0]), (imshape[1]*0.5, imshape[0]*0.6),(imshape[1]*0.5, imshape[0]*0.6),(imshape[1], imshape[0])] 
as my vertices to fit all resolution of pictures.
5. Get a masked image with the polygon area defined in step 4.
6. Apply Hough transformation on that masked image to find the lines, and draw the lines on an empty canvas.
7. Merge the canvas with lines and the original image to get a "pipeline highlighted" image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
