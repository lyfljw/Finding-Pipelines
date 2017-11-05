import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_points(slope, bias, y):
    return (int((y - bias) / slope), y)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    biases_left = 0
    slopes_left = 0
    total_weight_left = 0.001
    biases_right = 0
    slopes_right = 0
    total_weight_right = 0.001
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope, bias = np.polyfit([x1, x2],[y1,y2],1)
            if abs(slope) < .5:
                continue
            if abs(slope) > 0.8:
                continue
            length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            weight = length * 0.3
            if slope > 0:
                slopes_right += slope*weight
                total_weight_right += weight
                biases_right += bias*weight
            elif slope < 0:
                slopes_left += slope * weight
                total_weight_left += weight
                biases_left += bias*weight

    # get the slopes and bias
    left_slope = slopes_left/total_weight_left
    left_bias = biases_left/total_weight_left
    right_slope = slopes_right/total_weight_right
    right_bias = biases_right/total_weight_right
    # # y = mx+b
    y_size = img.shape[0]
    x_size = img.shape[1]
    # # get the top and bottom points

    # predict one line if one of them are undetectable in one image
    if left_slope == 0:
        left_slope = -right_slope
        left_bias = right_bias + right_slope * x_size + 10
    if right_slope == 0:
        right_slope = -left_slope
        right_bias = left_bias + left_slope * x_size - 10

    left_bottom_coor = get_points(left_slope, left_bias, y_size)
    left_top_coor = get_points(left_slope, left_bias, int(y_size * .6))
    right_bottom_coor = get_points(right_slope, right_bias, y_size)
    right_top_coor = get_points(right_slope, right_bias, int(y_size * .6))



    # draw the lines
    cv2.line(img, left_bottom_coor, left_top_coor, color, 15)
    cv2.line(img, right_bottom_coor, right_top_coor, color, 15)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # size of the image
    imshape = image.shape
    # kernal size for GaussianBlur
    kernel_size = 3  # only odd numbers
    # Define the parameters for Canny
    low_threshold = 50
    high_threshold = 150
    # parameters for Hough Trans
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 16  # minimum number of pixels making up a line
    max_line_gap = 25  # maximum gap in pixels between connectable line segments
    line_image_canvas = np.copy(image) * 0  # creating a blank to draw lines on

    # Note: always make a copy rather than simply using "="
    initial_img_copy = np.copy(image)
    current_image = np.copy(image)
    # first convert the image to gray
    current_image = grayscale(current_image)
    # Apply GaussianBlur
    current_image = gaussian_blur(current_image, kernel_size)
    # Apply Canny
    current_image = canny(current_image, low_threshold, high_threshold)
    # define the polygon

    vertices = np.array([[(0, imshape[0]), (imshape[1]*0.5, imshape[0]*0.6),
                         (imshape[1]*0.5, imshape[0]*0.6),(imshape[1], imshape[0])]],
                            dtype=np.int32)

    # create a masked image, all are out of the polygon should be empty
    masked_img = region_of_interest(current_image, vertices)
    # now Hough transformation to get the line image
    line_image_canvas = hough_lines(masked_img, rho, theta, threshold,
                                    min_line_length, max_line_gap)
    # draw_lines(line_image_canvas,line_image,thickness=5)
    result = weighted_img(line_image_canvas, initial_img_copy)

    return result



white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
