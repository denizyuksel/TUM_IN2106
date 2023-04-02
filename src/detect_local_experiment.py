import cv2 as cv
import numpy as np
import math

show_original = True
show_final = True
show_intermediate = True

## Give the file name for experimenting
filename = "sample.jpg"

################################################################################
# Load and scale the input image
################################################################################

img = cv.imread(filename, cv.IMREAD_COLOR)

scale_factor = 800 / img.shape[1]

w = int(scale_factor * img.shape[1])
h = int(scale_factor * img.shape[0])

scaled_dim = (w, h)

img = cv.resize(img, scaled_dim, interpolation=cv.INTER_LANCZOS4)

if (show_original):
    cv.imshow("Image", img)
    cv.waitKey(0)

################################################################################
# Color threshold and dilate & erode to get outlines of blue lines
################################################################################

# Detect blue color in HSV color space
hsv = cv.cvtColor(img.copy(), cv.COLOR_BGR2HSV)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
# Which pixels are blue, which are not
mask = cv.inRange(hsv, lower_blue, upper_blue)

kernel = np.ones((5, 5), np.uint8)
mask = cv.dilate(mask, kernel, iterations=5)
mask = cv.erode(mask, kernel, iterations=5)

if (show_intermediate):
    cv.imshow("mask", mask)
    cv.waitKey(0)

################################################################################
# Use the Hough transform to detect lines in the blue line mask
################################################################################

hough = img.copy()
lines = cv.HoughLines(mask, 5, 5 * np.pi / 180, 800, None, 0, 0)

# filter out lines
vertical_lines = []
horizontal_lines = []

# How much can horizontal and vertical lines can bend?
ep = np.pi / 180 * 5

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        if (theta > ep and theta < np.pi / 2 - ep and theta > np.pi / 2 + ep):
            continue
        if (theta < ep):
            horizontal_lines.append((rho, theta))
        if (theta < np.pi / 2 + ep and theta > np.pi / 2 - ep):
            vertical_lines.append((rho, theta))

print("horizontal lines not sorted: ", horizontal_lines)
print("vertical lines not sorted: ", vertical_lines)

horizontal_lines.sort(key=lambda x: x[0])
vertical_lines.sort(key=lambda x: x[0])

print("horizontal lines sorted: ", horizontal_lines)
print("vertical lines sorted: ", vertical_lines)

################################################################################
# Detect the red dot
################################################################################

lower_red_1 = np.array([0, 70, 50])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 70, 50])
upper_red_2 = np.array([180, 255, 255])
dot_mask_1 = cv.inRange(hsv, lower_red_1, upper_red_1)
dot_mask_2 = cv.inRange(hsv, lower_red_2, upper_red_2)
dot_mask = dot_mask_1 | dot_mask_2

red_pixels = np.where(dot_mask > 0)
red_center = np.average(red_pixels, axis=1).astype(int).tolist()
red_center.reverse()


################################################################################
# Find pairwise intersections of horizontal and vertical lines
################################################################################

def test_line(line, p, ep=np.pi / 180 * 5):
    r = line[0]
    t = line[1]
    print("line: ", line)
    print("r: ", line[0])
    print("t: ", line[1])
    print("point: ", p)
    print("x: ", p[0])
    print("y: ", p[1])

    if (t < ep):  # line is horizontal
        print("line is horizontal")
        det = p[0] - r

    elif (t > np.pi / 2 - ep and t < np.pi / 2 + ep):  # line is vertical
        print("line is vertical")
        det = p[1] - r

    print("determinant: ", det)
    print("-------")

    return det < 0


hor_dir = []
ver_dir = []

print("horizontal lines: ", horizontal_lines)
print("vertical lines: ", vertical_lines)

for horzline in horizontal_lines:
    hor_dir.append(test_line(horzline, red_center))
for vertline in vertical_lines:
    ver_dir.append(test_line(vertline, red_center))

hindex = 0
vindex = 0

print("hordir: ", hor_dir)
print("verdir: ", ver_dir)

for hdir in hor_dir:
    if hdir == False:
        hindex += 1

for vdir in ver_dir:
    if vdir == False:
        vindex += 1

grid_point = np.zeros(shape=(len(ver_dir) + 1, len(hor_dir) + 1)).astype(int)
grid_point[vindex, hindex] = 1
print(grid_point)
print(type(grid_point))

################################################################################
# Draw lines for visualization if show intermediate is set
################################################################################

if (show_intermediate or show_final):
    for i in range(0, len(horizontal_lines)):
        rho = horizontal_lines[i][0]
        theta = horizontal_lines[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
        pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
        cv.line(hough, pt1, pt2, (0, 255, 0), 3, cv.LINE_AA)

    for i in range(0, len(vertical_lines)):
        rho = vertical_lines[i][0]
        theta = vertical_lines[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
        pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
        cv.line(hough, pt1, pt2, (0, 255, 0), 3, cv.LINE_AA)

    cv.circle(hough, red_center, 5, (0, 0, 255), -1)

    cv.imshow("hough_lines", hough)
    cv.waitKey(0)
