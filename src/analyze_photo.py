import numpy
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
import numpy as np
import cv2 as cv
import math
import json

app = FastAPI()

def scale_image(img):
    scale_factor = 800 / img.shape[1]

    w = int(scale_factor * img.shape[1])
    h = int(scale_factor * img.shape[0])

    scaled_dim = (w, h)
    image = cv.resize(img, scaled_dim, interpolation=cv.INTER_LANCZOS4)

    return image


def create_blue_mask(img):
    # Detect blue color in HSV color space
    hsv = cv.cvtColor(img.copy(), cv.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # Which pixels are blue, which are not
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=5)
    mask = cv.erode(mask, kernel, iterations=5)

    return mask


def detect_red_center(img):
    hsv = cv.cvtColor(img.copy(), cv.COLOR_BGR2HSV)
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

    return red_center

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

    else:
        print("INSIDE ELSE")
        p1 = (0, r / math.sin(t))
        p2 = (r / math.cos(t), 0)
        det = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    print("determinant: ", det)
    print("-------")

    return det < 0

class Url(BaseModel):
    url: str

@app.post("/")
def download_and_analyze_image(image_link: Url):
    with requests.get(image_link.url) as rq:
        with open('test_image.png', 'wb') as file:
            file.write(rq.content)
            print(rq.headers)

    img = cv.imread('test_image.png', cv.IMREAD_COLOR)

    img = scale_image(img)
    blue_mask = create_blue_mask(img)

    ## Detect lines
    lines = cv.HoughLines(blue_mask, 5, 5 * np.pi / 180, 800, None, 0, 0)

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

    red_center = detect_red_center(img)

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
    as_list = numpy.ndarray.tolist(grid_point)

    return json.dumps(as_list)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
