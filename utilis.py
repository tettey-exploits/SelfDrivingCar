import cv2
import numpy as np

import connect_remote_system


def slope(xa, ya,  xb, yb):
    return (yb - ya) / (xb - xa)


def thresholding(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite, upperWhite = color_trackbar(img)
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    return maskWhite


def warp_img(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv == False:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    else:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    img_warp = cv2.warpPerspective(img, matrix, (w, h))
    return img_warp


def draw_points(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
        # cv2.circle()
    return img


def nothing(a):
    pass

#####Color Trackbar
cv2.namedWindow("CTrackbars")
cv2.resizeWindow("CTrackbars", 640, 240)
cv2.createTrackbar("Hue Min", "CTrackbars", 0, 255, nothing)
cv2.createTrackbar("Hue Max", "CTrackbars", 177, 255, nothing)
cv2.createTrackbar("Sat Min", "CTrackbars", 0, 255, nothing)
cv2.createTrackbar("Sat Max", "CTrackbars", 52, 255, nothing)
cv2.createTrackbar("Val Min", "CTrackbars", 80, 255, nothing)
cv2.createTrackbar("Val Max", "CTrackbars", 255, 255, nothing)

# ######Movement Direction
# cv2.namedWindow("MTrackbars")
# cv2.resizeWindow("MTrackbars", 640, 80)
# cv2.createTrackbar("FOB", "MTrackbars", 1, 2, nothing)
# cv2.createTrackbar("LOR", "MTrackbars", 1, 2, nothing)


def detect_red_color(img):
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(img, img, mask=red_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            image_frame = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(image_frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    cv2.imshow("Traffic Red", img)


def color_trackbar(img):
    h_min = cv2.getTrackbarPos("Hue Min", "CTrackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "CTrackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "CTrackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "CTrackbars")
    v_min = cv2.getTrackbarPos("Val Min", "CTrackbars")
    v_max = cv2.getTrackbarPos("Val Max", "CTrackbars")
    # print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    return lower, upper
    # mask = cv2.inRange(imgHSV, lower, upper)
    # img_result = cv2.bitwise_and(img, img, mask=mask)
    # imgStack = np.hstack([img_result, img])
    # imgStack1 = np.hstack([img_result, imgHSV])
    # imgStack2 = np.vstack([imgStack1, imgStack])
    # cv2.imshow('ImageResult', imgStack2)
    # cv2.imshow('Image', mask)


def initialize_trackbars(initial_trackbar_vals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initial_trackbar_vals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initial_trackbar_vals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initial_trackbar_vals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initial_trackbar_vals[3], hT, nothing)

###Create trackbar and use to camera servo motor
# cv2.namedWindow("Camera Servo Motor")
# cv2.resizeWindow("Camera Servo Motor", 480, 40)
# cv2.createTrackbar("Turn Degrees", "Camera Servo Motor", 90, 180, nothing)
#
# ##Get Camera Servo Track Position
# def servoTrackbar():
#     return cv2.getTrackbarPos("Turn Degrees", "Camera Servo Motor")


def val_trackbars(wT=480, hT=240):
    width_top = cv2.getTrackbarPos("Width Top", "Trackbars")
    height_top = cv2.getTrackbarPos("Height Top", "Trackbars")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(width_top, height_top), (wT-width_top, height_top),
                         (width_bottom, height_bottom), (wT-width_bottom, height_bottom)])
    return points


def get_histogram(img, min_percent=0.5, region=1):
    if region == 1:
        hist_values = np.sum(img, axis=0)
    else:
        hist_values = np.sum(img[img.shape[0]//region:,:], axis=0)   #Calculate the sum of pixel values on region x of axis 0 or colums

    max_value = np.max(hist_values) #Get the maximum value
    min_value = min_percent * max_value #What percentage of maximum value to consider as minimum when finding path.

    index_array = np.where(hist_values >= min_value) #Get the index of hist_values that are greater than the threshold
    # print('index array is')
    # print(index_array)
    base_point = int(np.average(index_array))

    #####After getting the vertical sum of pixel values and their respective indices, the Base point refers to the  average index
    #####of these indices, giving us the average value of how much more turn/curve there is to the side.

    img_hist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for x, intensity in enumerate(hist_values):
        cv2.circle(img_hist, (base_point, img.shape[0]), 20, (255, 0, 255), cv2.FILLED)
        cv2.line(img_hist, (x, img.shape[0]), (x, img.shape[0]-intensity//255//region), (0, 0, 255), 1)
    return base_point, img_hist


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        cv2.drawContours(img, cnt, -1, (255, 0, 0),
                         3)  # draw a shape with dimensions of recognised shape on another image
        area = cv2.contourArea(cnt)  # find area of contour
        peri = cv2.arcLength(cnt, True)  # find perimeter
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # approximate the corner points
        # print(area)
        # print(peri)
        # print(approx)
        # print(len(approx))
        objCor = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        #
        # if objCor == 3: objectType = "Triangle"
        # elif objCor == 4: objectType = "Square"
        # elif objCor == 6: objectType = "Hexagon"
        # elif objCor == 8: objectType = "Circular"
        # else:objectType = "Not identified"

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(img, objectType,
        #             (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

        return img


def stack_images(scale, imgArray, lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

