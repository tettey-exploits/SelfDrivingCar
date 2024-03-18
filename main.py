import sys
import math
import time
from time import sleep

import cv2
import numpy as np

import utilis
import connect_remote_system
import object_detection_module

cap = cv2.VideoCapture(0)#'20220314_065650.mp4')
curveList = []
avgVal = 2
max_average_curve = 0.5
img_width = 480
img_height = 240
# manual_curve_val = [-0.5, -0.4]

# System flags
engage_remote_wheel_control_system = False
remote_sys_init = False
traff_flag = 0

start_time = 0
end_time = 0


# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX


def get_lane_curve(img, img_result):
    img_copy = img.copy()

    ####### Step 1. Get Lane Color
    img_thres = utilis.thresholding(img)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0.4)
    img_canny = cv2.Canny(img_blur, 127, 255)
    # img_lane_detection = utilis.getContours(img_canny)
    # # if engage_remote_wheel_control_system:
    # cv2.imshow('Lane', img_lane_detection)

    # lines = cv2.HoughLinesP(img_canny, 6, np.pi / 60, 160, np.array([]), minLineLength=10, maxLineGap=250)
    # Draw lines on the image

    # if lines is not None:
    #     count_a = 0
    #     for line_a in lines:
    #         count_b = 0
    #         for line_b in lines:
    #             count_b = count_b + 1
    #             if count_a == count_b - 1:
    #                 continue
    #             xa_1, ya_1, xa_2, ya_2 = line_a[0]
    #             xb_1, yb_1, xb_2, yb_2 = line_b[0]
    #
    #             ## Calculate slope
    #             slope_a = utilis.slope(xa_1, ya_1, xa_2, ya_2)
    #             slope_b = utilis.slope(xb_1, yb_1, xb_2, yb_2)
    #             if abs(slope_a - slope_b) < 0.2:
    #
    #                 line_dist = math.sqrt((xa_1-xb_2)**2 + (ya_1-yb_2)**2)
    #                 print(line_dist)
    #                 cv2.line(img, (xa_1, ya_1), (xa_2, ya_2), (255, 0, 0), 3)
    #                 cv2.line(img, (xb_1, yb_1), (xb_2, yb_2), (0, 0, 255), 3)
    #
    #         count_a = count_a + 1

    ####### Step 2
    points = utilis.val_trackbars()
    img_warp = utilis.warp_img(img_thres, points, img_width, img_height)
    img_drw_warp_pnts = utilis.draw_points(img_copy, points)

    ####### Step 3
    middle_point, histo_diagram = utilis.get_histogram(img_warp, min_percent=0.5, region=4)
    curve_average_point, histo_diagram = utilis.get_histogram(img_warp, min_percent=0.9)
    curve_raw = middle_point - curve_average_point

    ####### Step 4
    curveList.append(curve_raw)

    if len(curveList) > avgVal:
        curveList.pop(0)
    curve_val = (int(sum(curveList)/len(curveList)))/100

    if 0.05 >= curve_val >= -0.02:
        curve_val = 0

    ####### If the curve is changing then relax
    if abs(curve_raw - curve_val) < 0.02:
        is_curving = False
    else:
        is_curving = True

    if curve_val > 1:
        curve_val = 1
    elif curve_val < -1:
        curve_val = -1

    ##### Step 5
    img_inv_warp = utilis.warp_img(img_warp, points, img_width, img_height, inv=True)
    img_inv_warp = cv2.cvtColor(img_inv_warp, cv2.COLOR_GRAY2BGR)
    img_inv_warp[0:img_height // 3, 0:img_width] = 0, 0, 0
    img_lane_color = np.zeros_like(img)
    img_lane_color[:] = 0, 255, 0
    img_lane_color = cv2.bitwise_and(img_inv_warp, img_lane_color)
    img_result = cv2.addWeighted(img_result, 1, img_lane_color, 1, 0)
    mid_y = 450

    object_detection_module.traffic_data(img)

    cv2.putText(img_result, str(curve_val), (img_width // 2 - 80, 85),
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 0, 255),
                thickness=2, lineType=cv2.LINE_4)

    # cv2.line(img_result, ((wT // 2 + (curve_val * 3)), mid_y - 2), (wT // 2 + (curve_val * 3)), (0, 0, 255), 1)
    for x in range(-30, 30):
        w = img_width // 20
        cv2.line(img_result, (w * x + int(curve_val // 50), mid_y - 10),
                 (w*x + int(curve_val//50), mid_y + 10), (0, 0, 255), 2)

    img_stacked = utilis.stack_images(0.7, ([img, img_drw_warp_pnts, img_warp], [histo_diagram, img_lane_color,
                                                                                 img_result]))

    # # cv2.imshow('Warp', imgWarp1)
    # cv2.imshow('Histogram', histo_diagram)
    # cv2.imshow('Warp Points', img_drw_warp_pnts)
    # cv2.imshow('Results', img_stacked)
    return curve_val, is_curving, img_stacked


if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    initialTrackbarVals = [84, 99, 19, 190]
    utilis.initialize_trackbars(initialTrackbarVals)
    try:
        print("Establishing connection with remote system...")
        connect_remote_system.get_image_stream()  # Test stream
        engage_remote_wheel_control_system = True
        print("Connection established. Press 'i' to initialise the remote system...")
    except:
        print("Could not establish connection with remote system...")
        if input('use your webcam?[y/n]: ') == 'y':
            engage_remote_wheel_control_system = False
        else:
            print("Please check and retry...")
            print("Exiting")
            sys.exit()
    count = 0
    while True:
        if engage_remote_wheel_control_system:
            try:
                imgOriginal = connect_remote_system.get_image_stream()  # Get video stream
            except:
                print("Could not get remote image stream.")
                sys.exit()
        else:
            success, imgOriginal = cap.read()

        img = imgOriginal.copy()
        # stop_width_in_frame = stop_data(img)

        img = cv2.resize(img, (img_width, img_height))
        imgResult = img.copy()

        movement_status = object_detection_module.detect_objects(img)  # Find Traffic Signs
        curve, is_curving, imgStacked = get_lane_curve(img, imgResult)                 # Find lane and determine curve

        # if traff_flag == 0 and movement_status == 0:
        #     print("Count down sequence initiated")
        #     start_time = time.time()
        #     traff_flag = 1
        # elif traff_flag == 1:
        #     end_time = time.time()
        # duration = end_time - start_time
        #
        # if 5 < duration:# < 100:
        #     print("Duration is greater than 5")
        #     traff_flag = 2

        if traff_flag == 0 and movement_status == 0:
            print("Count down sequence initiated")
            start_time = time.time()
            traff_flag = 1
        elif traff_flag == 1:
            end_time = time.time()
        duration = end_time - start_time

        if 5 < duration:# < 100:
            print("Duration is greater than 5")
            traff_flag = 0


        # print("Traff_flag is: " + str(traff_flag))

        # elif curve
        if curve > max_average_curve:
            max_average_curve = curve

        if engage_remote_wheel_control_system and remote_sys_init:
            if traff_flag == 1:  # If traff_flag is 1, => stop sign has been detected and count down has been initiated, so continue waiting
                movement_status = 0
            elif traff_flag == 2:
                movement_status = 1
            # curve = 0
            connect_remote_system.move(curve, max_average_curve, wait=movement_status, sensitivity=40, relax=is_curving)
        interrupt_key = cv2.waitKey(1)
        if interrupt_key == ord("i") or interrupt_key == ord("f"): # Wait for initialisation key
            if engage_remote_wheel_control_system and ~remote_sys_init:
                                    # Check if remote system is already engaged
                print("Initializing Remote Wheel Control System.")
                try:
                    if interrupt_key == ord("i"):
                        if connect_remote_system.initialise_remote_wheel_control_system(initialise=True):
                            remote_sys_init = True
                        # sleep(5)g
                    else:
                        connect_remote_system.initialise_remote_wheel_control_system(flash_on=True)
                except:
                    print("Could not initialise remote system. Please try again")
                    remote_sys_init = False
            else:
                print("The remote system is not pre-connected or already initialized")

        elif interrupt_key == ord("s"):
            while True:
                print("Stopped.")
                connect_remote_system.move(curve_val=0, wait=0)
                interrupt_key = cv2.waitKey(1)
                if interrupt_key == ord("g"):
                    break
        elif interrupt_key == ord("q"):
            print("Quitting")
            cv2.destroyAllWindows()
            break

        cv2.imshow("Results", imgStacked)

