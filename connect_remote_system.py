import time

import cv2
import numpy as np
from time import sleep
import requests
from io import BytesIO

robot_address = 'http://192.168.4.1'
stream_url = robot_address + ':81/stream'
wheel_speed_update_url = robot_address + ':82/update'
system_initialization_url = robot_address + ':82/system_setup'


def get_image_stream():
    # print(stream_url)
    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            img_data = BytesIO(chunk)
            cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
            rotated_image = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            cv_resized_img = cv2.resize(rotated_image, (480, 240), interpolation=cv2.INTER_AREA)

            return cv_resized_img


def move(curve_val, max_average_curve=0.5, sensitivity=50, wait=1, relax=False):
    traff_flag = False
    sensitivity += 100  # How sensitive is the system to change in curve values
    # max_average_curve = 0.5
    mean_duty_cycle = 60

    if relax:
        sleep(0.2)
        print("Relaxing")
    #     mean_duty_cycle = 65
    # else:
    #     # mean_duty_cycle = 60


    # turn = (curve_val  / max_average_curve) * (max_duty_cycle - mean_duty_cycle)
    if curve_val < 0:
        max_duty_cycle = 75
        turn = (curve_val * sensitivity) * 60
    else:
        max_duty_cycle = 80
        turn = (curve_val * sensitivity) * 40
    turn = turn / 100
    left_duty = mean_duty_cycle - round(turn)
    right_duty = mean_duty_cycle + round(turn)

    if wait == 0:
        # start_time = 0
        # end_time = 0
        # if traff_flag is False:
        #     start_time = time.time()
        #     traff_flag = True
        # else:
        #     end_time = time.time()
        # duration = start_time - end_time
        # if duration > 5:
        #     print("Duration is greater than 5")
        #     traff_flag = False
        #
        print("Traff_flag is: " + str(traff_flag))
    elif wait == 1:
        if left_duty < 35:
            left_duty = 0
        elif left_duty > max_duty_cycle:
            left_duty = max_duty_cycle
        if right_duty < 35:
            right_duty = 0
        elif right_duty > max_duty_cycle:
            right_duty = max_duty_cycle

    else:
        left_duty = 0
        right_duty = 0

    right_wheel_duty_cycle = 'R' + str(right_duty)
    left_wheel_duty_cycle = 'L' + str(left_duty)
    print("turn is: " + str(turn))
    print("curve is :" + str(curve_val))
    print(left_wheel_duty_cycle)
    print(right_wheel_duty_cycle)

    wheel_speed_update_url_2 = wheel_speed_update_url + '?move=' + str(wait) + '&leftWheelDutyCycle=' + \
                               str(left_wheel_duty_cycle) + '&rightWheelDutyCycle=' + str(right_wheel_duty_cycle)

    try:
        wheel_speed_update_response = requests.get(wheel_speed_update_url_2)
        if wheel_speed_update_response.text == "OK":
            return 1
    except:
        return 0
    else:
        return -1


def initialise_remote_wheel_control_system(initialise=False, flash_on=False):
    # Wait and make sure the remote system is properly initialised
    response = None
    count = 0

    while True:
        if count == 6:
            print("Failed to initialise remote wheel control system")
            print("Max number of retry attempts exceeded. Shutting down system.")
            return False
        if initialise:
            response = requests.get(system_initialization_url + '?initialise=true')
            print(system_initialization_url)
            if response.text == "System Initialized":
                print("Remote system initialised successfully")
                break
        elif flash_on:
            response = requests.get(system_initialization_url + '?flash_on=true')
            if response.text == "flash ok":
                print("Flash turned on")
                break
        if response is not None:
            print(response.text)
        count = count + 1
        sleep(1)

    return True
