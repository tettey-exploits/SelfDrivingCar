        ######stop sign detect
# def detectStopSign(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)
#
#     #   Detect the stop sign, x,y = origin points,w =width, h=height
#     for (x, y, w, h) in stop_sign_scaled:
#         #         Draw rectangle around the stop sign
#         stop_sign_rectangle = cv2.rectangle(img, (x, y), (x + y, y + h), (0, 255, 0), 4)
#
#         #           Write "Stop sign" on the bottom of the rectangle
#         cv2.putText(img=stop_sign_rectangle, text="Stop Sign", org=(x, y + h + 30),
#                                      fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 0, 255),
#                                      thickness=2, lineType=cv2.LINE_4)
#     calc_distance.calc_distance_func(img)

# cap = cv2.VideoCapture(0)

# # traffic signs detector object
# stop_detector = cv2.CascadeClassifier("cascade_stop_sign.xml")
# # turn_left_detector = cv2.CascadeClassifier("cascade_stop_sign.xml")
#
#
# # focal length finder function
# def focal_length(measured_distance, real_width, width_in_rf_image):
#     focal_length_value = (width_in_rf_image * measured_distance) / real_width
#     return focal_length_value
#
#
# # distance estimation function
# def distance_finder(focal_length, real_stop_width, stop_width_in_frame):
#     distance = (real_stop_width * focal_length) / stop_width_in_frame
#     return distance
#
#
# # face detector function
# def stop_data(image):
#     stop_width = 0
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     stop = stop_detector.detectMultiScale(gray_image, 1.3, 5)
#     for (x, y, h, w) in stop:
#         cv2.rectangle(image, (x, y), (x + w, y + h), RED, 3)
#         stop_width = w
#     return stop_width
#
#
# # reading reference image from directory
# ref_image = cv2.imread("stop_ref.jpg")
# ref_image_stop_width = stop_data(ref_image)
#
# focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_stop_width)
# # print(focal_length_found)
# cv2.imshow("stop_ref", ref_image)

# # Real world dimensions of Stop Sign
# KNOWN_DISTANCE = 15.8  # centimeter
# # width of stop in the real world or Object Plane
# KNOWN_WIDTH = 7.1  # centimeter


# #Check for last frame and start video again
# frameCounter +=1
# if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     frameCounter = 0