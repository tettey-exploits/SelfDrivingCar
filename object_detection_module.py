import time
import cv2
#
import utilis


PATH = "resources/"
KNOWN_DISTANCE = 15.8 # centimeter
# width of face in the real world or Object Plane
KNOWN_STOP_WIDTH = 7.1 # centimeter
#
# # Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX


# # stop detector object
stop_detector = cv2.CascadeClassifier(PATH + "cascade_stop_sign.xml")
traffic_light_detector = cv2.CascadeClassifier(PATH + "haar_xml_07_19.xml")

# # reading reference image from directory
ref_image = cv2.imread(PATH + "stop_ref.jpg")


classFile = PATH + 'coco.names'
faceCascade = cv2.CascadeClassifier(PATH + "haarcascade_russian_plate_number.xml")

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = PATH + 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = PATH + 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

thres = 0.5 #Threshold to detect object


# # focal length finder function
def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


def distance_finder(focal_length_param, real_stop_width, stop_width_in_frame):
    distance = (real_stop_width * focal_length_param) / stop_width_in_frame
    return distance


def stop_data(image):
    stop_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stop = stop_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in stop:
        cv2.rectangle(image, (x, y), (x + w, y + h), RED, 1)
        stop_width = w

    return stop_width


def traffic_data(image):
    traffic_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    traffic_light = traffic_light_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in traffic_light:
        cropped_image = image[y:y+h+30, x:x + w+30]
        # utilis.detect_red_color(cropped_image)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
        traffic_width = w

    # cv2.imshow("traffic", image)
    return traffic_width


ref_image_stop_width = stop_data(ref_image)

focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_STOP_WIDTH, ref_image_stop_width)


def calc_traffic_sign_distance(img):

    image = img
    movement_status = 1
    distance = 0

    stop_width_in_frame = stop_data(image)
    traffic_width_in_frame = traffic_data(image)

    if stop_width_in_frame != 0:
        distance = distance_finder(focal_length_found, KNOWN_STOP_WIDTH, stop_width_in_frame)
        # Drwaing Text on the screen
        cv2.putText(
            img, f"Distance = {round(distance,2)} CM", (50, 50), fonts, 1, RED, 1
        )
        if distance <= 40:
            print("Robot within traffic sign zone... Stopped")
            movement_status = 0
        else:
            print("Stop sign detected. Still approaching")
            movement_status = 1

    return movement_status


def detect_objects(img):
    movement_status = calc_traffic_sign_distance(img)
    class_ids, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(class_ids) != 0:  # or classId < (len(class_ids)-1):

        for classId, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            confidence = confidence * 100
            if classId == 3 or classId == 1 or classId == 10 or classId == 13:
                if confidence > 60:
                    if classId == 1:
                        movement_status = 2
                    else:
                        movement_status = 0
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)

                    cv2.putText(img, classNames[classId-1], (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    # cv2.putText(img, str(round(confidence, 2)), (box[2] - 40, box[1] + 30),
                    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    # cv2.putText(img, str(count), (40, 70),
                    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return movement_status
