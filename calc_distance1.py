import cv2

KNOWN_DISTANCE = 15.8 # centimeter
# width of face in the real world or Object Plane
KNOWN_WIDTH = 7.1 # centimeter

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(0)

# face detector object
stop_detector = cv2.CascadeClassifier("cascade_stop_sign.xml")


# focal length finder function
def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


# distance estimation function
def distance_finder(focal_length, real_stop_width, stop_width_in_frame):
    distance = (real_stop_width * focal_length) / stop_width_in_frame
    return distance


# face detector function
def stop_data(image):
    stop_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stop = stop_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in stop:
        cv2.rectangle(image, (x, y), (x + w, y + h), RED, 3)
        stop_width = w

    return stop_width


# reading reference image from directory
ref_image = cv2.imread("resources/stop_ref.jpg")
ref_image_stop_width = stop_data(ref_image)

focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_stop_width)
print(focal_length_found)
cv2.imshow("stop_ref", ref_image)


while True:
    _, frame = cap.read()
    # calling face_data function
    stop_width_in_frame = stop_data(frame)
    # finding the distance by calling function Distance
    if stop_width_in_frame != 0:
        Distance = distance_finder(focal_length_found, KNOWN_WIDTH, stop_width_in_frame)
        # Drwaing Text on the screen
        cv2.putText(
            frame, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, (RED), 2
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.imwrite("ref_image.jpg", frame)
        # break
# cap.release()
cv2.destroyAllWindows()
