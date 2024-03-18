import cv2
KNOWN_DISTANCE = 15.8

KNOWN_WIDTH = 4.0


GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(0)

traffic_light_sign_detector = cv2.CascadeClassifier('haar_xml_07_19.xml')

def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

def distance_finder(focal_length, real_traff_light_sign_width, traff_light_sign_width_in_frame):
    distance = (real_traff_light_sign_width * focal_length) / traff_light_sign_width_in_frame
    return distance

def traffic_light_sign_data(image):
    traffic_light_sign_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    traffic_light_Sign = traffic_light_sign_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in traffic_light_Sign:
        cv2.rectangle(image, (x, y), (x + w-30, y + h+30), RED, 3)
        traffic_light_sign_width = w
        print(w)

    return traffic_light_sign_width

# reading reference image from directory
ref_image = cv2.imread("test_files/Untitled.png")
ref_image_traffic_light_sign_width = traffic_light_sign_data(ref_image)

focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_traffic_light_sign_width)
print(focal_length_found)
cv2.imshow("ref_image", ref_image)

while True:
    _, frame = cap.read()

    # calling face_data function
    traff_light_sign_width_in_frame = traffic_light_sign_data(frame)
    # finding the distance by calling function Distance
    if traff_light_sign_width_in_frame != 0:
        Distance = distance_finder(focal_length_found, KNOWN_WIDTH, traff_light_sign_width_in_frame)
        # Drwaing Text on the screen
        cv2.putText(
            frame, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, (RED), 2
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.imwrite("ref_image.jpg", frame)
        break
cap.release()
cv2.destroyAllWindows()
