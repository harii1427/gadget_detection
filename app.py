import cv2
import numpy as np

def is_fire_present(frame):
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Bitwise-AND mask and original image
    fire_presence = cv2.bitwise_and(frame, frame, mask=mask)

    # Check if there are any red pixels
    if np.sum(mask) > 0:
        return True, mask
    return False, mask

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fire_detected, mask = is_fire_present(frame)
    if fire_detected:
        cv2.putText(frame, 'Fire Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Fire Detection", frame)
    cv2.imshow("Mask", mask)  # Show mask window

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
