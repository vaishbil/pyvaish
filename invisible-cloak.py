import cv2
import numpy as np
import time

# Capture initial background
cap = cv2.VideoCapture(0)
time.sleep(2)

# Collect background frame (ensure no one is in view during this)
for i in range(30):
    ret, background = cap.read()
    if ret:
        background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range for cloak detection
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to detect the red cloak
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Remove noise from the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))

    # Segment out the cloak region in the current frame and overlay background
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    # Combine the cloak and non-cloak areas
    final_output = cv2.add(cloak_area, non_cloak_area)

    # Display the final output
    cv2.imshow('Invisible Cloak', final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
