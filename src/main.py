import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 2. Display the resulting frame
    cv2.imshow('WRO Robot View', frame)

    # 3. Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()