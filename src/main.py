import cv2
import numpy as np
from picamera2 import Picamera2
import config
from vision.detector import ColorDetector

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888", "size": config.CAMERA_RESOLUTION}))
picam2.start()

detector = ColorDetector(config)

while True:
    frame = picam2.capture_array()

    red_cx, red_box = detector.get_object_center(frame, "red")
    green_cx, green_box = detector.get_object_center(frame, "green")

    if red_box:
        x, y, w, h = red_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(frame, (red_cx, y + h // 2), 5, (0, 0, 255), -1)
        print(f"Red pillar at x={red_cx}")

    if green_box:
        x, y, w, h = green_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (green_cx, y + h // 2), 5, (0, 255, 0), -1)
        print(f"Green pillar at x={green_cx}")

    cv2.imwrite("latest_frame.jpg", frame)