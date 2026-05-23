# tests/test_vision.py
#
# Standalone vision validator — runs the ColorDetector on the laptop webcam
# and overlays everything you need to judge red/green detection quality:
#   - bounding box + centroid for each color
#   - vertical "frame center" line (so you can SEE the pixel error)
#   - per-color error text
#   - FPS counter
#
# Keys:
#   q  quit
#   m  toggle the red & green binary mask windows (HSV debugging)
#   r  cycle which colors are active: both -> red only -> green only -> both

import time
import cv2
import numpy as np

import config
from vision.detector import ColorDetector

# --- Setup ---
detector = ColorDetector(config)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Could not open webcam (index 0).")

show_masks = False
color_mode = "both"   # "both" | "red" | "green"

# BGR drawing colors (NOT the HSV detection ranges)
BOX_BGR = {"red": (0, 0, 255), "green": (0, 255, 0)}
DOT_BGR = (255, 255, 0)   # cyan-ish, visible on both red and green boxes
TEXT_BGR = (255, 255, 255)

print("Vision validator running. Keys: q=quit  m=masks  r=cycle colors")

# --- Helpers ---
def _color_mask(detector, frame, color):
    """Re-derive the cleaned binary mask the detector uses (for visualization)."""
    height, width, _ = frame.shape
    roi_start_y = int(height * 0.4)
    roi = frame[roi_start_y:height, 0:width]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if color == "red":
        mask = cv2.inRange(hsv, detector.red_low, detector.red_high)
    else:
        mask = cv2.inRange(hsv, detector.green_low, detector.green_high)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # Pad mask back to full frame height so it's easy to compare visually
    full = np.zeros((height, width), dtype=np.uint8)
    full[roi_start_y:height, 0:width] = mask
    return full


def _draw_detection(frame, color_name, center_x, bbox):
    """Draw bbox, centroid, and error text for one detected color."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_BGR[color_name], 2)
    cv2.circle(frame, (center_x, y + h // 2), 5, DOT_BGR, -1)
    error = center_x - detector.frame_center_x
    label_y = 30 if color_name == "red" else 60
    cv2.putText(frame, f"{color_name}: cX={center_x}  err={error:+d}px",
                (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                BOX_BGR[color_name], 2)


# --- Main loop ---
fps_t0 = time.monotonic()
fps_frames = 0
fps_value = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, config.CAMERA_RESOLUTION)

    # Vertical reference line at the frame's horizontal center.
    # PID error is measured against this line.
    cv2.line(frame, (detector.frame_center_x, 0),
             (detector.frame_center_x, frame.shape[0]),
             (200, 200, 200), 1)

    # Detect each active color
    active = ("red", "green") if color_mode == "both" else (color_mode,)
    for c in active:
        center_x, bbox = detector.get_object_center(frame, color=c)
        if center_x is not None:
            _draw_detection(frame, c, center_x, bbox)

    # FPS — average over ~0.5s windows so it doesn't jitter
    fps_frames += 1
    elapsed = time.monotonic() - fps_t0
    if elapsed >= 0.5:
        fps_value = fps_frames / elapsed
        fps_t0 = time.monotonic()
        fps_frames = 0
    cv2.putText(frame, f"{fps_value:4.1f} FPS  mode={color_mode}",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                TEXT_BGR, 2)

    cv2.imshow("WRO Vision Validator", frame)

    if show_masks:
        cv2.imshow("Mask: red",   _color_mask(detector, frame, "red"))
        cv2.imshow("Mask: green", _color_mask(detector, frame, "green"))
    else:
        # Close mask windows when toggled off
        for win in ("Mask: red", "Mask: green"):
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(win)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        show_masks = not show_masks
    elif key == ord('r'):
        color_mode = {"both": "red", "red": "green", "green": "both"}[color_mode]

cap.release()
cv2.destroyAllWindows()
