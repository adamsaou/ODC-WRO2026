# tests/test_vision_pi.py
#
# Pi-side counterpart to tests/test_vision.py.
# Same vision logic and overlays, but:
#   - Frame source is Picamera2 (Pi OS Bookworm camera stack) instead of cv2.VideoCapture
#   - No cv2.imshow (we run headless over SSH and use opencv-python-headless)
#   - Annotated frames are written to tests/captures/latest.jpg every ~100 ms;
#     open that file in VS Code Remote-SSH and it will auto-refresh
#
# Run from the project root on the Pi:
#     python3 -m tests.test_vision_pi
#
# Flags:
#     --color {both,red,green}    which colors to detect (default: both)
#     --save-masks                also write latest_mask_red.jpg / latest_mask_green.jpg
#     --save-fps N                disk write rate in Hz (default: 10)
#     --archive                   also save every saved frame as captures/archive/frame_NNNNN.jpg

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

import config
from vision.detector import ColorDetector

# --- Where annotated frames land ---
OUTPUT_DIR = Path(__file__).parent / "captures"
LIVE_FRAME = OUTPUT_DIR / "latest.jpg"
LIVE_MASK_RED = OUTPUT_DIR / "latest_mask_red.jpg"
LIVE_MASK_GREEN = OUTPUT_DIR / "latest_mask_green.jpg"

# --- Drawing colors (same palette as the laptop validator) ---
BOX_BGR = {"red": (0, 0, 255), "green": (0, 255, 0)}
DOT_BGR = (255, 255, 0)
TEXT_BGR = (255, 255, 255)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--color", choices=["both", "red", "green"], default="both")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--save-fps", type=float, default=10.0,
                   help="Disk-write rate in Hz")
    p.add_argument("--archive", action="store_true",
                   help="Also save every written frame to captures/archive/")
    return p.parse_args()


def _color_mask(detector, frame, color):
    """Mirror of the masking pipeline inside ColorDetector — used for visual debugging only."""
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
    full = np.zeros((height, width), dtype=np.uint8)
    full[roi_start_y:height, 0:width] = mask
    return full


def _draw_detection(frame, color_name, center_x, bbox, frame_center_x):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_BGR[color_name], 2)
    cv2.circle(frame, (center_x, y + h // 2), 5, DOT_BGR, -1)
    error = center_x - frame_center_x
    label_y = 30 if color_name == "red" else 60
    cv2.putText(frame, f"{color_name}: cX={center_x}  err={error:+d}px",
                (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                BOX_BGR[color_name], 2)


def main():
    args = _parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_dir = OUTPUT_DIR / "archive"
    if args.archive:
        archive_dir.mkdir(exist_ok=True)

    detector = ColorDetector(config)

    # Picamera2 quirk: requesting "RGB888" actually yields a numpy array whose
    # channel order is B, G, R — i.e. already what OpenCV expects. So we do NOT
    # cvtColor afterwards; an extra swap here would flip R and B and break red
    # detection while leaving green intact (green sits in the middle channel).
    picam2 = Picamera2()
    cam_cfg = picam2.create_preview_configuration(
        main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
    )
    picam2.configure(cam_cfg)
    picam2.start()
    time.sleep(0.5)  # let auto-exposure settle

    print(f"Capture size: {config.CAMERA_RESOLUTION}")
    print(f"Color mode  : {args.color}")
    print(f"Save masks  : {args.save_masks}")
    print(f"Archive     : {args.archive}")
    print(f"Live preview: {LIVE_FRAME}")
    print("Open the latest.jpg in VS Code; it auto-refreshes. Ctrl+C to stop.")

    save_interval = 1.0 / max(args.save_fps, 0.1)
    last_save = 0.0
    archive_idx = 0

    fps_t0 = time.monotonic()
    fps_frames = 0
    fps_value = 0.0

    try:
        while True:
            # Picamera2 returns BGR-ordered data here (despite the "RGB888" name).
            # Feed it straight into OpenCV — no cvtColor.
            frame = picam2.capture_array()
            if frame.shape[2] == 4:           # drop alpha if format gave us 4 channels
                frame = frame[:, :, :3]

            # Vertical reference line at the frame's horizontal center
            cv2.line(frame, (detector.frame_center_x, 0),
                     (detector.frame_center_x, frame.shape[0]),
                     (200, 200, 200), 1)

            # Detect requested colors
            active = ("red", "green") if args.color == "both" else (args.color,)
            for c in active:
                center_x, bbox = detector.get_object_center(frame, color=c)
                if center_x is not None:
                    _draw_detection(frame, c, center_x, bbox, detector.frame_center_x)

            # FPS — averaged over 0.5 s windows
            fps_frames += 1
            elapsed = time.monotonic() - fps_t0
            if elapsed >= 0.5:
                fps_value = fps_frames / elapsed
                fps_t0 = time.monotonic()
                fps_frames = 0
            cv2.putText(frame, f"{fps_value:4.1f} FPS  mode={args.color}",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        TEXT_BGR, 2)

            # Throttled disk writes
            now = time.monotonic()
            if now - last_save >= save_interval:
                cv2.imwrite(str(LIVE_FRAME), frame)
                if args.save_masks:
                    cv2.imwrite(str(LIVE_MASK_RED),
                                _color_mask(detector, frame, "red"))
                    cv2.imwrite(str(LIVE_MASK_GREEN),
                                _color_mask(detector, frame, "green"))
                if args.archive:
                    cv2.imwrite(str(archive_dir / f"frame_{archive_idx:05d}.jpg"),
                                frame)
                    archive_idx += 1
                last_save = now

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        picam2.stop()
        print(f"Final FPS    : {fps_value:.1f}")
        print(f"Last frame at: {LIVE_FRAME}")


if __name__ == "__main__":
    main()
