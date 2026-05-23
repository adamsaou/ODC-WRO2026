
import cv2
import numpy as np

class ColorDetector:
    def __init__(self, config):
        """
        Pass your config settings into the detector so it knows 
        what color ranges and resolutions to use.
        """
        self.config = config
        
        # Convert config lists to numpy arrays for OpenCV processing
        self.red_low = np.array(config.HSV_RED_LOW)
        self.red_high = np.array(config.HSV_RED_HIGH)
        self.green_low = np.array(config.HSV_GREEN_LOW)
        self.green_high = np.array(config.HSV_GREEN_HIGH)
        
        # The center of your frame (e.g., 640 / 2 = 320)
        self.frame_center_x = config.CAMERA_RESOLUTION[0] // 2

    def get_object_center(self, frame, color="red"):
        """
        Finds the center X coordinate of the largest target color blob.
        Returns None if no object of that color is found.
        """
        # 1. Crop to Region of Interest (Ignore top 40% of the screen to save Pi 4 processing power)
        height, width, _ = frame.shape
        roi_start_y = int(height * 0.4)
        roi_frame = frame[roi_start_y:height, 0:width]

        # 2. Convert from BGR (standard) to HSV (stable color tracking)
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        # 3. Create a binary mask (makes matching pixels white, others black)
        if color == "red":
            mask = cv2.inRange(hsv, self.red_low, self.red_high)
        else:
            mask = cv2.inRange(hsv, self.green_low, self.green_high)

        # 4. Clean up the image (remove tiny speckles of light/noise)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 5. Find contours (the outlines of the colored shapes)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (the closest/biggest pillar)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter out tiny objects that are too far away to matter
            if cv2.contourArea(largest_contour) > 200:
                # Calculate the center (Moments) of the shape
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"]) + roi_start_y # Adjust back to full frame coordinate
                    
                    # Return the horizontal center and the bounding box for drawing
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    return cX, (x, y + roi_start_y, w, h)

        return None, None