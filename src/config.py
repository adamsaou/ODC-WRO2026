# config.py

# === Camera Settings ===
# Low resolution = Higher FPS
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 30

# Color Thresholds (HSV Ranges for OpenCV) 
HSV_RED_LOW = [0, 120, 70]
HSV_RED_HIGH = [10, 255, 255]

HSV_GREEN_LOW = [35, 60, 40]
HSV_GREEN_HIGH = [85, 255, 255]

#=== PID Controller Constants ===
KP = 0.5  # Proportional
KI = 0.0  # Integral
KD = 0.1  # Derivative

# === Hardware Pin Assignments ===

        # Steering Servo (PWM)
SERVO_PIN = 18 

        # L298N Motor Driver
MOTOR_PWM_PIN = 12  # ENA Pin
MOTOR_DIR_PIN1 = 23 # IN1 Pin
MOTOR_DIR_PIN2 = 24 # IN2 Pin