# control/pid.py

class PID:
    """
    Generic discrete PID controller.

    Sensor-agnostic — instantiate one per control axis:
        - Vision steering   (error = pillar_cX - frame_center_x)
        - IMU heading hold  (error = target_yaw - current_yaw)
        - Wall follow       (error = left_tof - right_tof)

    The caller supplies dt every step (deterministic, easy to unit-test;
    no hidden time.monotonic() calls).
    """

    def __init__(self, kp, ki, kd,
                 setpoint=0.0,
                 output_limits=(None, None),
                 integral_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        # Clamp the final output (e.g. servo angle range, PWM duty range).
        self.out_min, self.out_max = output_limits

        # Clamp the integral term to prevent wind-up when the actuator saturates.
        self.i_min, self.i_max = integral_limits

        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        """Call between runs or when the control axis is re-engaged."""
        self._integral = 0.0
        self._prev_error = None

    def update(self, measurement, dt):
        """
        Run one PID step.

        measurement : current sensor reading (cX, yaw, distance, ...)
        dt          : seconds since the previous call (caller-supplied)

        Returns the clamped control output.
        """
        if dt <= 0.0:
            # Caller passed a stale/zero dt — return last-known proportional
            # term only, avoid divide-by-zero in the derivative.
            return self._clamp(self.kp * (self.setpoint - measurement),
                               self.out_min, self.out_max)

        error = self.setpoint - measurement

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral = self._clamp(self._integral, self.i_min, self.i_max)

        # Derivative on error (first call has no previous sample → 0)
        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = (self.kp * error
                  + self.ki * self._integral
                  + self.kd * derivative)

        return self._clamp(output, self.out_min, self.out_max)

    @staticmethod
    def _clamp(value, lo, hi):
        if lo is not None and value < lo:
            return lo
        if hi is not None and value > hi:
            return hi
        return value
