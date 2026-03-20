"""
PID-based heading controller for parafoil trajectory tracking.

This controller intentionally reuses the same trajectory selection,
lookahead guidance, symmetric brake scheduling, and brake allocation logic
as the ADRC controller so that comparisons isolate the inner-loop
heading-control method as much as possible.
"""

import numpy as np

from control.adrc_controller import ParafoilADRCController


class HeadingPID:
    """Wrapped-angle PID with anti-windup and filtered yaw-rate feedback."""

    def __init__(
        self,
        kp: float = 1.6,
        ki: float = 0.08,
        kd: float = 0.55,
        integral_limit: float = 0.8,
        derivative_alpha: float = 0.85,
        u_min: float = -1.0,
        u_max: float = 1.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = abs(integral_limit)
        self.derivative_alpha = np.clip(derivative_alpha, 0.0, 0.999)
        self.u_min = u_min
        self.u_max = u_max

        self.integral = 0.0
        self.prev_y = None
        self.filtered_y_rate = 0.0
        self.u_last = 0.0

    def reset(self, y0: float = 0.0):
        self.integral = 0.0
        self.prev_y = self._wrap_angle(y0)
        self.filtered_y_rate = 0.0
        self.u_last = 0.0

    def update(self, ref: float, y: float, dt: float) -> float:
        ref = self._wrap_angle(ref)
        y = self._wrap_angle(y)

        if dt <= 0.0:
            return float(np.clip(self.u_last, self.u_min, self.u_max))

        if self.prev_y is None:
            self.prev_y = y

        error = self._wrap_angle(ref - y)
        raw_y_rate = self._wrap_angle(y - self.prev_y) / dt
        self.filtered_y_rate = (
            self.derivative_alpha * self.filtered_y_rate
            + (1.0 - self.derivative_alpha) * raw_y_rate
        )

        candidate_integral = np.clip(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        u_candidate = (
            self.kp * error
            + self.ki * candidate_integral
            - self.kd * self.filtered_y_rate
        )
        if not ((u_candidate > self.u_max and error > 0.0) or
                (u_candidate < self.u_min and error < 0.0)):
            self.integral = candidate_integral

        u = (
            self.kp * error
            + self.ki * self.integral
            - self.kd * self.filtered_y_rate
        )
        self.u_last = float(np.clip(u, self.u_min, self.u_max))
        self.prev_y = y
        return self.u_last

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class ParafoilPIDController(ParafoilADRCController):
    """
    PID version of the parafoil controller.

    It subclasses the ADRC controller and swaps only the heading inner loop,
    keeping the rest of the closed-loop tracking stack unchanged.
    """

    def __init__(
        self,
        heading_kp: float = 1.6,
        heading_ki: float = 0.08,
        heading_kd: float = 0.55,
        heading_integral_limit: float = 0.8,
        heading_derivative_alpha: float = 0.85,
        **kwargs,
    ):
        super().__init__(
            heading_kp=heading_kp,
            heading_kd=heading_kd,
            **kwargs,
        )
        self.heading_adrc = HeadingPID(
            kp=heading_kp,
            ki=heading_ki,
            kd=heading_kd,
            integral_limit=heading_integral_limit,
            derivative_alpha=heading_derivative_alpha,
            u_min=-self.max_deflection,
            u_max=self.max_deflection,
        )
        self.reset()
