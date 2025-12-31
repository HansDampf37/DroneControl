"""
Drone model for simulation (more physically grounded).
Key changes vs. your version:
- Motor dynamics: first-order lag (tau) instead of dt/time_to_full_thrust ramp
- Thrust model: per-motor thrust = max_thrust_per_motor * cmd^2 (cmd in [0,1])
- Orientation: quaternion state (avoids Euler integration artifacts)
- Aerodynamic drag: Fd = -0.5*rho*CdA*|v_rel|*v_rel
- Angular damping: torque = -k_w*omega - k_w2*|omega|*omega (dt-consistent)
- Optional "pendulum stabilization" removed by default (kept as an optional feature flag)
"""

import numpy as np
from typing import Tuple, Optional


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product, q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    # q = [w, x, y, z]
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # ZYX (yaw-pitch-roll)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return quat_normalize(np.array([w, x, y, z], dtype=np.float32))


def euler_from_quat(q: np.ndarray) -> np.ndarray:
    # returns roll, pitch, yaw (ZYX)
    w, x, y, z = q

    # roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


class Drone:
    """
    Quadcopter drone in X-configuration (physics-oriented).

    Notes on parameters:
    - max_thrust_per_motor: Newton at cmd=1.0. Choose so hover cmd ~ 0.35..0.65.
      Example: mass=1.0 kg => weight ~ 9.81 N => per-motor hover thrust ~ 2.45 N.
      If max_thrust_per_motor=10 N, hover cmd ~ sqrt(2.45/10)=0.495.
    - motor_tau: first-order time constant (s). Typical: 0.03..0.12 depending on prop/esc.
    - CdA: drag area (m^2). Typical small quad: ~0.01..0.05.
    - k_w, k_w2: rotational damping coefficients (N*m/(rad/s) and N*m/(rad/s)^2).
    """

    def __init__(
        self,
        mass: float = 1.0,
        arm_length: float = 0.25,
        inertia: Optional[np.ndarray] = None,  # [Ix, Iy, Iz]
        max_thrust_per_motor: float = 10.0,  # N
        yaw_moment_per_thrust: float = 0.02,  # Nm per N (very rough)
        motor_tau: float = 0.08,  # s (first-order lag)
        rho: float = 1.225,  # kg/m^3
        CdA: float = 0.02,  # m^2
        k_w: float = 0.02,  # N*m/(rad/s)
        k_w2: float = 0.002,  # N*m/(rad/s)^2
        enable_pendulum: bool = False,
        center_of_mass_offset: float = 0.03,
        pendulum_k: float = 0.0,
    ):
        self.mass = float(mass)
        self.arm_length = float(arm_length)

        # Reasonable default inertia for a ~1kg, ~0.25m-arm quad if none is provided
        self.inertia = (
            inertia.astype(np.float32)
            if inertia is not None
            else np.array([0.02, 0.02, 0.04], dtype=np.float32)
        )

        self.max_thrust_per_motor = float(max_thrust_per_motor)
        self.yaw_moment_per_thrust = float(yaw_moment_per_thrust)
        self.motor_tau = max(1e-4, float(motor_tau))

        self.rho = float(rho)
        self.CdA = float(CdA)

        self.k_w = float(k_w)
        self.k_w2 = float(k_w2)

        # Optional pendulum-like restoring torque (default off)
        self.enable_pendulum = bool(enable_pendulum)
        self.center_of_mass_offset = float(center_of_mass_offset)
        self.pendulum_k = float(pendulum_k)

        # Rotor geometry (X)
        angle = np.pi / 4
        self.rotor_positions = np.array(
            [
                [self.arm_length * np.cos(angle), self.arm_length * np.sin(angle), 0.0],   # 0 front-right
                [-self.arm_length * np.cos(angle), -self.arm_length * np.sin(angle), 0.0], # 1 rear-left
                [-self.arm_length * np.cos(angle), self.arm_length * np.sin(angle), 0.0],  # 2 front-left
                [self.arm_length * np.cos(angle), -self.arm_length * np.sin(angle), 0.0],  # 3 rear-right
            ],
            dtype=np.float32,
        )
        # Direction: +1 CW, -1 CCW (choose consistent pattern)
        self.rotor_directions = np.array([1, 1, -1, -1], dtype=np.float32)

        # State
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)

        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w,x,y,z
        self.angular_velocity = np.zeros(3, dtype=np.float32)  # rad/s
        self.angular_acceleration = np.zeros(3, dtype=np.float32)

        # Motor command (desired) and motor state (actual), both in [0,1]
        self.motor_cmd = np.zeros(4, dtype=np.float32)
        self.motor_thrusts = np.zeros(4, dtype=np.float32)  # "actual cmd" after lag, in [0,1]

    def get_rotation_matrix(self) -> np.ndarray:
        return quat_to_rotation_matrix(self.orientation_q)

    def get_euler(self) -> np.ndarray:
        return euler_from_quat(self.orientation_q)

    def get_normal(self) -> np.ndarray:
        return self.get_rotation_matrix() @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def get_hover_thrust_cmd(self, gravity: float = 9.81) -> float:
        total = self.mass * gravity
        per_motor = total / 4.0
        # cmd^2 * maxT = per_motor => cmd = sqrt(per_motor/maxT)
        return float(np.sqrt(np.clip(per_motor / self.max_thrust_per_motor, 0.0, 1.0)))

    def reset(self, initial_orientation: np.ndarray | None = None, gravity: float = 9.81, initial_position: np.ndarray | None = None):
        if initial_position is None:
            self.position[:] = 0
        else:
            self.position[:] = initial_position
        self.velocity[:] = 0
        self.acceleration[:] = 0

        if initial_orientation is None:
            rpy = np.random.uniform(-0.05, 0.05, 3).astype(np.float32)
        else:
            rpy = np.array(initial_orientation, dtype=np.float32)

        self.orientation_q = quat_from_euler(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        self.angular_velocity[:] = 0
        self.angular_acceleration[:] = 0

        hover_cmd = self.get_hover_thrust_cmd(gravity)
        self.motor_cmd[:] = hover_cmd
        self.motor_thrusts[:] = hover_cmd

    def update(
        self,
        motor_cmd: np.ndarray,
        dt: float,
        wind_vector: np.ndarray | None = None,
        gravity: float = 9.81,
        max_velocity: float = 60.0,
        max_angular_velocity: float = 25.0,
    ):
        dt = float(dt)
        if wind_vector is None:
            wind_vector = np.zeros(3, dtype=np.float32)
        else:
            wind_vector = wind_vector.astype(np.float32)

        # 1) Update commanded motor inputs (still normalized 0..1)
        self.motor_cmd = np.clip(motor_cmd.astype(np.float32), 0.0, 1.0)

        # 2) Motor dynamics: first-order lag towards command
        # x += (u - x) * (dt/tau)  (stable for small dt; clamp gain to avoid overshoot if dt>tau)
        alpha = np.clip(dt / self.motor_tau, 0.0, 1.0)
        self.motor_thrusts += (self.motor_cmd - self.motor_thrusts) * alpha
        self.motor_thrusts = np.clip(self.motor_thrusts, 0.0, 1.0)

        # 3) Compute per-motor thrust in Newtons (quadratic mapping)
        thrusts_N = self.max_thrust_per_motor * (self.motor_thrusts ** 2)

        # 4) Forces
        R = self.get_rotation_matrix()
        # Total thrust in body frame (+Z body)
        total_thrust_body = np.array([0.0, 0.0, float(np.sum(thrusts_N))], dtype=np.float32)
        total_thrust_world = R @ total_thrust_body

        gravity_force = np.array([0.0, 0.0, -self.mass * gravity], dtype=np.float32)

        # Aerodynamic drag (quadratic)
        v_rel = self.velocity - wind_vector
        speed = float(np.linalg.norm(v_rel))
        if speed > 1e-6:
            drag_force = -0.5 * self.rho * self.CdA * speed * v_rel
        else:
            drag_force = np.zeros(3, dtype=np.float32)

        total_force = total_thrust_world + gravity_force + drag_force
        self.acceleration = total_force / self.mass

        # 5) Torques from thrusts
        torque = self._compute_torque(thrusts_N)

        # Optional pendulum-like restoring torque (off by default)
        if self.enable_pendulum and self.pendulum_k != 0.0:
            roll, pitch, _ = self.get_euler()
            # very simple restoring torque model (still not "typical quad" physics)
            pend = self.pendulum_k * np.array(
                [
                    -self.mass * gravity * self.center_of_mass_offset * np.sin(roll),
                    -self.mass * gravity * self.center_of_mass_offset * np.sin(pitch),
                    0.0,
                ],
                dtype=np.float32,
            )
            torque = torque + pend

        # Rotational damping torque
        w = self.angular_velocity
        w_norm = float(np.linalg.norm(w))
        torque_drag = -self.k_w * w
        if w_norm > 1e-6:
            torque_drag += -self.k_w2 * w_norm * w
        torque = torque + torque_drag

        self.angular_acceleration = torque / self.inertia

        # 6) Integrate linear (semi-implicit Euler)
        self.velocity += self.acceleration * dt
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        self.position += self.velocity * dt

        # 7) Integrate angular velocity
        self.angular_velocity += self.angular_acceleration * dt
        self.angular_velocity = np.clip(self.angular_velocity, -max_angular_velocity, max_angular_velocity)

        # 8) Integrate quaternion orientation: q_dot = 0.5 * [0, ω] ⊗ q  (ω in body frame)
        omega_q = np.array([0.0, self.angular_velocity[0], self.angular_velocity[1], self.angular_velocity[2]], dtype=np.float32)
        q_dot = 0.5 * quat_mul(omega_q, self.orientation_q)
        self.orientation_q = quat_normalize(self.orientation_q + q_dot * dt)

    def _compute_torque(self, thrusts_N: np.ndarray) -> np.ndarray:
        # Roll torque: right (0,3) - left (1,2)
        roll_torque = (thrusts_N[0] + thrusts_N[3] - thrusts_N[1] - thrusts_N[2]) * (
            self.arm_length / np.sqrt(2)
        )

        # Pitch torque: front (0,2) - rear (1,3)
        pitch_torque = (thrusts_N[0] + thrusts_N[2] - thrusts_N[1] - thrusts_N[3]) * (
            self.arm_length / np.sqrt(2)
        )

        # Yaw torque: reactive moment ~ direction * thrust (rough)
        yaw_torque = float(np.sum(self.rotor_directions * thrusts_N) * self.yaw_moment_per_thrust)

        return np.array([roll_torque, pitch_torque, yaw_torque], dtype=np.float32)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            (position, velocity, euler_rpy, angular_velocity, motor_thrusts)
            - motor_thrusts here are normalized "actual cmd" in [0,1]
        """
        return (
            self.position.copy(),
            self.velocity.copy(),
            self.get_euler().copy(),
            self.angular_velocity.copy(),
            self.motor_thrusts.copy(),
        )

    def set_state(
        self,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        orientation_euler: np.ndarray | None = None,
        angular_velocity: np.ndarray | None = None,
        motor_cmd: np.ndarray | None = None,
        motor_thrusts: np.ndarray | None = None,
    ):
        if position is not None:
            self.position = position.astype(np.float32)
        if velocity is not None:
            self.velocity = velocity.astype(np.float32)
        if orientation_euler is not None:
            r, p, y = orientation_euler.astype(np.float32)
            self.orientation_q = quat_from_euler(float(r), float(p), float(y))
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity.astype(np.float32)
        if motor_cmd is not None:
            self.motor_cmd = np.clip(motor_cmd.astype(np.float32), 0.0, 1.0)
        if motor_thrusts is not None:
            self.motor_thrusts = np.clip(motor_thrusts.astype(np.float32), 0.0, 1.0)

    def check_crash(self, z_velocity_threshold: float = -20.0, tilt_threshold_rad: float | None = None, ground_level: float = 0.0) -> bool:
        # Check if drone hit the ground
        if self.position[2] <= ground_level:
            return True
        # Check if drone is falling too fast
        if self.velocity[2] < z_velocity_threshold:
            return True
        # Check if drone is tilted too much
        if tilt_threshold_rad is not None:
            roll, pitch, _ = self.get_euler()
            if abs(roll) > tilt_threshold_rad or abs(pitch) > tilt_threshold_rad:
                return True
        return False
