"""
Drone model for simulation.
Encapsulates the physics and state of a quadcopter drone.
"""
import numpy as np
from typing import Tuple


class Drone:
    """
    Quadcopter drone in X-configuration.

    Manages:
    - Physical parameters (mass, inertia, etc.)
    - State (position, velocity, orientation, etc.)
    - Physics simulation (forces, torques, integration)
    """

    def __init__(
        self,
        mass: float = 1.0,
        arm_length: float = 0.25,
        inertia: np.ndarray = None,
        thrust_coef: float = 10.0,
        torque_coef: float = 0.1,
        linear_drag_coef: float = 0.01,
        angular_drag_coef: float = 0.05,
        center_of_mass_offset: float = 0.03,
        pendulum_damping: float = 0.5,
    ):
        """
        Initializes the drone with its physical properties.

        Args:
            mass: Mass of the drone in kilograms. Default is 1.0 kg.
            arm_length: Distance from center to each rotor in meters. Default is 0.25 m.
            inertia: Moment of inertia tensor [Ix, Iy, Iz] in kg*m^2.
                If None, defaults to [0.01, 0.01, 0.02] for a typical quadcopter.
            thrust_coef: Thrust coefficient that maps motor input to force.
                Higher values mean more thrust per motor input. Default is 10.0.
            torque_coef: Torque coefficient for reactive torque from rotor spin.
                Default is 0.1.
            linear_drag_coef: Linear drag coefficient for air resistance.
                Drag force scales linearly with velocity. Default is 0.01.
            angular_drag_coef: Angular drag coefficient for rotational damping.
                Reduces angular velocity over time. Default is 0.05.
            center_of_mass_offset: Offset of center of mass from geometric center in meters.
                Creates a pendulum stabilization effect. Default is 0.03 m.
            pendulum_damping: Damping factor for pendulum stabilization effect.
                Higher values provide stronger self-stabilization. Default is 0.5.
        """
        # Intrinsic physical parameters of the drone
        self.mass = mass
        self.arm_length = arm_length
        self.inertia = inertia if inertia is not None else np.array([0.01, 0.01, 0.02])
        self.thrust_coef = thrust_coef
        self.torque_coef = torque_coef
        self.linear_drag_coef = linear_drag_coef
        self.angular_drag_coef = angular_drag_coef
        self.center_of_mass_offset = center_of_mass_offset
        self.pendulum_damping = pendulum_damping

        # Rotor configuration (X-formation)
        # Motor 0: front-right (+x, +y), Motor 1: rear-left (-x, -y)
        # Motor 2: front-left (-x, +y), Motor 3: rear-right (+x, -y)
        angle = np.pi / 4  # 45 degrees
        self.rotor_positions = np.array([
            [self.arm_length * np.cos(angle), self.arm_length * np.sin(angle), 0],      # Motor 0
            [-self.arm_length * np.cos(angle), -self.arm_length * np.sin(angle), 0],    # Motor 1
            [-self.arm_length * np.cos(angle), self.arm_length * np.sin(angle), 0],     # Motor 2
            [self.arm_length * np.cos(angle), -self.arm_length * np.sin(angle), 0],     # Motor 3
        ], dtype=np.float32)

        # Rotor rotation directions (1 = CW, -1 = CCW)
        self.rotor_directions = np.array([1, 1, -1, -1])

        # State
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(3, dtype=np.float32)  # Roll, Pitch, Yaw
        self.angular_velocity = np.zeros(3, dtype=np.float32)

    def reset(self, initial_orientation: np.ndarray = None):
        """
        Resets the drone to its initial state.

        Args:
            initial_orientation: Optional initial orientation as [roll, pitch, yaw] in radians.
                If None, a small random orientation is applied.
        """
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)

        if initial_orientation is not None:
            self.orientation = initial_orientation.astype(np.float32)
        else:
            # Small random initial orientation
            self.orientation = np.random.uniform(-0.1, 0.1, 3).astype(np.float32)

        self.angular_velocity = np.zeros(3, dtype=np.float32)

    def update(
        self,
        motor_thrusts: np.ndarray,
        dt: float,
        wind_vector: np.ndarray = None,
        gravity: float = 9.81,
        max_velocity: float = 40.0,
        max_angular_velocity: float = 10.0,
    ):
        """
        Updates the drone physics for one timestep.

        Simulates the physical behavior of the drone based on motor inputs and environmental
        factors using simplified Euler integration.

        Args:
            motor_thrusts: Array of 4 normalized thrust values in range [0, 1], one per motor.
                Order: [front-right, rear-left, front-left, rear-right].
            dt: Timestep duration in seconds. Smaller values increase accuracy but require
                more computation steps.
            wind_vector: Optional wind velocity vector [wx, wy, wz] in m/s.
                If None, no wind is applied. Wind affects drag forces.
            gravity: Gravitational acceleration in m/s². Default is 9.81 m/s² (Earth).
            max_velocity: Maximum linear velocity component in m/s. Velocity is clipped
                to this value to prevent numerical instability. Default is 40.0 m/s.
            max_angular_velocity: Maximum angular velocity component in rad/s. Angular
                velocity is clipped to this value. Default is 10.0 rad/s.
        """
        if wind_vector is None:
            wind_vector = np.zeros(3, dtype=np.float32)

        # 1. Rotation from body-frame to world-frame
        R = self.get_rotation_matrix()

        # 2. Thrust direction in world-frame
        thrust_direction_world = R @ np.array([0, 0, 1], dtype=np.float32)

        # 3. Project velocity onto thrust direction
        velocity_in_thrust_direction = np.dot(self.velocity, thrust_direction_world)

        # 4. Thrust modification based on velocity
        # At negative velocity (falling): more thrust efficiency
        # At positive velocity (climbing): less thrust efficiency
        max_speed_in_thrust_dir = 30.0  # m/s
        speed_factor = max(0.0, 1.0 - (velocity_in_thrust_direction / max_speed_in_thrust_dir))

        # 5. Calculate thrust forces
        thrusts = motor_thrusts * self.thrust_coef * speed_factor

        # 6. Total force in body-frame
        total_thrust_body = np.array([0, 0, np.sum(thrusts)], dtype=np.float32)

        # 7. Rotate to world-frame
        total_force_world = R @ total_thrust_body

        # 8. Gravity
        gravity_force = np.array([0, 0, -self.mass * gravity], dtype=np.float32)

        # 9. Air drag (relative to wind)
        relative_velocity = self.velocity - wind_vector
        relative_speed = np.linalg.norm(relative_velocity)

        if relative_speed > 0.01:
            drag_force = -self.linear_drag_coef * relative_speed * relative_velocity
        else:
            drag_force = np.zeros(3, dtype=np.float32)

        # 10. Total force and linear acceleration
        total_force = total_force_world + gravity_force + drag_force
        linear_acceleration = total_force / self.mass

        # 11. Calculate torque
        torque = self._compute_torque(thrusts)

        # 12. Pendulum stabilization
        roll, pitch, _ = self.orientation
        pendulum_torque = self.pendulum_damping * np.array([
            -self.mass * gravity * self.center_of_mass_offset * np.sin(roll),
            -self.mass * gravity * self.center_of_mass_offset * np.sin(pitch),
            0.0
        ], dtype=np.float32)

        # 13. Angular acceleration
        angular_acceleration = (torque + pendulum_torque) / self.inertia

        # 14. Integration (Euler method)
        self.velocity += linear_acceleration * dt
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        self.position += self.velocity * dt

        self.angular_velocity += angular_acceleration * dt
        self.angular_velocity *= (1 - self.angular_drag_coef)
        self.angular_velocity = np.clip(
            self.angular_velocity,
            -max_angular_velocity,
            max_angular_velocity
        )
        self.orientation += self.angular_velocity * dt

        # Normalize Euler angles to [-pi, pi]
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

    def _compute_torque(self, thrusts: np.ndarray) -> np.ndarray:
        """
        Calculates the torque based on rotor thrusts.

        In X-configuration:
        - Roll: Thrust difference between right and left motors
        - Pitch: Thrust difference between front and rear motors
        - Yaw: Difference in rotation directions (reactive torque)

        Args:
            thrusts: Array of 4 thrust values, one per motor.

        Returns:
            Torque vector [roll_torque, pitch_torque, yaw_torque] in N*m.
        """
        # Roll torque (around X-axis): Right (0, 3) vs. Left (1, 2)
        roll_torque = (thrusts[0] + thrusts[3] - thrusts[1] - thrusts[2]) * \
                      self.arm_length / np.sqrt(2)

        # Pitch torque (around Y-axis): Front (0, 2) vs. Rear (1, 3)
        pitch_torque = (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3]) * \
                       self.arm_length / np.sqrt(2)

        # Yaw torque (around Z-axis): Reactive torques
        yaw_torque = np.sum(self.rotor_directions * thrusts) * self.torque_coef

        return np.array([roll_torque, pitch_torque, yaw_torque], dtype=np.float32)

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Computes the rotation matrix from body-frame to world-frame.

        Uses Euler angles [roll, pitch, yaw] in ZYX convention (yaw-pitch-roll).

        Returns:
            3x3 rotation matrix that transforms vectors from body coordinates
            to world coordinates.
        """
        roll, pitch, yaw = self.orientation

        # Roll (rotation around X-axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ], dtype=np.float32)

        # Pitch (rotation around Y-axis)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ], dtype=np.float32)

        # Yaw (rotation around Z-axis)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combined rotation: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the current state of the drone.

        Returns:
            Tuple of (position, velocity, orientation, angular_velocity).
            All arrays are copied to prevent external modification.
        """
        return (
            self.position.copy(),
            self.velocity.copy(),
            self.orientation.copy(),
            self.angular_velocity.copy()
        )

    def set_state(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        orientation: np.ndarray = None,
        angular_velocity: np.ndarray = None
    ):
        """
        Sets the state of the drone.

        Args:
            position: New position [x, y, z] in meters. If None, position is unchanged.
            velocity: New velocity [vx, vy, vz] in m/s. If None, velocity is unchanged.
            orientation: New orientation [roll, pitch, yaw] in radians.
                If None, orientation is unchanged.
            angular_velocity: New angular velocity [wx, wy, wz] in rad/s.
                If None, angular velocity is unchanged.
        """
        if position is not None:
            self.position = position.astype(np.float32)
        if velocity is not None:
            self.velocity = velocity.astype(np.float32)
        if orientation is not None:
            self.orientation = orientation.astype(np.float32)
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity.astype(np.float32)

    def check_crash(
        self,
        z_velocity_threshold: float = -20.0,
        tilt_threshold_rad: float = None
    ) -> bool:
        """
        Checks if the drone has crashed.

        A crash is detected if any of the following conditions are met:
        - Vertical velocity exceeds the threshold (falling too fast)
        - Roll or pitch angle exceeds the tilt threshold (unstable orientation)

        Args:
            z_velocity_threshold: Threshold for vertical velocity in m/s.
                Negative values indicate downward motion. Default is -20.0 m/s.
                If drone falls faster than this, a crash is detected.
            tilt_threshold_rad: Threshold for roll/pitch angles in radians.
                If None, tilt checking is disabled. If absolute value of roll
                or pitch exceeds this threshold, a crash is detected.

        Returns:
            True if crash is detected, False otherwise.
        """
        # Falling too fast
        if self.velocity[2] < z_velocity_threshold:
            return True

        # Extreme tilt
        if tilt_threshold_rad is not None:
            roll, pitch, _ = self.orientation
            if abs(roll) > tilt_threshold_rad or abs(pitch) > tilt_threshold_rad:
                return True

        return False

