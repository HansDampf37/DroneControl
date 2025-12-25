import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from .renderer import DroneEnvRenderer
from .drone import Drone
from .wind import Wind


class DroneEnv(gym.Env):
    """
    Gymnasium environment for quadcopter control with reinforcement learning.

    The drone (X-configuration) must fly to a target point and stay there.

    - Action Space: 4 motors, each [-1, 1] for thrust changes (positive increases thrust)
    - Observation Space: Relative position to target, velocity, acceleration (linear),
                         orientation (Euler angles), angular velocity, angular acceleration,
                         wind vector
    - Physics: Simplified model with forces perpendicular to rotor plane, scaled by motor power
    - Wind: Dynamic, changes over time (Ornstein-Uhlenbeck process)
    - Reward: Dense reward function: exp(-distance)
    - Termination: After fixed max_steps or when drone crashes
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_steps: int = 1000,
        dt: float = 0.01,  # Timestep in seconds (100 Hz)
        target_change_interval: Optional[int] = None,  # Steps until target changes
        wind_strength_range: Tuple[float, float] = (0.0, 5.0),  # m/s
        use_wind: bool = False,  # Enable wind simulation
        render_mode: Optional[str] = None,
        # Crash detection parameters
        enable_crash_detection: bool = True,
        enable_out_of_bounds_detection: bool = True,
        crash_z_vel_threshold: float = -20.0,  # Drone falling faster than 20 m/s = crash
        crash_tilt_threshold: float = 80.0,  # Roll/pitch above this angle = crash (degrees)
    ):
        """
        Initializes the drone environment.

        Args:
            max_steps: Maximum number of steps per episode before truncation. Default is 1000.
            dt: Timestep duration in seconds. Default is 0.01s (100 Hz simulation rate).
                Smaller values increase accuracy but slow down simulation.
            target_change_interval: Number of steps after which the target position changes.
                If None, target remains fixed throughout the episode. Default is None.
            wind_strength_range: Tuple of (min, max) wind speeds in m/s. Wind is randomly
                sampled within this range using Ornstein-Uhlenbeck process. Default is (0.0, 5.0).
            use_wind: Whether to enable wind simulation. If False, wind is always zero
                regardless of wind_strength_range. Default is False.
            render_mode: Rendering mode. Options are:
                - None: No rendering (fastest)
                - "human": Interactive matplotlib visualization
                - "rgb_array": Returns RGB arrays for video recording
            enable_crash_detection: Whether to detect and terminate on crashes.
                If True, episode ends when crash is detected. Default is True.
            enable_out_of_bounds_detection: Whether to detect and terminate on large distances to the target position
                If True, episode ends when the drone is too far away from the target. Default is True.
            crash_z_vel_threshold: Threshold for crash detection based on vertical velocity in m/s.
                Negative values indicate downward motion. If drone falls faster than this value,
                a crash is detected. Default is -20.0 m/s.
            crash_tilt_threshold: Threshold for crash detection based on tilt angle in degrees.
                If absolute roll or pitch exceeds this angle, a crash is detected.
                Default is 80.0 degrees.
        """
        super().__init__()

        # Validate render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. "
                f"Supported modes: {self.metadata['render_modes']}"
            )

        self.max_steps = max_steps
        self.dt = dt
        self.target_change_interval = target_change_interval
        self.render_mode = render_mode

        # Crash detection
        self.enable_crash_detection = enable_crash_detection
        self.enable_out_of_bounds_detection = enable_out_of_bounds_detection
        self.crash_z_vel_threshold = crash_z_vel_threshold
        self.crash_tilt_threshold = np.deg2rad(crash_tilt_threshold)  # Convert to radians

        # Simulation parameters
        self.gravity = 9.81  # m/s^2
        self.max_velocity_component = 40.0  # m/s
        self.max_angular_velocity_component = 10.0  # rad/s

        # Create drone (only intrinsic properties)
        self.drone = Drone(
            mass=1.0,
            arm_length=0.10,
            inertia=np.array([0.01, 0.01, 0.02]),
            thrust_coef=10.0,
            torque_coef=0.1,
            linear_drag_coef=0.01,
            angular_drag_coef=0.05,
            center_of_mass_offset=0.05,
            pendulum_damping=0.9,
        )

        # Wind simulation
        self.wind = Wind(
            strength_range=wind_strength_range,
            theta=0.5,  # Mean reversion rate
            sigma=1.0,   # Volatility
            enabled=use_wind
        )

        # Action Space: 4 motors, each [-1, 1] for thrust changes
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.space_side_length = 3
        max_wind_velocity = self.wind.strength_range[1]

        # Maximum acceleration bounds (reasonable physical limits for a drone)
        self.max_acceleration = 50.0  # m/s^2
        self.max_angular_acceleration = 100.0  # rad/s^2

        # Observation Space
        # [0:3]   - Position relative to target (x, y, z)
        # [3:6]   - Linear velocity (vx, vy, vz)
        # [6:9]   - Linear acceleration (ax, ay, az)
        # [9:12]  - Orientation as Euler angles (roll, pitch, yaw)
        # [12:15] - Angular velocity (wx, wy, wz)
        # [15:18] - Angular acceleration (awx, awy, awz)
        # [18:21] - Normal vector (drone facing direction, unit vector)
        # [21:24] - Wind vector absolute (wx, wy, wz)
        obs_low = np.array(
            [-self.space_side_length / 2] * 3 +  # relative position
            [-self.max_velocity_component] * 3 +  # velocity
            [-self.max_acceleration] * 3 +  # linear acceleration
            [-np.pi] * 3 +  # Euler angles
            [-self.max_angular_velocity_component] * 3 +  # angular velocity
            [-self.max_angular_acceleration] * 3 +  # angular acceleration
            [-1.0] * 3 +  # normal vector (unit vector, each component in [-1, 1])
            [-max_wind_velocity] * 3,  # wind
            dtype=np.float32
        )
        obs_high = np.array(
            [self.space_side_length / 2] * 3 + # relative position
            [self.max_velocity_component] * 3 +  # velocity
            [self.max_acceleration] * 3 +  # linear acceleration
            [np.pi] * 3 +  # Euler angles
            [self.max_angular_velocity_component] * 3 +  # angular velocity
            [self.max_angular_acceleration] * 3 +  # angular acceleration
            [1.0] * 3 +  # normal vector (unit vector, each component in [-1, 1])
            [max_wind_velocity] * 3, # wind
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.max_dist_to_target = np.sqrt(3 * self.space_side_length ** 2)

        # Environment state
        self.target_position = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.initial_distance = 0

        # Rendering
        self.renderer = DroneEnvRenderer(render_mode=render_mode, space_side_length=self.space_side_length)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to initial state.

        Args:
            seed: Random seed for reproducibility. If provided, the environment's
                random number generator is seeded. Default is None.
            options: Additional reset options (currently unused). Default is None.

        Returns:
            Tuple of (observation, info) where:
            - observation: Initial observation array
            - info: Dictionary with additional information (distance, positions, etc.)
        """
        super().reset(seed=seed)

        # Reset drone
        initial_orientation = np.random.uniform(-0.1, 0.1, 3).astype(np.float32)
        self.drone.reset(initial_orientation=initial_orientation, gravity=self.gravity)

        # Random target point
        self.target_position = self._generate_random_target()
        self.initial_distance = np.linalg.norm(self.target_position - self.drone.position)

        # Reset wind
        self.wind.reset()

        self.step_count = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one simulation timestep.

        Args:
            action: Array of 4 motor thrust change values in range [-1, 1].
                Positive values increase thrust, negative values decrease it.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) where:
            - observation: Current observation array after step
            - reward: Reward obtained in this step
            - terminated: True if episode ended due to crash
            - truncated: True if episode ended due to max_steps reached
            - info: Dictionary with additional information including 'crashed' flag
        """
        action = np.clip(action, -1.0, 1.0)

        # Wind update
        self.wind.update(self.dt)

        # Target position update (if configured)
        if self.target_change_interval is not None:
            if self.step_count > 0 and self.step_count % self.target_change_interval == 0:
                self.target_position = self._generate_random_target()

        # Physics update via Drone class with simulation parameters
        self.drone.update(
            thrust_changes=action,
            dt=self.dt,
            wind_vector=self.wind.get_vector(),
            gravity=self.gravity,
            max_velocity=self.max_velocity_component,
            max_angular_velocity=self.max_angular_velocity_component,
        )

        # Observation and reward
        observation = self._get_observation()
        reward = self._compute_reward()

        # Termination
        self.step_count += 1
        crashed = self._check_crash()
        out_of_bounds = self._check_out_of_bounds()
        terminated = crashed or out_of_bounds # Episode ends on crash or if drone leaves space
        truncated = self.step_count >= self.max_steps

        info = self._get_info()
        info['crashed'] = crashed  # Add crash info
        info['out_of_bounds'] = out_of_bounds

        return observation, reward, terminated, truncated, info


    def _generate_random_target(self) -> np.ndarray:
        """
        Generates a random target point in 3D space using spherical coordinates.

        The target is positioned within a hemisphere around the drone's starting position,
        with random distance, azimuth (horizontal angle), and elevation (vertical angle).

        Returns:
            Random target position as [x, y, z] array in meters.
        """
        # Random position in spherical coordinates
        # Ensure target is not at origin to avoid zero initial distance
        min_distance = 1.0  # Minimum 1 meter from origin

        while True:
            x = np.random.uniform(-self.space_side_length / 2, self.space_side_length / 2)
            y = np.random.uniform(-self.space_side_length / 2, self.space_side_length / 2)
            z = np.random.uniform(-self.space_side_length / 2, self.space_side_length / 2)

            target = np.array([x, y, z], dtype=np.float32)

            # Check if target is far enough from origin (drone starting position)
            if np.linalg.norm(target) >= min_distance:
                return target

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the current observation vector.

        The observation includes:
        - [0:3]   Relative vector to target (target_pos - drone_pos)
        - [3:6]   Linear velocity of drone
        - [6:9]   Linear acceleration of drone (from previous timestep)
        - [9:12]  Orientation (Euler angles: roll, pitch, yaw)
        - [12:15] Angular velocity
        - [15:18] Angular acceleration (from previous timestep)
        - [18:21] Normal vector (drone facing direction, unit vector)
        - [21:24] Wind vector (absolute wind velocity)

        Returns:
            Observation array with 24 elements.
        """
        vect_to_target = self.target_position - self.drone.position
        observation = np.concatenate([
            vect_to_target,                # [0:3]
            self.drone.velocity,           # [3:6]
            self.drone.acceleration,       # [6:9]
            self.drone.orientation,        # [9:12]
            self.drone.angular_velocity,   # [12:15]
            self.drone.angular_acceleration, # [15:18]
            self.drone.get_normal(),       # [18:21]
            self.wind.get_vector(),        # [21:24]
        ]).astype(np.float32)

        return observation

    def _compute_reward(self) -> float:
        """
        Computes the dense reward based on distance to target.

        The reward is calculated as: ((max_distance - distance) / max_distance)^2
        This quadratic formulation provides:
        - Reward of 1.0 when exactly at target (distance = 0)
        - Reward approaching 0.0 as distance approaches max_distance
        - Stronger gradient near the target for better learning

        Returns:
            Reward value in range [0.0, 1.0].
        """
        distance = np.linalg.norm(self.target_position - self.drone.position)
        return np.exp(-distance)

    def _check_crash(self) -> bool:
        """
        Checks if the drone has crashed.

        Uses two criteria:
        1. Vertical velocity threshold (primary, most efficient)
        2. Extreme tilt (secondary, for uncontrolled drone)

        Returns:
            True if crash detected, False otherwise.
        """
        if not self.enable_crash_detection:
            return False

        return self.drone.check_crash(
            z_velocity_threshold=self.crash_z_vel_threshold,
            tilt_threshold_rad=self.crash_tilt_threshold
        )

    def _check_out_of_bounds(self) -> bool:
        if not self.enable_out_of_bounds_detection:
            return False

        return np.linalg.norm(self.drone.position - self.target_position) > self.max_dist_to_target

    def _get_info(self) -> Dict[str, Any]:
        """
        Returns additional information about the current state.

        Returns:
            Dictionary containing:
            - distance_to_target: Euclidean distance to target in meters
            - position: Current drone position [x, y, z]
            - target_position: Target position [x, y, z]
            - step_count: Current step number in episode
        """
        distance = np.linalg.norm(self.target_position - self.drone.position)
        distance_progress = self.initial_distance - distance
        return {
            "distance_to_target": float(distance),
            "position": self.drone.position.copy(),
            "target_position": self.target_position.copy(),
            "step_count": self.step_count,
            "episode_time": self.step_count * self.dt,
            "distance_progress": distance_progress,
        }

    def render(self):
        """
        Renders the environment visualization.

        Displays 2D views (top-down XY and front XZ) similar to technical drawings.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode is None:
            return None

        # Get rotation matrix from drone
        R = self.drone.get_rotation_matrix()

        # Delegate to renderer
        return self.renderer.render(
            position=self.drone.position,
            velocity=self.drone.velocity,
            orientation=self.drone.orientation,
            angular_velocity=self.drone.angular_velocity,
            target_position=self.target_position,
            wind_vector=self.wind.get_vector(),
            rotation_matrix=R,
            rotor_positions=self.drone.rotor_positions,
            step_count=self.step_count,
            reward=self._compute_reward(),
            motor_thrusts=self.drone.motor_thrusts,
            dt=self.dt
        )

    def close(self):
        """Closes the environment and cleans up resources."""
        self.renderer.close()

