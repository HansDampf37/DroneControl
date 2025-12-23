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

    - Action Space: 4 motors, each 0-1 (0-100% thrust)
    - Observation Space: Relative position to target, velocity, orientation (Euler angles),
                         angular velocity, wind vector
    - Physics: Simplified model with forces perpendicular to rotor plane, scaled by motor power
    - Wind: Dynamic, changes over time (Ornstein-Uhlenbeck process)
    - Reward: Dense reward function: ((max_distance - distance) / max_distance) ** 2
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
            crash_z_vel_threshold: Threshold for crash detection based on vertical velocity in m/s.
                Negative values indicate downward motion. If drone falls faster than this value,
                a crash is detected. Default is -20.0 m/s.
            crash_tilt_threshold: Threshold for crash detection based on tilt angle in degrees.
                If absolute roll or pitch exceeds this angle, a crash is detected.
                Default is 80.0 degrees.
        """
        super().__init__()

        self.max_steps = max_steps
        self.dt = dt
        self.target_change_interval = target_change_interval
        self.render_mode = render_mode

        # Crash detection
        self.enable_crash_detection = enable_crash_detection
        self.crash_z_vel_threshold = crash_z_vel_threshold
        self.crash_tilt_threshold = np.deg2rad(crash_tilt_threshold)  # Convert to radians

        # Simulation parameters
        self.gravity = 9.81  # m/s^2
        self.max_velocity_component = 40.0  # m/s
        self.max_angular_velocity_component = 10.0  # rad/s

        # Create drone (only intrinsic properties)
        self.drone = Drone(
            mass=1.0,
            arm_length=0.25,
            inertia=np.array([0.01, 0.01, 0.02]),
            thrust_coef=10.0,
            torque_coef=0.1,
            linear_drag_coef=0.01,
            angular_drag_coef=0.05,
            center_of_mass_offset=0.03,
            pendulum_damping=0.5,
        )

        # Wind simulation
        self.wind = Wind(
            strength_range=wind_strength_range,
            theta=0.15,  # Mean reversion rate
            sigma=1.0,   # Volatility
            enabled=use_wind
        )

        # Action Space: 4 motors, each 0-1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        space_side_length = 50
        max_wind_velocity = self.wind.strength_range[1]

        # Observation Space
        # [0:3]   - Position relative to target (x, y, z)
        # [3:6]   - Linear velocity (vx, vy, vz)
        # [6:9]   - Orientation as Euler angles (roll, pitch, yaw)
        # [9:12]  - Angular velocity (wx, wy, wz)
        # [12:15] - Wind vector absolute (wx, wy, wz)
        obs_low = np.array(
            [-space_side_length] * 3 +  # relative position
            [-self.max_velocity_component] * 3 +  # velocity
            [-np.pi] * 3 +  # Euler angles
            [-self.max_angular_velocity_component] * 3 +  # angular velocity
            [-max_wind_velocity] * 3,   # wind
            dtype=np.float32
        )
        obs_high = np.array(
            [space_side_length] * 3 + # relative position
            [self.max_velocity_component] * 3 +  # velocity
            [np.pi] * 3 +  # Euler angles
            [self.max_angular_velocity_component] * 3 +  # angular velocity
            [max_wind_velocity] * 3,   # wind
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.max_dist_to_target = np.sqrt(3 * space_side_length ** 2)

        # Environment state
        self.target_position = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        # Rendering
        self.renderer = DroneEnvRenderer(render_mode=render_mode)

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
        self.drone.reset(initial_orientation=initial_orientation)

        # Random target point
        self.target_position = self._generate_random_target()

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
            action: Array of 4 motor thrust values in range [0, 1].
                Values outside this range are clipped.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) where:
            - observation: Current observation array after step
            - reward: Reward obtained in this step
            - terminated: True if episode ended due to crash
            - truncated: True if episode ended due to max_steps reached
            - info: Dictionary with additional information including 'crashed' flag
        """
        action = np.clip(action, 0.0, 1.0)

        # Wind update
        self.wind.update(self.dt)

        # Target position update (if configured)
        if self.target_change_interval is not None:
            if self.step_count > 0 and self.step_count % self.target_change_interval == 0:
                self.target_position = self._generate_random_target()

        # Physics update via Drone class with simulation parameters
        self.drone.update(
            motor_thrusts=action,
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
        terminated = crashed  # Episode ends on crash
        truncated = self.step_count >= self.max_steps

        info = self._get_info()
        info['crashed'] = crashed  # Add crash info

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
        distance = np.random.uniform(0.0, self.max_dist_to_target // 2)
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(-np.pi / 4, np.pi / 4)

        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)

        return np.array([x, y, z], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the current observation vector.

        The observation includes:
        - [0:3]   Relative vector to target (target_pos - drone_pos)
        - [3:6]   Linear velocity of drone
        - [6:9]   Orientation (Euler angles: roll, pitch, yaw)
        - [9:12]  Angular velocity
        - [12:15] Wind vector (absolute wind velocity)

        Returns:
            Observation array with 15 elements.
        """
        vect_to_target = self.target_position - self.drone.position
        observation = np.concatenate([
            vect_to_target,                # [0:3]
            self.drone.velocity,           # [3:6]
            self.drone.orientation,        # [6:9]
            self.drone.angular_velocity,   # [9:12]
            self.wind.get_vector(),        # [12:15]
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
        distance = np.clip(distance, 0, self.max_dist_to_target)
        margin = (self.max_dist_to_target - distance) / self.max_dist_to_target
        return float(margin ** 2)

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
            tilt_threshold_rad=self.crash_tilt_threshold,
            max_distance=self.max_dist_to_target,
            target_position=self.target_position
        )

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
        return {
            "distance_to_target": float(distance),
            "position": self.drone.position.copy(),
            "target_position": self.target_position.copy(),
            "step_count": self.step_count,
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
            reward=self._compute_reward()
        )

    def close(self):
        """Closes the environment and cleans up resources."""
        self.renderer.close()

