import abc
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces, Space

from .drone import Drone
from .renderer import DroneEnvRenderer
from .renderer_pygame import PyGameRenderer
from .wind import Wind


class DroneEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Gymnasium environment for quadcopter control with reinforcement learning.

    The drone (X-configuration) must fly to a target point and stay there.

    - Action Space: 4 motors, each [0, 1] for target thrust
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
            use_wind: bool = True,
            render_mode: Optional[str] = None,
            renderer_type: str = "pygame",  # "pygame" or "matplotlib"
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
            renderer_type: Type of renderer to use. Options are:
                - "pygame": Fast pygame-based 3D renderer (default, recommended)
                - "matplotlib": Slower matplotlib-based 2D multi-view renderer
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
            pendulum_k=0.1,
            enable_pendulum=True
        )

        # Wind simulation
        self.wind = Wind(
            strength_range=wind_strength_range,
            theta=0.5,  # Mean reversion rate
            sigma=1.0,  # Volatility
            enabled=use_wind
        )

        # Action Space: 4 motors, each [-1, 1] for thrust changes
        self.action_space = spaces.Box(
            low=0.0,
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
            [-np.inf] * 4 +  # quaternions
            [-self.max_angular_velocity_component] * 3 +  # angular velocity
            [-self.max_angular_acceleration] * 3 +  # angular acceleration
            [-1.0] * 3 +  # normal vector (unit vector, each component in [-1, 1])
            [-max_wind_velocity] * 3 +  # wind
            [0] * 8,  # target and current thrust
            dtype=np.float32
        )
        obs_high = np.array(
            [self.space_side_length / 2] * 3 +  # relative position
            [self.max_velocity_component] * 3 +  # velocity
            [self.max_acceleration] * 3 +  # linear acceleration
            [np.inf] * 4 +  # quaternions
            [self.max_angular_velocity_component] * 3 +  # angular velocity
            [self.max_angular_acceleration] * 3 +  # angular acceleration
            [1.0] * 3 +  # normal vector (unit vector, each component in [-1, 1])
            [max_wind_velocity] * 3 +  # wind
            [1] * 8,  # target and current thrust
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.max_dist_to_target = np.sqrt(3 * self.space_side_length ** 2)
        self.vel_scale = 3

        # Environment state
        self.target_position = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.initial_distance = 0

        # Rendering - choose renderer based on renderer_type
        if renderer_type == "pygame":
            self.renderer = PyGameRenderer(render_mode=render_mode, space_side_length=self.space_side_length)
        elif renderer_type == "matplotlib":
            self.renderer = DroneEnvRenderer(render_mode=render_mode, space_side_length=self.space_side_length)
        else:
            raise ValueError(f"Invalid renderer_type: {renderer_type}. Options are 'pygame' or 'matplotlib'.")

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
        action = np.clip(action, 0.0, 1.0)

        # Wind update
        self.wind.update(self.dt)

        # Target position update (if configured)
        if self.target_change_interval is not None:
            if self.step_count > 0 and self.step_count % self.target_change_interval == 0:
                self.target_position = self._generate_random_target()

        # Physics update via Drone class with simulation parameters
        self.drone.update(
            motor_cmd=action,
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
        terminated = crashed or out_of_bounds  # Episode ends on crash or if drone leaves space
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
        - [9:13]  Orientation (quaternions)
        - [13:16] Angular velocity
        - [16:19] Angular acceleration (from previous timestep)
        - [19:22] Normal vector (drone facing direction, unit vector)
        - [22:25] Wind vector (absolute wind velocity)
        - [25:29] Motor thrusts (currently)
        - [29:33] Motor thrusts (target)
        Returns:
            Observation array with 33 elements.
        """
        vect_to_target = self.target_position - self.drone.position
        observation = np.concatenate([
            vect_to_target,  # [0:3]
            self.drone.velocity,  # [3:6]
            self.drone.acceleration,  # [6:9]
            self.drone.orientation_q,  # [9:13]
            self.drone.angular_velocity,  # [13:16]
            self.drone.angular_acceleration,  # [16:19]
            self.drone.get_normal(),  # [19:22]
            self.wind.get_vector(),  # [22:25]
            self.drone.motor_thrusts,  # [25:29]
            self.drone.motor_cmd,  # [29:33]
        ]).astype(np.float32)

        return observation

    def _compute_reward(self) -> float:
        """
        Computes the dense reward based on distance to target and the velocity towards the target.

        The reward is calculated as: r_pos + alpha * (1 - r_pos) r_vel with
        r_pos = exp(-distance)
        r_vel = tanh(dot(drone_vel , direction_to_target)/beta)
        This formulation provides:
        - Reward of 1.0 when exactly at target (distance = 0)
        - Reward approaching 0.0 when distance grows
        - Stronger gradient near the target for better learning
        - general incentive to move towards target because of velocity term

        Returns:
            Reward value in range [-1.0, 1.0].
        """
        distance = np.linalg.norm(self.target_position - self.drone.position)
        direction_target = (self.target_position - self.drone.position) / (distance + 1e-6)
        correct_vel = np.dot(self.drone.velocity, direction_target)
        reward_position = np.exp(-1.0 * distance)
        reward_vel = np.tanh(correct_vel / self.vel_scale)
        reward = reward_position + 0.5 * (1.0 - reward_position) * reward_vel
        return reward

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

        # Delegate to renderer
        return self.renderer.render(self, Reward=f"{self._compute_reward():.2f}")

    def close(self):
        """Closes the environment and cleans up resources."""
        self.renderer.close()


class SequentialWaypointEnv(DroneEnv):
    """
    Sequential waypoint navigation environment.

    The drone must navigate through a sequence of waypoints as quickly as possible.
    Extends DroneEnv with waypoint tracking, speed-based rewards, and decaying checkpoint bonuses.

    - Observation Space: Extended with next waypoint position (3 additional floats)
    - Reward: Speed in right direction + decaying checkpoint bonus
    - Episode ends: When all waypoints are reached or crash/timeout occurs
    """

    def __init__(
            self,
            max_num_waypoints: int = 15,
            waypoint_reach_threshold_m: float = 0.4,
            checkpoint_bonus: float = 10.0,
            bonus_decay_rate_per_sec: float = 2.0,
            **kwargs
    ):
        """
        Initializes the sequential waypoint environment.

        Args:
            max_num_waypoints: The maximum number of waypoints to reach.
            waypoint_reach_threshold_m: Distance threshold to consider waypoint reached (meters).
            checkpoint_bonus: Initial bonus reward for reaching each waypoint.
            bonus_decay_rate_per_sec: Rate at which checkpoint bonus decays per second after last checkpoint.
            **kwargs: Additional arguments passed to DroneEnv parent class.
        """
        super().__init__(**kwargs)

        self.waypoint_reach_threshold_m = waypoint_reach_threshold_m
        self.checkpoint_bonus = checkpoint_bonus
        self.bonus_decay_rate_per_sec = bonus_decay_rate_per_sec
        self.max_num_waypoints = max_num_waypoints

        # Waypoint tracking
        self.next_waypoint = np.zeros(3, dtype=np.float32)
        self.waypoints_reached = 0
        self.time_since_last_checkpoint = 0

        # Extend observation space to include next waypoint (3 additional floats)
        obs_low = np.concatenate([
            self.observation_space.low,
            np.array([-self.space_side_length * 2] * 3, dtype=np.float32),
            np.array([0], dtype=np.float32),
        ])
        obs_high = np.concatenate([
            self.observation_space.high,
            np.array([self.space_side_length * 2] * 3, dtype=np.float32),
            np.array([np.inf], dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.max_dist_to_target *= 2 # relax out of bounds constraint to allow drone to move more freely

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets environment and generates waypoint sequence."""
        # Initialize tracking
        self.waypoints_reached = 0
        self.time_since_last_checkpoint = 0
        self.next_waypoint = self._generate_random_target()

        # Parent reset
        super().reset(seed=seed, options=options)

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one timestep with waypoint advancement logic."""
        # Call parent step (handles physics, wind, crash detection)
        obs, reward, terminated, truncated, info = super().step(action)

        # Check waypoint reached
        distance_to_waypoint = np.linalg.norm(self.target_position - self.drone.position)
        if distance_to_waypoint <= self.waypoint_reach_threshold_m:
            self.waypoints_reached += 1
            self.time_since_last_checkpoint = 0

            # Advance to next waypoint
            self.target_position = self.next_waypoint
            self.next_waypoint = self._generate_random_target()
        else:
            self.time_since_last_checkpoint += self.dt

        # get observation reward and info
        observation = self._get_observation()
        reward = self._compute_reward()
        info = self._get_info()

        truncated = truncated or self.max_num_waypoints == self.waypoints_reached

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Constructs observation including next waypoint."""
        base_obs = super()._get_observation()
        vect_to_next = self.next_waypoint - self.drone.position
        time_since_last_checkpoint = np.array([self.time_since_last_checkpoint])
        return np.concatenate([base_obs, vect_to_next, time_since_last_checkpoint]).astype(np.float32)

    def _compute_reward(self) -> float:
        """
        Computes reward based on:
        1. Speed in the right direction (velocity alignment)
        2. Decaying checkpoint bonus for reaching waypoints
        """
        distance = np.linalg.norm(self.target_position - self.drone.position)
        if distance < self.waypoint_reach_threshold_m:
            reward = max(self.checkpoint_bonus - self.bonus_decay_rate_per_sec * self.time_since_last_checkpoint, 0.0)
        else:
            direction_target = (self.target_position - self.drone.position) / (distance + 1e-6)
            correct_vel = np.dot(self.drone.velocity, direction_target)
            reward = np.tanh(correct_vel / self.vel_scale)

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Returns extended information about waypoint progress."""
        info = super()._get_info()

        info.update({
            'waypoints_reached': self.waypoints_reached,
            'current_waypoint_pos': self.target_position.copy(),
            'next_waypoint_pos': self.next_waypoint.copy(),
            'steps_since_checkpoint': self.time_since_last_checkpoint,
        })
        return info


class DroneEnvWrapper(abc.ABC, gym.Wrapper):
    def __init__(self, drone_env: DroneEnv):
        gym.Wrapper.__init__(self, drone_env)
        self.action_space = self._get_action_space()

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)

    @abc.abstractmethod
    def _get_action_space(self) -> Space:
        pass

    @abc.abstractmethod
    def _transform_action(self, action: np.ndarray) -> np.ndarray:
        pass

    def step(self, action):
        return self.env.step(self._transform_action(action))


class MotionPrimitiveActionWrapper(DroneEnvWrapper):
    """
    This env wraps the DroneEnv class. Instead of interpreting actions as target motor thrusts, we interpret them as
    targets for movement primitives:
    - hover
    - roll
    - pitch
    - yaw
    """

    def _get_action_space(self) -> Space:
        return self.env.action_space

    def _transform_action(self, action: np.ndarray):
        """
        Maps incoming hover, roll, pitch, yaw commands to single motor thrust commands.
        :param action: an array containing actions for hovering, roll, pitch, and yaw
        :return: an array containing individual motor commands
        """
        hover, roll, pitch, yaw = action
        per_motor_command = np.array([
            hover + roll + pitch + yaw,
            hover - roll - pitch + yaw,
            hover - roll + pitch - yaw,
            hover + roll - pitch - yaw,
        ])
        return per_motor_command


class ThrustChangeController(DroneEnvWrapper):
    """
    This env wraps the DroneEnv class. Instead of interpreting actions as target motor thrusts, we interpret them as
    changes to the existing target motor thrusts:
    """

    def _transform_action(self, action: np.ndarray) -> np.ndarray:
        """
        Maps incoming changes to existing thrusts to absolute thrust commands.
        :param action: an array containing thrust changes
        :return: an array containing absolute thrusts
        """
        # can increase motor command from 0%-100% in 1 second
        return self.env.dt * action + self.env.drone.motor_cmd

    def _get_action_space(self) -> Space:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
