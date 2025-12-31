"""Drone RL Environment Package."""
import gymnasium

from .drone import Drone
from .env import DroneEnv, SequentialWaypointEnv, MotionPrimitiveActionWrapper, ThrustChangeController
from .renderer import DroneEnvRenderer


class RLlibDroneEnv(gymnasium.Wrapper):
    """
    RLlib-compatible wrapper for DroneEnv.

    RLlib calls environments with EnvClass(env_config), but DroneEnv
    expects named parameters (**kwargs). This wrapper translates
    the env_config dictionary into kwargs.
    """

    def __init__(self, config=None):
        """
        Initializes the environment with RLlib's env_config.

        Args:
            config: Dictionary with environment configuration from RLlib.
                If None, default configuration is used.
        """
        if config is None:
            config = {}

        # Extract parameters from config or use defaults
        max_steps = config.get("max_steps", 1000)
        dt = config.get("dt", 0.01)
        target_change_interval = config.get("target_change_interval", None)
        wind_strength_range = config.get("wind_strength_range", (0.0, 5.0))
        use_wind = config.get("use_wind", True)
        render_mode = config.get("render_mode", None)
        enable_crash_detection = config.get("enable_crash_detection", True)
        crash_z_vel_threshold = config.get("crash_z_vel_threshold", -20.0)
        crash_tilt_threshold = config.get("crash_tilt_threshold", 80.0)

        # Call parent constructor with named parameters
        super().__init__(
            SequentialWaypointEnv(
                waypoint_reach_threshold=0.3,
                max_steps=max_steps,
                dt=dt,
                target_change_interval=target_change_interval,
                wind_strength_range=wind_strength_range,
                use_wind=use_wind,
                render_mode=render_mode,
                enable_crash_detection=enable_crash_detection,
                crash_z_vel_threshold=crash_z_vel_threshold,
                crash_tilt_threshold=crash_tilt_threshold,
            )
        )
        wrappers = config.get("wrappers", [])
        for wrapper_class in wrappers:
            self.env = wrapper_class(self.env)


class RLlibSequentialWaypointEnv(gymnasium.Wrapper):
    """
    RLlib-compatible wrapper for SequentialWaypointEnv.

    RLlib calls environments with EnvClass(env_config), but SequentialWaypointEnv
    expects named parameters (**kwargs). This wrapper translates
    the env_config dictionary into kwargs.
    """

    def __init__(self, config=None):
        """
        Initializes the environment with RLlib's env_config.

        Args:
            config: Dictionary with environment configuration from RLlib.
                If None, default configuration is used.
        """
        if config is None:
            config = {}

        # Extract SequentialWaypointEnv-specific parameters
        waypoint_reach_threshold = config.get("waypoint_reach_threshold", 0.4)
        waypoint_spacing_range = config.get("waypoint_spacing_range", (2.0, 4.0))
        checkpoint_bonus = config.get("checkpoint_bonus", 10.0)
        bonus_decay_rate_per_sec = config.get("bonus_decay_rate_per_sec", 2.0)
        speed_reward_weight = config.get("speed_reward_weight", 0.5)

        # Extract base DroneEnv parameters
        max_steps = config.get("max_steps", 1000)
        dt = config.get("dt", 0.01)
        target_change_interval = config.get("target_change_interval", None)
        wind_strength_range = config.get("wind_strength_range", (0.0, 5.0))
        use_wind = config.get("use_wind", True)
        render_mode = config.get("render_mode", None)
        enable_crash_detection = config.get("enable_crash_detection", True)
        enable_out_of_bounds_detection = config.get("enable_out_of_bounds_detection", True)
        crash_z_vel_threshold = config.get("crash_z_vel_threshold", -20.0)
        crash_tilt_threshold = config.get("crash_tilt_threshold", 80.0)

        # Call parent constructor with named parameters
        super().__init__(
            SequentialWaypointEnv(
                waypoint_reach_threshold=waypoint_reach_threshold,
                waypoint_spacing_range=waypoint_spacing_range,
                checkpoint_bonus=checkpoint_bonus,
                bonus_decay_rate_per_sec=bonus_decay_rate_per_sec,
                speed_reward_weight=speed_reward_weight,
                max_steps=max_steps,
                dt=dt,
                target_change_interval=target_change_interval,
                wind_strength_range=wind_strength_range,
                use_wind=use_wind,
                render_mode=render_mode,
                enable_crash_detection=enable_crash_detection,
                enable_out_of_bounds_detection=enable_out_of_bounds_detection,
                crash_z_vel_threshold=crash_z_vel_threshold,
                crash_tilt_threshold=crash_tilt_threshold,
            )
        )
        wrappers = config.get("wrappers", [])
        for wrapper_class in wrappers:
            self.env = wrapper_class(self.env)


__all__ = ['DroneEnv', 'SequentialWaypointEnv', 'RLlibDroneEnv', 'RLlibSequentialWaypointEnv', 'DroneEnvRenderer', 'Drone']
__version__ = '0.1.0'

