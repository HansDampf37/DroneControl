"""Drone RL Environment Package."""
from .env import DroneEnv
from .renderer import DroneEnvRenderer
from .drone import Drone


class RLlibDroneEnv(DroneEnv):
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


__all__ = ['DroneEnv', 'RLlibDroneEnv', 'DroneEnvRenderer', 'Drone']
__version__ = '0.1.0'

