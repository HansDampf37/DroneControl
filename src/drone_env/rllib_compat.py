import gymnasium

from src.drone_env import DroneEnv, SequentialWaypointEnv


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

        # Extract env class
        env_class = config.get("env_class", DroneEnv)

        # Extract SequentialWaypointEnv-specific parameters
        max_num_waypoints = config.get("max_num_waypoints", 15)
        waypoint_reach_threshold_m = config.get("waypoint_reach_threshold", 0.4)
        checkpoint_bonus = config.get("checkpoint_bonus", 10.0)
        bonus_decay_rate_per_sec = config.get("bonus_decay_rate_per_sec", 2.0)

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
        if env_class == SequentialWaypointEnv:
            super().__init__(
                SequentialWaypointEnv(
                    max_num_waypoints=max_num_waypoints,
                    waypoint_reach_threshold_m=waypoint_reach_threshold_m,
                    checkpoint_bonus=checkpoint_bonus,
                    bonus_decay_rate_per_sec=bonus_decay_rate_per_sec,
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
        elif env_class == DroneEnv:
            super().__init__(
                DroneEnv(
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
        else:
            raise NotImplementedError(f"Unknown env class {env_class}")

        wrappers = config.get("wrappers", [])
        for wrapper_class in wrappers:
            self.env = wrapper_class(self.env)
