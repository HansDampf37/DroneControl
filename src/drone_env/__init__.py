"""Drohnen-RL Environment Package."""
from .env import DroneEnv


class RLlibDroneEnv(DroneEnv):
    """
    RLlib-kompatibler Wrapper für DroneEnv.

    RLlib ruft Environments mit EnvClass(env_config) auf,
    aber DroneEnv erwartet benannte Parameter (**kwargs).
    Dieser Wrapper übersetzt das env_config Dictionary in kwargs.
    """

    def __init__(self, config=None):
        """
        Initialisiert das Environment mit RLlib's env_config.

        Args:
            config: Dictionary mit Environment-Konfiguration von RLlib
        """
        if config is None:
            config = {}

        # Extrahiere Parameter aus config oder verwende Defaults
        max_steps = config.get("max_steps", 1000)
        dt = config.get("dt", 0.01)
        target_change_interval = config.get("target_change_interval", None)
        wind_strength_range = config.get("wind_strength_range", (0.0, 5.0))
        render_mode = config.get("render_mode", None)
        enable_crash_detection = config.get("enable_crash_detection", True)
        crash_z_threshold = config.get("crash_z_threshold", -5.0)
        crash_tilt_threshold = config.get("crash_tilt_threshold", 80.0)

        # Rufe Parent-Konstruktor mit benannten Parametern auf
        super().__init__(
            max_steps=max_steps,
            dt=dt,
            target_change_interval=target_change_interval,
            wind_strength_range=wind_strength_range,
            render_mode=render_mode,
            enable_crash_detection=enable_crash_detection,
            crash_z_threshold=crash_z_threshold,
            crash_tilt_threshold=crash_tilt_threshold,
        )


__all__ = ['DroneEnv', 'RLlibDroneEnv']
__version__ = '0.1.0'

