"""
Wind simulation using Ornstein-Uhlenbeck process.
"""
import numpy as np
from typing import Tuple


class Wind:
    """
    Wind simulation using an Ornstein-Uhlenbeck process.

    The Ornstein-Uhlenbeck process creates realistic, slowly-varying wind that
    tends to revert to zero over time (mean-reverting stochastic process).
    This produces more realistic wind behavior than simple random noise.
    """

    def __init__(
        self,
        strength_range: Tuple[float, float] = (0.0, 5.0),
        theta: float = 0.5,
        sigma: float = 1.0,
        enabled: bool = False
    ):
        """
        Initializes the wind simulation.

        Args:
            strength_range: Tuple of (min, max) wind speeds in m/s.
                Wind speed is clamped to this range. Default is (0.0, 5.0).
            theta: Mean reversion rate (how quickly wind returns to zero).
                Higher values = faster return to calm. Default is 0.5.
            sigma: Volatility (how much random variation occurs).
                Higher values = more turbulent wind. Default is 1.0.
            enabled: Whether wind simulation is active. If False, wind is always zero.
                Default is False.
        """
        self.strength_range = strength_range
        self.theta = theta
        self.sigma = sigma
        self.enabled = enabled

        # Current wind velocity vector [wx, wy, wz] in m/s
        self.vector = np.zeros(3, dtype=np.float32)

    def reset(self):
        """Resets wind to zero (calm conditions)."""
        self.vector = np.zeros(3, dtype=np.float32)

    def update(self, dt: float):
        """
        Updates the wind vector using the Ornstein-Uhlenbeck process.

        The update equation is:
            dx = theta * (0 - x) * dt + sigma * dW
        where:
            - theta: mean reversion rate
            - sigma: volatility
            - dW: Wiener process (Brownian motion) increment

        Args:
            dt: Time step in seconds. Smaller values provide smoother wind evolution.
        """
        # Only update wind if enabled
        if not self.enabled:
            self.vector = np.zeros(3, dtype=np.float32)
            return

        # Ornstein-Uhlenbeck process
        # Drift term: pulls wind back toward zero
        drift = -self.theta * self.vector * dt

        # Diffusion term: random perturbation
        diffusion = self.sigma * np.random.normal([0] * 3, [np.sqrt(dt), np.sqrt(dt), np.sqrt(dt) / 5])

        # Update wind vector
        self.vector += drift + diffusion

        # Clamp wind speed to maximum allowed
        wind_speed = np.linalg.norm(self.vector)
        max_wind = self.strength_range[1]
        if wind_speed > max_wind:
            self.vector = self.vector / wind_speed * max_wind

    def get_vector(self) -> np.ndarray:
        """
        Returns the current wind velocity vector.

        Returns:
            Wind vector [wx, wy, wz] in m/s as a float32 array.
        """
        return self.vector.copy()

    def set_enabled(self, enabled: bool):
        """
        Enables or disables wind simulation.

        Args:
            enabled: If True, wind simulation is active. If False, wind is zero.
        """
        self.enabled = enabled
        if not enabled:
            self.vector = np.zeros(3, dtype=np.float32)

    def __repr__(self) -> str:
        """String representation of the wind state."""
        status = "enabled" if self.enabled else "disabled"
        speed = np.linalg.norm(self.vector)
        return (f"Wind(status={status}, speed={speed:.2f} m/s, "
                f"vector=[{self.vector[0]:.2f}, {self.vector[1]:.2f}, {self.vector[2]:.2f}])")

