import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from .renderer import DroneEnvRenderer
from .drone import Drone


class DroneEnv(gym.Env):
    """
    Gymnasium Environment für Quadcopter-Steuerung mit RL.

    Die Drohne (X-Konfiguration) soll zu einem Zielpunkt fliegen und dort bleiben.
    - Action Space: 4 Motoren, jeweils 0-1 (0-100% Thrust)
    - Observation Space: Relative Position zum Ziel, Geschwindigkeit, Orientierung (Euler),
                         Winkelgeschwindigkeit, Windvektor
    - Physik: Vereinfacht, Kräfte senkrecht zur Rotorebene, skalieren mit Motor-Power
    - Wind: Dynamisch, ändert sich über Zeit (Ornstein-Uhlenbeck-Prozess)
    - Reward: Dense, ((max_distance-distance)/max-distance) ** 2
    - Termination: Nach fixen max_steps oder wenn drone crasht
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_steps: int = 1000,
        dt: float = 0.01,  # Zeitschritt in Sekunden (100 Hz)
        target_change_interval: Optional[int] = None,  # Schritte bis Ziel sich ändert
        wind_strength_range: Tuple[float, float] = (0.0, 5.0),  # m/s
        use_wind: bool = False,  # Wind-Simulation aktivieren
        render_mode: Optional[str] = None,
        # Crash-Detektion Parameter
        enable_crash_detection: bool = True,
        crash_z_vel_threshold: float = -20.0,  # Drohne fällt schneller als 10 m/s = Crash
        crash_tilt_threshold: float = 80.0,  # Roll/Pitch über diesem Winkel = Crash (Grad)
    ):
        super().__init__()

        self.max_steps = max_steps
        self.dt = dt
        self.target_change_interval = target_change_interval
        self.wind_strength_range = wind_strength_range
        self.use_wind = use_wind
        self.render_mode = render_mode

        # Crash-Detektion
        self.enable_crash_detection = enable_crash_detection
        self.crash_z_vel_threshold = crash_z_vel_threshold
        self.crash_tilt_threshold = np.deg2rad(crash_tilt_threshold)  # In Radiant konvertieren

        # Simulations-Parameter
        self.gravity = 9.81  # m/s^2
        self.max_velocity_component = 40.0  # m/s
        self.max_angular_velocity_component = 10.0  # rad/s

        # Drohne erstellen (nur intrinsische Eigenschaften)
        self.drone = Drone(
            mass=1.0,
            arm_length=0.25,
            inertia=np.array([0.01, 0.01, 0.02]),
            thrust_coeff=10.0,
            torque_coeff=0.1,
            linear_drag_coeff=0.01,
            angular_drag_coeff=0.05,
            center_of_mass_offset=0.03,
            pendulum_damping=0.5,
        )

        # Wind-Parameter (Ornstein-Uhlenbeck-Prozess)
        self.wind_theta = 0.15  # Mean reversion rate
        self.wind_sigma = 1.0   # Volatility
        self.wind_vector = np.zeros(3, dtype=np.float32)

        # Action Space: 4 Motoren, jeweils 0-1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        space_side_length = 50
        # Observation Space
        # [0:3]   - Position relativ zum Ziel (x, y, z)
        # [3:6]   - Lineare Geschwindigkeit (vx, vy, vz)
        # [6:9]   - Orientierung als Euler-Winkel (roll, pitch, yaw)
        # [9:12]  - Winkelgeschwindigkeit (wx, wy, wz)
        # [12:15] - Windvektor absolut (wx, wy, wz)
        obs_low = np.array(
            [-space_side_length] * 3 +  # relative Position
            [-self.max_velocity_component] * 3 +  # Geschwindigkeit
            [-np.pi] * 3 +  # Euler-Winkel
            [-self.max_angular_velocity_component] * 3,  # Winkelgeschwindigkeit
           # [-self.max_wind_velocity] * 3,   # Wind
            dtype=np.float32
        )
        obs_high = np.array(
            [space_side_length] * 3 + # relative position
            [self.max_velocity_component] * 3 +  # Geschwindigkeit
            [np.pi] * 3 +  # Euler-Winkel
            [self.max_angular_velocity_component] * 3,  # Winkelgeschwindigkeit
            # [-self.max_wind_velocity] * 3,   # Wind
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.max_dist_to_target = np.sqrt(3 * space_side_length ** 2)

        # Environment State
        self.target_position = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        # Rendering
        self.renderer = DroneEnvRenderer(render_mode=render_mode)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset des Environments."""
        super().reset(seed=seed)

        # Drohne zurücksetzen
        initial_orientation = np.random.uniform(-0.1, 0.1, 3).astype(np.float32)
        self.drone.reset(initial_orientation=initial_orientation)

        # Zufälliger Zielpunkt
        self.target_position = self._generate_random_target()

        # Wind zurücksetzen (optional aktiviert)
        self.wind_vector = np.zeros(3, dtype=np.float32)

        self.step_count = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ein Zeitschritt der Simulation."""
        action = np.clip(action, 0.0, 1.0)

        # Wind-Update (optional)
        self._update_wind()

        # Zielpunkt-Update (falls konfiguriert)
        if self.target_change_interval is not None:
            if self.step_count > 0 and self.step_count % self.target_change_interval == 0:
                self.target_position = self._generate_random_target()

        # Physik-Update über Drone-Klasse mit Simulations-Parametern
        self.drone.update(
            motor_thrusts=action,
            dt=self.dt,
            wind_vector=self.wind_vector,
            gravity=self.gravity,
            max_velocity=self.max_velocity_component,
            max_angular_velocity=self.max_angular_velocity_component,
        )

        # Observation und Reward
        observation = self._get_observation()
        reward = self._compute_reward()

        # Termination
        self.step_count += 1
        crashed = self._check_crash()
        terminated = crashed  # Episode endet bei Crash
        truncated = self.step_count >= self.max_steps

        info = self._get_info()
        info['crashed'] = crashed  # Füge Crash-Info hinzu

        return observation, reward, terminated, truncated, info


    def _update_wind(self):
        """Aktualisiert den Windvektor mit Ornstein-Uhlenbeck-Prozess."""
        # Wind nur updaten wenn aktiviert
        if not self.use_wind:
            self.wind_vector = np.zeros(3, dtype=np.float32)
            return

        # Ornstein-Uhlenbeck: dx = theta * (0 - x) * dt + sigma * dW
        drift = -self.wind_theta * self.wind_vector * self.dt
        diffusion = self.wind_sigma * np.random.normal(0, np.sqrt(self.dt), 3)
        self.wind_vector += drift + diffusion

        # Begrenze Windstärke
        wind_speed = np.linalg.norm(self.wind_vector)
        max_wind = self.wind_strength_range[1]
        if wind_speed > max_wind:
            self.wind_vector = self.wind_vector / wind_speed * max_wind

    def _generate_random_target(self) -> np.ndarray:
        """Generiert einen zufälligen Zielpunkt."""
        # Zufällige Position in sphärischen Koordinaten
        distance = np.random.uniform(5.0, self.max_dist_to_target)
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(-np.pi / 4, np.pi / 4)

        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)

        return np.array([x, y, z], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        vect_to_target = self.target_position - self.drone.position
        observation = np.concatenate([
            vect_to_target,                # [0:3]
            self.drone.velocity,           # [3:6]
            self.drone.orientation,        # [6:9]
            self.drone.angular_velocity,   # [9:12]
            #self.wind_vector,             # [12:15]
        ]).astype(np.float32)

        return observation

    def _compute_reward(self) -> float:
        """Berechnet den Dense Reward: 1/(1 + distance_to_target)."""
        distance = np.linalg.norm(self.target_position - self.drone.position)
        distance = np.clip(distance, 0, self.max_dist_to_target)
        margin = (self.max_dist_to_target - distance) / self.max_dist_to_target
        return float(margin ** 2)

    def _check_crash(self) -> bool:
        """
        Prüft ob die Drohne abgestürzt ist.

        Verwendet zwei Kriterien:
        1. Low z-coordinate (primär, am effizientesten)
        2. Extreme tilt (sekundär, für unkontrollierte Drohne)

        Returns:
            bool: True wenn Crash detektiert, sonst False
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
        """Zusätzliche Informationen."""
        distance = np.linalg.norm(self.target_position - self.drone.position)
        return {
            "distance_to_target": float(distance),
            "position": self.drone.position.copy(),
            "target_position": self.target_position.copy(),
            "step_count": self.step_count,
        }

    def render(self):
        """Visualisierung (2D Top-Down-View und Front-View wie technische Zeichnung)."""
        if self.render_mode is None:
            return None

        # Hole Rotationsmatrix von Drone
        R = self.drone.get_rotation_matrix()

        # Delegiere an Renderer
        return self.renderer.render(
            position=self.drone.position,
            velocity=self.drone.velocity,
            orientation=self.drone.orientation,
            angular_velocity=self.drone.angular_velocity,
            target_position=self.target_position,
            wind_vector=self.wind_vector,
            rotation_matrix=R,
            rotor_positions=self.drone.rotor_positions,
            step_count=self.step_count,
            reward=self._compute_reward()
        )

    def close(self):
        """Cleanup."""
        self.renderer.close()

