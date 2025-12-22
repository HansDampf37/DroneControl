import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import matplotlib
matplotlib.use('TkAgg')  # Backend explizit setzen für stabiles Rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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
    - Termination: Nach fixen max_steps
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_steps: int = 1000,
        dt: float = 0.01,  # Zeitschritt in Sekunden (100 Hz)
        target_change_interval: Optional[int] = None,  # Schritte bis Ziel sich ändert
        wind_strength_range: Tuple[float, float] = (0.0, 5.0),  # m/s
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
        self.render_mode = render_mode

        # Crash-Detektion
        self.enable_crash_detection = enable_crash_detection
        self.crash_z_vel_threshold = crash_z_vel_threshold
        self.crash_tilt_threshold = np.deg2rad(crash_tilt_threshold)  # In Radiant konvertieren

        # Drohnen-Parameter (Quadcopter X-Konfiguration)
        self.mass = 1.0  # kg
        self.arm_length = 0.25  # m (Distanz von Center zu Rotor)
        self.inertia = np.array([0.01, 0.01, 0.02])  # kg*m^2 (Ix, Iy, Iz)

        # Pendel-Stabilisierung (Massenschwerpunkt unter Rotoren)
        self.center_of_mass_offset = 0.03  # m unterhalb der Rotoren
        self.pendulum_damping = 0.5  # Dämpfungsfaktor für Pendelbewegung

        # Rotor-Parameter
        self.thrust_coeff = 10.0  # Thrust = thrust_coeff * motor_power
        self.torque_coeff = 0.1  # Torque = torque_coeff * motor_power
        self.gravity = 9.81  # m/s^2

        # Luftwiderstand (Drag)
        self.linear_drag_coeff = 0.01 # Dämpfung für lineare Geschwindigkeit
        self.angular_drag_coeff = 0.05  # Dämpfung für Winkelgeschwindigkeit

        # Rotor-Positionen in X-Konfiguration (relativ zu Drohnen-Center)
        # Motor 0: vorne-rechts (+x, +y), Motor 1: hinten-links (-x, -y)
        # Motor 2: vorne-links (-x, +y), Motor 3: hinten-rechts (+x, -y)
        angle = np.pi / 4  # 45 Grad
        self.rotor_positions = np.array([
            [self.arm_length * np.cos(angle), self.arm_length * np.sin(angle), 0],      # Motor 0
            [-self.arm_length * np.cos(angle), -self.arm_length * np.sin(angle), 0],    # Motor 1
            [-self.arm_length * np.cos(angle), self.arm_length * np.sin(angle), 0],     # Motor 2
            [self.arm_length * np.cos(angle), -self.arm_length * np.sin(angle), 0],     # Motor 3
        ])

        # Rotor-Drehrichtungen (1 = CW, -1 = CCW)
        # In X-Konfiguration: 0 und 1 drehen in eine Richtung, 2 und 3 in die andere
        self.rotor_directions = np.array([1, 1, -1, -1])

        # Wind-Parameter (Ornstein-Uhlenbeck-Prozess)
        self.wind_theta = 0.15  # Mean reversion rate
        self.wind_sigma = 1.0   # Volatility

        # Action Space: 4 Motoren, jeweils 0-1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        space_side_length = 50
        self.max_angular_velocity_component = 10
        self.max_velocity_component = 40
        self.max_wind_velocity_component = 10
        # Observation Space
        # [0:3]   - Position relativ zum Ziel (x, y, z)
        # [3:6]   - Lineare Geschwindigkeit (vx, vy, vz)
        # [6:9]   - Orientierung als Euler-Winkel (roll, pitch, yaw)
        # [9:12]  - Winkelgeschwindigkeit (wx, wy, wz)
        # [12:15] - Windvektor absolut (wx, wy, wz)
        obs_low = np.array(
            [-space_side_length] * 3 +  # Position
            [-space_side_length] * 3 +  # Ziel position
            [-self.max_velocity_component] * 3 +  # Geschwindigkeit
            [-np.pi] * 3 +  # Euler-Winkel
            [-self.max_angular_velocity_component] * 3,  # Winkelgeschwindigkeit
           # [-self.max_wind_velocity] * 3,   # Wind
            dtype=np.float32
        )
        obs_high = np.array(
            [space_side_length] * 3 +
            [space_side_length] * 3 +  # Ziel position
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

        # State-Variablen
        self.position = np.zeros(3, dtype=np.float32)  # Immer [0, 0, 0] in Beobachtung
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(3, dtype=np.float32)  # Roll, Pitch, Yaw
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.target_position = np.zeros(3, dtype=np.float32)
        self.wind_vector = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        # Rendering
        self.fig = None
        self.ax = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset des Environments."""
        super().reset(seed=seed)

        # State zurücksetzen
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.random.uniform(-0.1, 0.1, 3).astype(np.float32)  # Kleine zufällige Startorientierung
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        # Zufälliger Zielpunkt
        self.target_position = self._generate_random_target()

        # Zufälliger initialer Wind
        # wind_strength = np.random.uniform(*self.wind_strength_range)
        # wind_direction = np.random.uniform(0, 2 * np.pi)
        # wind_elevation = np.random.uniform(-np.pi / 6, np.pi / 6)  # Meist horizontal
        # self.wind_vector = np.array([
        #     wind_strength * np.cos(wind_elevation) * np.cos(wind_direction),
        #     wind_strength * np.cos(wind_elevation) * np.sin(wind_direction),
        #     wind_strength * np.sin(wind_elevation)
        # ], dtype=np.float32)

        self.step_count = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ein Zeitschritt der Simulation."""
        action = np.clip(action, 0.0, 1.0)

        # Wind-Update (Ornstein-Uhlenbeck-Prozess)
        # self._update_wind()

        # Zielpunkt-Update (falls konfiguriert)
        if self.target_change_interval is not None:
            if self.step_count > 0 and self.step_count % self.target_change_interval == 0:
                self.target_position = self._generate_random_target()

        # Physik-Berechnung
        self._update_physics(action)

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

    def _update_physics(self, action: np.ndarray):
        """Aktualisiert die Physik der Drohne basierend auf Motor-Actions."""
        # 1. Rotation von Body-Frame zu World-Frame berechnen
        R = self._get_rotation_matrix(self.orientation)

        # 2. Thrust-Richtung im World-Frame (Z-Achse des Body-Frame zeigt nach oben)
        thrust_direction_world = R @ np.array([0, 0, 1], dtype=np.float32)

        # 3. Geschwindigkeit in Thrust-Richtung projizieren
        velocity_in_thrust_direction = np.dot(self.velocity, thrust_direction_world)

        # 4. Thrust-Modifikation basierend auf Geschwindigkeit in Thrust-Richtung
        # Bei negativer Geschwindigkeit (Fallen): Faktor > 1.0 (mehr Thrust)
        # Bei positiver Geschwindigkeit (Steigen): Faktor < 1.0 (weniger Thrust)
        # Physik: Rotor-Effizienz steigt wenn Luft entgegenströmt, sinkt wenn Luft weggedrückt wird
        max_speed_in_thrust_dir = 30.0  # m/s - maximale effektive Geschwindigkeit
        speed_factor = max(0.0, 1.0 - (velocity_in_thrust_direction / max_speed_in_thrust_dir))

        # 5. Thrust-Kräfte berechnen (mit Geschwindigkeits-Dämpfung)
        thrusts = action * self.thrust_coeff * speed_factor

        # 6. Gesamtkraft in Body-Frame (alle Kräfte zeigen nach oben in Z)
        total_thrust_body = np.array([0, 0, np.sum(thrusts)], dtype=np.float32)

        # 7. Rotation zu World-Frame
        total_force_world = R @ total_thrust_body

        # 8. Gravitation hinzufügen
        gravity_force = np.array([0, 0, -self.mass * self.gravity], dtype=np.float32)

        # 9. Luftwiderstand (basierend auf relativer Geschwindigkeit zur Luft/Wind)
        # Relative Geschwindigkeit: v_rel = v_drone - v_wind
        # Wenn die Drohne genau mit dem Wind fliegt, gibt es keinen Widerstand
        relative_velocity = self.velocity - self.wind_vector
        relative_speed = np.linalg.norm(relative_velocity)

        if relative_speed > 0.01:
            # Drag proportional zu v_rel² (quadratisch) in Richtung der Relativgeschwindigkeit
            drag_force = -self.linear_drag_coeff * relative_speed * relative_velocity
        else:
            drag_force = np.zeros(3, dtype=np.float32)

        # 10. Gesamtkraft und lineare Beschleunigung
        total_force = total_force_world + gravity_force + drag_force
        linear_acceleration = total_force / self.mass

        # 11. Drehmoment berechnen (mit geschwindigkeitsangepassten Thrusts)
        torque = self._compute_torque(thrusts)

        # 12. Pendel-Stabilisierung (Massenschwerpunkt unter Rotoren)
        # Rückstellmoment proportional zu Roll/Pitch
        roll, pitch, _ = self.orientation
        pendulum_torque = self.pendulum_damping * np.array([
            -self.mass * self.gravity * self.center_of_mass_offset * np.sin(roll),
            -self.mass * self.gravity * self.center_of_mass_offset * np.sin(pitch),
            0.0  # Kein Yaw-Effekt
        ], dtype=np.float32)

        # 13. Winkelbeschleunigung (mit Pendel-Effekt)
        angular_acceleration = (torque + pendulum_torque) / self.inertia

        # 14. Integration (Euler)
        self.velocity += linear_acceleration * self.dt
        self.velocity = np.clip(self.velocity, -self.max_velocity_component, self.max_velocity_component)
        self.position += self.velocity * self.dt

        self.angular_velocity += angular_acceleration * self.dt
        self.angular_velocity *= (1 - self.angular_drag_coeff)
        self.angular_velocity = np.clip(self.angular_velocity, -self.max_angular_velocity_component, self.max_angular_velocity_component)
        self.orientation += self.angular_velocity * self.dt

        # Normalisiere Euler-Winkel auf [-pi, pi]
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

    def _compute_torque(self, thrusts: np.ndarray) -> np.ndarray:
        """
        Berechnet das Drehmoment basierend auf Rotor-Thrusts.

        In X-Konfiguration:
        - Roll wird durch Thrust-Differenz zwischen linken und rechten Motoren erzeugt
        - Pitch durch Differenz zwischen vorderen und hinteren Motoren
        - Yaw durch Differenz in Drehrichtungen (reaktives Torque)
        """
        # Roll-Torque (um X-Achse): Rechte Motoren (0, 3) vs. Linke Motoren (1, 2)
        roll_torque = (thrusts[0] + thrusts[3] - thrusts[1] - thrusts[2]) * self.arm_length / np.sqrt(2)

        # Pitch-Torque (um Y-Achse): Vordere Motoren (0, 2) vs. Hintere Motoren (1, 3)
        pitch_torque = (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3]) * self.arm_length / np.sqrt(2)

        # Yaw-Torque (um Z-Achse): Reaktive Torques basierend auf Drehrichtung
        yaw_torque = np.sum(self.rotor_directions * thrusts) * self.torque_coeff

        return np.array([roll_torque, pitch_torque, yaw_torque], dtype=np.float32)

    def _get_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        Rotation Matrix von Body-Frame zu World-Frame.
        Euler-Winkel: [roll, pitch, yaw] (ZYX-Konvention)
        """
        roll, pitch, yaw = euler

        # Roll (X-Achse)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # Pitch (Y-Achse)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Yaw (Z-Achse)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Kombinierte Rotation: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx

    def _update_wind(self):
        """Aktualisiert den Windvektor mit Ornstein-Uhlenbeck-Prozess."""
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
        observation = np.concatenate([
            self.position,             # [0:3]
            self.target_position,      # [3:6]
            self.velocity,             # [6:9]
            self.orientation,          # [9:12]
            self.angular_velocity,     # [12:15]
            #self.wind_vector,          # [15:18]
        ]).astype(np.float32)

        return observation

    def _compute_reward(self) -> float:
        """Berechnet den Dense Reward: 1/(1 + distance_to_target)."""
        distance = np.linalg.norm(self.target_position - self.position)
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

        # Primär: Fällt zu schnell
        if self.velocity[2] < self.crash_z_vel_threshold:
            return True

        # Sekundär: Extreme Neigung (komplett außer Kontrolle)
        roll, pitch, _ = self.orientation
        if abs(roll) > self.crash_tilt_threshold or abs(pitch) > self.crash_tilt_threshold:
            return True

        # Tertiär: Entfernung vom Ziel zu hoch
        distance = np.linalg.norm(self.target_position - self.position)
        if distance > self.max_dist_to_target:
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Zusätzliche Informationen."""
        distance = np.linalg.norm(self.target_position - self.position)
        return {
            "distance_to_target": float(distance),
            "position": self.position.copy(),
            "target_position": self.target_position.copy(),
            "step_count": self.step_count,
        }

    def render(self):
        """Visualisierung (2D Top-Down-View und Front-View wie technische Zeichnung)."""
        if self.render_mode is None:
            return

        # Erstelle Figure beim ersten Aufruf
        if self.fig is None:
            # Für human mode: interaktiver Modus
            if self.render_mode == "human":
                plt.ion()

            # 2 Subplots: Oben Draufsicht (XY), Unten Vorderansicht (XZ)
            self.fig, (self.ax_top, self.ax_front) = plt.subplots(2, 1, figsize=(10, 14))
            self.fig.set_facecolor('white')
            self.fig.subplots_adjust(hspace=0.3)
        else:
            # Axes existieren bereits, weise sie zu falls nötig
            self.ax_top = self.fig.axes[0]
            self.ax_front = self.fig.axes[1]

        # Clear beide Ansichten
        self.ax_top.clear()
        self.ax_front.clear()
        self.ax_top.set_facecolor('#f0f0f0')
        self.ax_front.set_facecolor('#f0f0f0')

        # ========== TOP VIEW (Draufsicht XY) ==========
        self.ax_top.set_xlim(-30, 30)
        self.ax_top.set_ylim(-30, 30)
        self.ax_top.set_aspect('equal')
        self.ax_top.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        self.ax_top.set_xlabel('X (m)', fontsize=11)
        self.ax_top.set_ylabel('Y (m)', fontsize=11)
        self.ax_top.set_title(f'Draufsicht (Top View) - Step: {self.step_count}', fontsize=12, fontweight='bold')

        # ========== FRONT VIEW (Vorderansicht XZ) ==========
        self.ax_front.set_xlim(-30, 30)
        self.ax_front.set_ylim(-30, 30)
        self.ax_front.set_aspect('equal')
        self.ax_front.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        self.ax_front.set_xlabel('X (m)', fontsize=11)
        self.ax_front.set_ylabel('Z (Höhe) (m)', fontsize=11)
        self.ax_front.set_title('Vorderansicht (Front View)', fontsize=12, fontweight='bold')

        # Rotations-Matrix für aktuelle Orientierung
        R = self._get_rotation_matrix(self.orientation)

        # --- TOP VIEW: Drohne (XY-Ebene) ---
        drone_circle_top = Circle(
            (self.position[0], self.position[1]),
            0.3,
            color='#0066cc',
            alpha=0.9,
            zorder=5,
            label='Drohne'
        )
        self.ax_top.add_patch(drone_circle_top)

        # --- FRONT VIEW: Drohne (XZ-Ebene) ---
        drone_circle_front = Circle(
            (self.position[0], self.position[2]),
            0.3,
            color='#0066cc',
            alpha=0.9,
            zorder=5,
            label='Drohne'
        )
        self.ax_front.add_patch(drone_circle_front)

        # ========== ROTOREN ZEICHNEN ==========
        rotor_colors = ['#ff6666', '#ff6666', '#66ff66', '#66ff66']  # Rot: CW, Grün: CCW
        rotor_scale = 3.0  # Skalierung für bessere Visualisierung

        for i, (rotor_pos_body, color) in enumerate(zip(self.rotor_positions, rotor_colors)):
            # Transformiere Rotor-Position von Body-Frame zu World-Frame
            rotor_pos_world = R @ rotor_pos_body
            rotor_pos_world_scaled = rotor_pos_world * rotor_scale

            # World-Koordinaten der Rotoren
            rotor_x = self.position[0] + rotor_pos_world_scaled[0]
            rotor_y = self.position[1] + rotor_pos_world_scaled[1]
            rotor_z = self.position[2] + rotor_pos_world_scaled[2]

            # --- TOP VIEW: Rotoren in XY-Ebene ---
            self.ax_top.plot(
                [self.position[0], rotor_x],
                [self.position[1], rotor_y],
                color='#666666',
                linewidth=2.5,
                zorder=4,
                alpha=0.8
            )
            rotor_circle_top = Circle(
                (rotor_x, rotor_y),
                0.15,
                color=color,
                alpha=0.8,
                zorder=6
            )
            self.ax_top.add_patch(rotor_circle_top)

            # --- FRONT VIEW: Rotoren in XZ-Ebene ---
            self.ax_front.plot(
                [self.position[0], rotor_x],
                [self.position[2], rotor_z],
                color='#666666',
                linewidth=2.5,
                zorder=4,
                alpha=0.8
            )
            rotor_circle_front = Circle(
                (rotor_x, rotor_z),
                0.15,
                color=color,
                alpha=0.8,
                zorder=6
            )
            self.ax_front.add_patch(rotor_circle_front)

        # ========== NEIGUNG/ORIENTIERUNG ==========
        # Normale der Drohne (zeigt nach "oben" im Body-Frame)
        normal_body = np.array([0, 0, 1])
        normal_world = R @ normal_body

        # --- TOP VIEW: Projektion der Neigung auf XY-Ebene ---
        tilt_x = normal_world[0]
        tilt_y = normal_world[1]
        tilt_magnitude = np.sqrt(tilt_x**2 + tilt_y**2)

        if tilt_magnitude > 0.01:
            tilt_scale = 1.5
            self.ax_top.arrow(
                self.position[0], self.position[1],
                tilt_x * tilt_scale, tilt_y * tilt_scale,
                head_width=0.3,
                head_length=0.25,
                fc='#ff9900',
                ec='#ff9900',
                linewidth=2.5,
                zorder=7,
                alpha=0.9,
                label='Neigung'
            )

        # --- FRONT VIEW: Neigung in XZ-Ebene ---
        tilt_x_front = normal_world[0]
        tilt_z_front = normal_world[2]

        if abs(tilt_x_front) > 0.01 or abs(tilt_z_front - 1.0) > 0.01:
            tilt_scale_front = 1.5
            self.ax_front.arrow(
                self.position[0], self.position[2],
                tilt_x_front * tilt_scale_front, tilt_z_front * tilt_scale_front,
                head_width=0.3,
                head_length=0.25,
                fc='#ff9900',
                ec='#ff9900',
                linewidth=2.5,
                zorder=7,
                alpha=0.9,
                label='Neigung'
            )

        # ========== ZIELPUNKT ==========
        # --- TOP VIEW: Ziel in XY-Ebene ---
        target_circle_top = Circle(
            (self.target_position[0], self.target_position[1]),
            1.0,
            color='#00cc00',
            alpha=0.6,
            zorder=4,
            label='Ziel'
        )
        self.ax_top.add_patch(target_circle_top)

        # Ziel-Kreuz (Top View)
        cross_size = 0.5
        self.ax_top.plot(
            [self.target_position[0] - cross_size, self.target_position[0] + cross_size],
            [self.target_position[1], self.target_position[1]],
            'g-', linewidth=2, zorder=5
        )
        self.ax_top.plot(
            [self.target_position[0], self.target_position[0]],
            [self.target_position[1] - cross_size, self.target_position[1] + cross_size],
            'g-', linewidth=2, zorder=5
        )

        # Verbindungslinie zur Drohne (Top View)
        self.ax_top.plot(
            [self.position[0], self.target_position[0]],
            [self.position[1], self.target_position[1]],
            'k--',
            alpha=0.4,
            linewidth=1.5,
            zorder=1
        )

        # --- FRONT VIEW: Ziel in XZ-Ebene ---
        target_circle_front = Circle(
            (self.target_position[0], self.target_position[2]),
            1.0,
            color='#00cc00',
            alpha=0.6,
            zorder=4,
            label='Ziel'
        )
        self.ax_front.add_patch(target_circle_front)

        # Ziel-Kreuz (Front View)
        self.ax_front.plot(
            [self.target_position[0] - cross_size, self.target_position[0] + cross_size],
            [self.target_position[2], self.target_position[2]],
            'g-', linewidth=2, zorder=5
        )
        self.ax_front.plot(
            [self.target_position[0], self.target_position[0]],
            [self.target_position[2] - cross_size, self.target_position[2] + cross_size],
            'g-', linewidth=2, zorder=5
        )

        # Verbindungslinie zur Drohne (Front View)
        self.ax_front.plot(
            [self.position[0], self.target_position[0]],
            [self.position[2], self.target_position[2]],
            'k--',
            alpha=0.4,
            linewidth=1.5,
            zorder=1
        )

        # ========== WIND-VEKTOR (nur Top View) ==========
        wind_scale = 3.0
        wind_x = self.wind_vector[0] * wind_scale
        wind_y = self.wind_vector[1] * wind_scale
        if np.linalg.norm([wind_x, wind_y]) > 0.1:  # Nur zeichnen wenn Wind spürbar
            self.ax_top.arrow(
                -25, 25,
                wind_x, wind_y,
                head_width=0.7,
                head_length=0.7,
                fc='#cc0000',
                ec='#cc0000',
                linewidth=2,
                alpha=0.8,
                zorder=3,
                label='Wind'
            )

        # ========== INFO-BOX ==========
        distance = np.linalg.norm(self.target_position - self.position)
        velocity_mag = np.linalg.norm(self.velocity)
        wind_mag = np.linalg.norm(self.wind_vector)

        # Konvertiere Winkel zu Grad
        roll_deg = np.rad2deg(self.orientation[0])
        pitch_deg = np.rad2deg(self.orientation[1])
        yaw_deg = np.rad2deg(self.orientation[2])

        info_text = f'Step: {self.step_count}\n'
        info_text += f'Distanz: {distance:.2f}m\n'
        info_text += f'Höhe: {self.position[2]:.2f}m\n'
        info_text += f'Geschw.: {velocity_mag:.2f}m/s\n'
        info_text += f'Wind: {wind_mag:.2f}m/s\n'
        info_text += f'Roll: {roll_deg:.1f}°\n'
        info_text += f'Pitch: {pitch_deg:.1f}°\n'
        info_text += f'Yaw: {yaw_deg:.1f}°\n'
        info_text += f'Reward: {self._compute_reward():.4f}'

        self.ax_top.text(
            0.02, 0.98,
            info_text,
            transform=self.ax_top.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            family='monospace'
        )

        # Legenden
        self.ax_top.legend(loc='upper right', fontsize=9)
        self.ax_front.legend(loc='upper right', fontsize=9)

        # Rendering durchführen
        if self.render_mode == "human":
            # Zeichne alles und zeige es an
            plt.draw()
            plt.pause(0.01)  # Pause für GUI-Update
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            width, height = self.fig.canvas.get_width_height()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            image = image.reshape((height, width, 3))
            return image

    def close(self):
        """Cleanup."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax_top = None
            self.ax_front = None
        if self.render_mode == "human":
            plt.ioff()  # Interaktiven Modus beenden

