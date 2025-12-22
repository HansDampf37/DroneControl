"""
Drohnen-Modell für die Simulation.
Kapselt die Physik und den Zustand einer Quadcopter-Drohne.
"""
import numpy as np
from typing import Tuple


class Drone:
    """
    Quadcopter-Drohne in X-Konfiguration.

    Verwaltet:
    - Physikalische Parameter (Masse, Trägheit, etc.)
    - Zustand (Position, Geschwindigkeit, Orientierung, etc.)
    - Physik-Simulation (Kräfte, Drehmomente, Integration)
    """

    def __init__(
        self,
        mass: float = 1.0,
        arm_length: float = 0.25,
        inertia: np.ndarray = None,
        thrust_coeff: float = 10.0,
        torque_coeff: float = 0.1,
        linear_drag_coeff: float = 0.01,
        angular_drag_coeff: float = 0.05,
        center_of_mass_offset: float = 0.03,
        pendulum_damping: float = 0.5,
    ):
        """
        Initialisiert die Drohne mit ihren physikalischen Eigenschaften.

        Args:
            mass: Masse der Drohne in kg
            arm_length: Distanz von Center zu Rotor in m
            inertia: Trägheitsmoment [Ix, Iy, Iz] in kg*m^2
            thrust_coeff: Thrust-Koeffizient
            torque_coeff: Torque-Koeffizient
            linear_drag_coeff: Luftwiderstand linear
            angular_drag_coeff: Luftwiderstand rotational
            center_of_mass_offset: Massenschwerpunkt-Offset in m
            pendulum_damping: Pendel-Dämpfungsfaktor
        """
        # Intrinsische physikalische Parameter der Drohne
        self.mass = mass
        self.arm_length = arm_length
        self.inertia = inertia if inertia is not None else np.array([0.01, 0.01, 0.02])
        self.thrust_coeff = thrust_coeff
        self.torque_coeff = torque_coeff
        self.linear_drag_coeff = linear_drag_coeff
        self.angular_drag_coeff = angular_drag_coeff
        self.center_of_mass_offset = center_of_mass_offset
        self.pendulum_damping = pendulum_damping

        # Rotor-Konfiguration (X-Formation)
        # Motor 0: vorne-rechts (+x, +y), Motor 1: hinten-links (-x, -y)
        # Motor 2: vorne-links (-x, +y), Motor 3: hinten-rechts (+x, -y)
        angle = np.pi / 4  # 45 Grad
        self.rotor_positions = np.array([
            [self.arm_length * np.cos(angle), self.arm_length * np.sin(angle), 0],      # Motor 0
            [-self.arm_length * np.cos(angle), -self.arm_length * np.sin(angle), 0],    # Motor 1
            [-self.arm_length * np.cos(angle), self.arm_length * np.sin(angle), 0],     # Motor 2
            [self.arm_length * np.cos(angle), -self.arm_length * np.sin(angle), 0],     # Motor 3
        ], dtype=np.float32)

        # Rotor-Drehrichtungen (1 = CW, -1 = CCW)
        self.rotor_directions = np.array([1, 1, -1, -1])

        # Zustand
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(3, dtype=np.float32)  # Roll, Pitch, Yaw
        self.angular_velocity = np.zeros(3, dtype=np.float32)

    def reset(self, initial_orientation: np.ndarray = None):
        """
        Setzt die Drohne zurück auf Initialzustand.

        Args:
            initial_orientation: Optionale Start-Orientierung [roll, pitch, yaw]
        """
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)

        if initial_orientation is not None:
            self.orientation = initial_orientation.astype(np.float32)
        else:
            # Kleine zufällige Startorientierung
            self.orientation = np.random.uniform(-0.1, 0.1, 3).astype(np.float32)

        self.angular_velocity = np.zeros(3, dtype=np.float32)

    def update(
        self,
        motor_thrusts: np.ndarray,
        dt: float,
        wind_vector: np.ndarray = None,
        gravity: float = 9.81,
        max_velocity: float = 40.0,
        max_angular_velocity: float = 10.0,
    ):
        """
        Aktualisiert die Drohnen-Physik für einen Zeitschritt.

        Args:
            motor_thrusts: Array mit 4 normalisierten Thrust-Werten [0, 1]
            dt: Zeitschritt in Sekunden
            wind_vector: Optionaler Windvektor [wx, wy, wz] in m/s
            gravity: Gravitationskonstante in m/s^2
            max_velocity: Maximale Geschwindigkeit in m/s (Clipping)
            max_angular_velocity: Maximale Winkelgeschwindigkeit in rad/s (Clipping)
        """
        if wind_vector is None:
            wind_vector = np.zeros(3, dtype=np.float32)

        # 1. Rotation von Body-Frame zu World-Frame
        R = self.get_rotation_matrix()

        # 2. Thrust-Richtung im World-Frame
        thrust_direction_world = R @ np.array([0, 0, 1], dtype=np.float32)

        # 3. Geschwindigkeit in Thrust-Richtung projizieren
        velocity_in_thrust_direction = np.dot(self.velocity, thrust_direction_world)

        # 4. Thrust-Modifikation basierend auf Geschwindigkeit
        # Bei negativer Geschwindigkeit (Fallen): mehr Thrust-Effizienz
        # Bei positiver Geschwindigkeit (Steigen): weniger Thrust-Effizienz
        max_speed_in_thrust_dir = 30.0  # m/s
        speed_factor = max(0.0, 1.0 - (velocity_in_thrust_direction / max_speed_in_thrust_dir))

        # 5. Thrust-Kräfte berechnen
        thrusts = motor_thrusts * self.thrust_coeff * speed_factor

        # 6. Gesamtkraft in Body-Frame
        total_thrust_body = np.array([0, 0, np.sum(thrusts)], dtype=np.float32)

        # 7. Rotation zu World-Frame
        total_force_world = R @ total_thrust_body

        # 8. Gravitation
        gravity_force = np.array([0, 0, -self.mass * gravity], dtype=np.float32)

        # 9. Luftwiderstand (relativ zum Wind)
        relative_velocity = self.velocity - wind_vector
        relative_speed = np.linalg.norm(relative_velocity)

        if relative_speed > 0.01:
            drag_force = -self.linear_drag_coeff * relative_speed * relative_velocity
        else:
            drag_force = np.zeros(3, dtype=np.float32)

        # 10. Gesamtkraft und lineare Beschleunigung
        total_force = total_force_world + gravity_force + drag_force
        linear_acceleration = total_force / self.mass

        # 11. Drehmoment berechnen
        torque = self._compute_torque(thrusts)

        # 12. Pendel-Stabilisierung
        roll, pitch, _ = self.orientation
        pendulum_torque = self.pendulum_damping * np.array([
            -self.mass * gravity * self.center_of_mass_offset * np.sin(roll),
            -self.mass * gravity * self.center_of_mass_offset * np.sin(pitch),
            0.0
        ], dtype=np.float32)

        # 13. Winkelbeschleunigung
        angular_acceleration = (torque + pendulum_torque) / self.inertia

        # 14. Integration (Euler)
        self.velocity += linear_acceleration * dt
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        self.position += self.velocity * dt

        self.angular_velocity += angular_acceleration * dt
        self.angular_velocity *= (1 - self.angular_drag_coeff)
        self.angular_velocity = np.clip(
            self.angular_velocity,
            -max_angular_velocity,
            max_angular_velocity
        )
        self.orientation += self.angular_velocity * dt

        # Normalisiere Euler-Winkel auf [-pi, pi]
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

    def _compute_torque(self, thrusts: np.ndarray) -> np.ndarray:
        """
        Berechnet das Drehmoment basierend auf Rotor-Thrusts.

        In X-Konfiguration:
        - Roll: Thrust-Differenz zwischen rechten und linken Motoren
        - Pitch: Differenz zwischen vorderen und hinteren Motoren
        - Yaw: Differenz in Drehrichtungen (reaktives Torque)

        Args:
            thrusts: Array mit 4 Thrust-Werten

        Returns:
            Drehmoment-Vektor [roll_torque, pitch_torque, yaw_torque]
        """
        # Roll-Torque (um X-Achse): Rechte (0, 3) vs. Linke (1, 2)
        roll_torque = (thrusts[0] + thrusts[3] - thrusts[1] - thrusts[2]) * \
                      self.arm_length / np.sqrt(2)

        # Pitch-Torque (um Y-Achse): Vordere (0, 2) vs. Hintere (1, 3)
        pitch_torque = (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3]) * \
                       self.arm_length / np.sqrt(2)

        # Yaw-Torque (um Z-Achse): Reaktive Torques
        yaw_torque = np.sum(self.rotor_directions * thrusts) * self.torque_coeff

        return np.array([roll_torque, pitch_torque, yaw_torque], dtype=np.float32)

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Berechnet die Rotationsmatrix von Body-Frame zu World-Frame.

        Verwendet Euler-Winkel [roll, pitch, yaw] in ZYX-Konvention.

        Returns:
            3x3 Rotationsmatrix
        """
        roll, pitch, yaw = self.orientation

        # Roll (X-Achse)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ], dtype=np.float32)

        # Pitch (Y-Achse)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ], dtype=np.float32)

        # Yaw (Z-Achse)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Kombinierte Rotation: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gibt den aktuellen Zustand der Drohne zurück.

        Returns:
            Tuple von (position, velocity, orientation, angular_velocity)
        """
        return (
            self.position.copy(),
            self.velocity.copy(),
            self.orientation.copy(),
            self.angular_velocity.copy()
        )

    def set_state(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        orientation: np.ndarray = None,
        angular_velocity: np.ndarray = None
    ):
        """
        Setzt den Zustand der Drohne.

        Args:
            position: Neue Position [x, y, z]
            velocity: Neue Geschwindigkeit [vx, vy, vz]
            orientation: Neue Orientierung [roll, pitch, yaw]
            angular_velocity: Neue Winkelgeschwindigkeit [wx, wy, wz]
        """
        if position is not None:
            self.position = position.astype(np.float32)
        if velocity is not None:
            self.velocity = velocity.astype(np.float32)
        if orientation is not None:
            self.orientation = orientation.astype(np.float32)
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity.astype(np.float32)

    def check_crash(
        self,
        z_velocity_threshold: float = -20.0,
        tilt_threshold_rad: float = None,
        max_distance: float = None,
        target_position: np.ndarray = None
    ) -> bool:
        """
        Prüft ob die Drohne abgestürzt ist.

        Args:
            z_velocity_threshold: Schwellwert für vertikale Geschwindigkeit
            tilt_threshold_rad: Schwellwert für Roll/Pitch in Radiant
            max_distance: Maximale Entfernung vom Ziel
            target_position: Zielposition für Distanz-Check

        Returns:
            True wenn Crash detektiert, sonst False
        """
        # Fällt zu schnell
        if self.velocity[2] < z_velocity_threshold:
            return True

        # Extreme Neigung
        if tilt_threshold_rad is not None:
            roll, pitch, _ = self.orientation
            if abs(roll) > tilt_threshold_rad or abs(pitch) > tilt_threshold_rad:
                return True

        # Zu weit vom Ziel entfernt
        if max_distance is not None and target_position is not None:
            distance = np.linalg.norm(target_position - self.position)
            if distance > max_distance:
                return True

        return False

