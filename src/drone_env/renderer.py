"""
Renderer für das DroneEnv - Visualisierung der Drohne und des Ziels.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Backend explizit setzen für stabiles Rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional


class DroneEnvRenderer:
    """
    Renderer-Klasse für DroneEnv.

    Visualisiert die Drohne in zwei Ansichten:
    - Top View (Draufsicht XY-Ebene)
    - Front View (Vorderansicht XZ-Ebene)
    """

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialisiert den Renderer.

        Args:
            render_mode: "human" für interaktive Anzeige, "rgb_array" für Array-Output
        """
        self.render_mode = render_mode

        # Matplotlib Komponenten
        self.fig = None
        self.ax_top = None
        self.ax_front = None

        # Rendering-Objekte für Performance (werden wiederverwendet)
        self._render_objects = {
            'drone_circle_top': None,
            'drone_circle_front': None,
            'rotor_lines_top': [],
            'rotor_lines_front': [],
            'rotor_circles_top': [],
            'rotor_circles_front': [],
            'tilt_arrow_top': None,
            'tilt_arrow_front': None,
            'target_circle_top': None,
            'target_circle_front': None,
            'target_cross_top': [],
            'target_cross_front': [],
            'connection_line_top': None,
            'connection_line_front': None,
            'wind_arrow': None,
            'info_text': None,
            'boden_line': None,
        }

    def initialize(self):
        """Initialisiert die Rendering-Komponenten beim ersten Aufruf."""
        if self.render_mode is None:
            return

        # Für human mode: interaktiver Modus
        if self.render_mode == "human":
            plt.ion()

        # 2 Subplots: Oben Draufsicht (XY), Unten Vorderansicht (XZ)
        self.fig, (self.ax_top, self.ax_front) = plt.subplots(2, 1, figsize=(10, 14))
        self.fig.set_facecolor('white')
        self.fig.subplots_adjust(hspace=0.3)

        # ========== TOP VIEW (Draufsicht XY) ==========
        self.ax_top.set_xlim(-30, 30)
        self.ax_top.set_ylim(-30, 30)
        self.ax_top.set_aspect('equal')
        self.ax_top.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        self.ax_top.set_xlabel('X (m)', fontsize=11)
        self.ax_top.set_ylabel('Y (m)', fontsize=11)
        self.ax_top.set_facecolor('#f0f0f0')

        # ========== FRONT VIEW (Vorderansicht XZ) ==========
        self.ax_front.set_xlim(-30, 30)
        self.ax_front.set_ylim(-30, 30)
        self.ax_front.set_aspect('equal')
        self.ax_front.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        self.ax_front.set_xlabel('X (m)', fontsize=11)
        self.ax_front.set_ylabel('Z (Höhe) (m)', fontsize=11)
        self.ax_front.set_title('Vorderansicht (Front View)', fontsize=12, fontweight='bold')
        self.ax_front.set_facecolor('#f0f0f0')

        # Boden-Linie in Front-View
        self._render_objects['boden_line'] = self.ax_front.axhline(
            y=0, color='brown', linestyle='-', linewidth=2, alpha=0.5, label='Boden'
        )

    def render(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray,
        target_position: np.ndarray,
        wind_vector: np.ndarray,
        rotation_matrix: np.ndarray,
        rotor_positions: np.ndarray,
        step_count: int,
        reward: float
    ):
        """
        Rendert die aktuelle Szene.

        Args:
            position: Position der Drohne [x, y, z]
            velocity: Geschwindigkeit der Drohne [vx, vy, vz]
            orientation: Orientierung der Drohne [roll, pitch, yaw]
            angular_velocity: Winkelgeschwindigkeit [wx, wy, wz]
            target_position: Position des Ziels [x, y, z]
            wind_vector: Windvektor [wx, wy, wz]
            rotation_matrix: 3x3 Rotationsmatrix Body->World
            rotor_positions: Nx3 Array mit Rotor-Positionen im Body-Frame
            step_count: Aktuelle Schrittzahl
            reward: Aktueller Reward-Wert

        Returns:
            RGB array wenn render_mode="rgb_array", sonst None
        """
        if self.render_mode is None:
            return None

        # Erstelle Figure beim ersten Aufruf
        first_render = self.fig is None
        if first_render:
            self.initialize()

        # Update Titel
        self.ax_top.set_title(f'Draufsicht (Top View) - Step: {step_count}', fontsize=12, fontweight='bold')

        # ========== DROHNE ZEICHNEN ==========
        self._render_drone(position, first_render)

        # ========== ROTOREN ZEICHNEN ==========
        self._render_rotors(position, rotation_matrix, rotor_positions, first_render)

        # ========== NEIGUNG/ORIENTIERUNG ==========
        self._render_orientation(position, rotation_matrix, first_render)

        # ========== ZIELPUNKT ==========
        self._render_target(position, target_position, first_render)

        # ========== WIND-VEKTOR ==========
        self._render_wind(wind_vector, first_render)

        # ========== INFO-BOX ==========
        self._render_info(
            position, velocity, orientation, target_position,
            wind_vector, step_count, reward
        )

        # Legenden nur beim ersten Render
        if first_render:
            self.ax_top.legend(loc='upper right', fontsize=9)
            self.ax_front.legend(loc='upper right', fontsize=9)

        # Rendering durchführen
        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.01)  # Pause für GUI-Update
            return None
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            height, width = self.fig.canvas.get_width_height()
            image = buf.reshape((height, width, 4))[:, :, :3]  # RGBA -> RGB
            return image

    def _render_drone(self, position: np.ndarray, first_render: bool):
        """Rendert den Drohnen-Körper."""
        # TOP VIEW
        if first_render:
            self._render_objects['drone_circle_top'] = Circle(
                (position[0], position[1]),
                0.3,
                color='#0066cc',
                alpha=0.9,
                zorder=5,
                label='Drohne'
            )
            self.ax_top.add_patch(self._render_objects['drone_circle_top'])
        else:
            self._render_objects['drone_circle_top'].center = (position[0], position[1])

        # FRONT VIEW
        if first_render:
            self._render_objects['drone_circle_front'] = Circle(
                (position[0], position[2]),
                0.3,
                color='#0066cc',
                alpha=0.9,
                zorder=5,
                label='Drohne'
            )
            self.ax_front.add_patch(self._render_objects['drone_circle_front'])
        else:
            self._render_objects['drone_circle_front'].center = (position[0], position[2])

    def _render_rotors(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        rotor_positions: np.ndarray,
        first_render: bool
    ):
        """Rendert die Rotoren der Drohne."""
        rotor_colors = ['#ff6666', '#ff6666', '#66ff66', '#66ff66']  # Rot: CW, Grün: CCW
        rotor_scale = 3.0  # Skalierung für bessere Visualisierung

        # Initialisiere Listen beim ersten Render
        if first_render:
            self._render_objects['rotor_lines_top'] = []
            self._render_objects['rotor_lines_front'] = []
            self._render_objects['rotor_circles_top'] = []
            self._render_objects['rotor_circles_front'] = []

        for i, (rotor_pos_body, color) in enumerate(zip(rotor_positions, rotor_colors)):
            # Transformiere Rotor-Position von Body-Frame zu World-Frame
            rotor_pos_world = rotation_matrix @ rotor_pos_body
            rotor_pos_world_scaled = rotor_pos_world * rotor_scale

            # World-Koordinaten der Rotoren
            rotor_x = position[0] + rotor_pos_world_scaled[0]
            rotor_y = position[1] + rotor_pos_world_scaled[1]
            rotor_z = position[2] + rotor_pos_world_scaled[2]

            # --- TOP VIEW: Rotoren in XY-Ebene ---
            if first_render:
                line_top, = self.ax_top.plot(
                    [position[0], rotor_x],
                    [position[1], rotor_y],
                    color='#666666',
                    linewidth=2.5,
                    zorder=4,
                    alpha=0.8
                )
                self._render_objects['rotor_lines_top'].append(line_top)

                rotor_circle_top = Circle(
                    (rotor_x, rotor_y),
                    0.15,
                    color=color,
                    alpha=0.8,
                    zorder=6
                )
                self.ax_top.add_patch(rotor_circle_top)
                self._render_objects['rotor_circles_top'].append(rotor_circle_top)
            else:
                self._render_objects['rotor_lines_top'][i].set_data(
                    [position[0], rotor_x],
                    [position[1], rotor_y]
                )
                self._render_objects['rotor_circles_top'][i].center = (rotor_x, rotor_y)

            # --- FRONT VIEW: Rotoren in XZ-Ebene ---
            if first_render:
                line_front, = self.ax_front.plot(
                    [position[0], rotor_x],
                    [position[2], rotor_z],
                    color='#666666',
                    linewidth=2.5,
                    zorder=4,
                    alpha=0.8
                )
                self._render_objects['rotor_lines_front'].append(line_front)

                rotor_circle_front = Circle(
                    (rotor_x, rotor_z),
                    0.15,
                    color=color,
                    alpha=0.8,
                    zorder=6
                )
                self.ax_front.add_patch(rotor_circle_front)
                self._render_objects['rotor_circles_front'].append(rotor_circle_front)
            else:
                self._render_objects['rotor_lines_front'][i].set_data(
                    [position[0], rotor_x],
                    [position[2], rotor_z]
                )
                self._render_objects['rotor_circles_front'][i].center = (rotor_x, rotor_z)

    def _render_orientation(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        first_render: bool
    ):
        """Rendert die Orientierung/Neigung der Drohne."""
        # Normale der Drohne (zeigt nach "oben" im Body-Frame)
        normal_body = np.array([0, 0, 1])
        normal_world = rotation_matrix @ normal_body

        # --- TOP VIEW: Projektion der Neigung auf XY-Ebene ---
        tilt_x = normal_world[0]
        tilt_y = normal_world[1]
        tilt_magnitude = np.sqrt(tilt_x**2 + tilt_y**2)

        # Entferne alte Pfeile falls vorhanden
        if self._render_objects['tilt_arrow_top'] is not None:
            self._render_objects['tilt_arrow_top'].remove()
            self._render_objects['tilt_arrow_top'] = None

        if tilt_magnitude > 0.01:
            tilt_scale = 1.5
            self._render_objects['tilt_arrow_top'] = self.ax_top.arrow(
                position[0], position[1],
                tilt_x * tilt_scale, tilt_y * tilt_scale,
                head_width=0.3,
                head_length=0.25,
                fc='#ff9900',
                ec='#ff9900',
                linewidth=2.5,
                zorder=7,
                alpha=0.9,
                label='Neigung' if first_render else ''
            )

        # --- FRONT VIEW: Neigung in XZ-Ebene ---
        tilt_x_front = normal_world[0]
        tilt_z_front = normal_world[2]

        # Entferne alten Pfeil falls vorhanden
        if self._render_objects['tilt_arrow_front'] is not None:
            self._render_objects['tilt_arrow_front'].remove()
            self._render_objects['tilt_arrow_front'] = None

        if abs(tilt_x_front) > 0.01 or abs(tilt_z_front - 1.0) > 0.01:
            tilt_scale_front = 1.5
            self._render_objects['tilt_arrow_front'] = self.ax_front.arrow(
                position[0], position[2],
                tilt_x_front * tilt_scale_front, tilt_z_front * tilt_scale_front,
                head_width=0.3,
                head_length=0.25,
                fc='#ff9900',
                ec='#ff9900',
                linewidth=2.5,
                zorder=7,
                alpha=0.9,
                label='Neigung' if first_render else ''
            )

    def _render_target(
        self,
        position: np.ndarray,
        target_position: np.ndarray,
        first_render: bool
    ):
        """Rendert den Zielpunkt."""
        # --- TOP VIEW: Ziel in XY-Ebene ---
        if first_render:
            self._render_objects['target_circle_top'] = Circle(
                (target_position[0], target_position[1]),
                1.0,
                color='#00cc00',
                alpha=0.6,
                zorder=4,
                label='Ziel'
            )
            self.ax_top.add_patch(self._render_objects['target_circle_top'])

            # Ziel-Kreuz (Top View)
            cross_size = 0.5
            line1, = self.ax_top.plot(
                [target_position[0] - cross_size, target_position[0] + cross_size],
                [target_position[1], target_position[1]],
                'g-', linewidth=2, zorder=5
            )
            line2, = self.ax_top.plot(
                [target_position[0], target_position[0]],
                [target_position[1] - cross_size, target_position[1] + cross_size],
                'g-', linewidth=2, zorder=5
            )
            self._render_objects['target_cross_top'] = [line1, line2]

            # Verbindungslinie zur Drohne (Top View)
            self._render_objects['connection_line_top'], = self.ax_top.plot(
                [position[0], target_position[0]],
                [position[1], target_position[1]],
                'k--',
                alpha=0.4,
                linewidth=1.5,
                zorder=1
            )
        else:
            # Update Positionen
            self._render_objects['target_circle_top'].center = (target_position[0], target_position[1])

            cross_size = 0.5
            self._render_objects['target_cross_top'][0].set_data(
                [target_position[0] - cross_size, target_position[0] + cross_size],
                [target_position[1], target_position[1]]
            )
            self._render_objects['target_cross_top'][1].set_data(
                [target_position[0], target_position[0]],
                [target_position[1] - cross_size, target_position[1] + cross_size]
            )

            self._render_objects['connection_line_top'].set_data(
                [position[0], target_position[0]],
                [position[1], target_position[1]]
            )

        # --- FRONT VIEW: Ziel in XZ-Ebene ---
        if first_render:
            self._render_objects['target_circle_front'] = Circle(
                (target_position[0], target_position[2]),
                1.0,
                color='#00cc00',
                alpha=0.6,
                zorder=4,
                label='Ziel'
            )
            self.ax_front.add_patch(self._render_objects['target_circle_front'])

            # Ziel-Kreuz (Front View)
            cross_size = 0.5
            line1, = self.ax_front.plot(
                [target_position[0] - cross_size, target_position[0] + cross_size],
                [target_position[2], target_position[2]],
                'g-', linewidth=2, zorder=5
            )
            line2, = self.ax_front.plot(
                [target_position[0], target_position[0]],
                [target_position[2] - cross_size, target_position[2] + cross_size],
                'g-', linewidth=2, zorder=5
            )
            self._render_objects['target_cross_front'] = [line1, line2]

            # Verbindungslinie zur Drohne (Front View)
            self._render_objects['connection_line_front'], = self.ax_front.plot(
                [position[0], target_position[0]],
                [position[2], target_position[2]],
                'k--',
                alpha=0.4,
                linewidth=1.5,
                zorder=1
            )
        else:
            # Update Positionen
            self._render_objects['target_circle_front'].center = (target_position[0], target_position[2])

            cross_size = 0.5
            self._render_objects['target_cross_front'][0].set_data(
                [target_position[0] - cross_size, target_position[0] + cross_size],
                [target_position[2], target_position[2]]
            )
            self._render_objects['target_cross_front'][1].set_data(
                [target_position[0], target_position[0]],
                [target_position[2] - cross_size, target_position[2] + cross_size]
            )

            self._render_objects['connection_line_front'].set_data(
                [position[0], target_position[0]],
                [position[2], target_position[2]]
            )

    def _render_wind(self, wind_vector: np.ndarray, first_render: bool):
        """Rendert den Wind-Vektor."""
        wind_scale = 3.0
        wind_x = wind_vector[0] * wind_scale
        wind_y = wind_vector[1] * wind_scale
        wind_mag = np.linalg.norm([wind_x, wind_y])

        # Entferne alten Wind-Pfeil falls vorhanden
        if self._render_objects['wind_arrow'] is not None:
            self._render_objects['wind_arrow'].remove()
            self._render_objects['wind_arrow'] = None

        if wind_mag > 0.1:  # Nur zeichnen wenn Wind spürbar
            self._render_objects['wind_arrow'] = self.ax_top.arrow(
                -25, 25,
                wind_x, wind_y,
                head_width=0.7,
                head_length=0.7,
                fc='#cc0000',
                ec='#cc0000',
                linewidth=2,
                alpha=0.8,
                zorder=3,
                label='Wind' if first_render else ''
            )

    def _render_info(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
        target_position: np.ndarray,
        wind_vector: np.ndarray,
        step_count: int,
        reward: float
    ):
        """Rendert die Info-Box mit Statusdaten."""
        distance = np.linalg.norm(target_position - position)
        velocity_mag = np.linalg.norm(velocity)
        wind_mag_full = np.linalg.norm(wind_vector)

        # Konvertiere Winkel zu Grad
        roll_deg = np.rad2deg(orientation[0])
        pitch_deg = np.rad2deg(orientation[1])
        yaw_deg = np.rad2deg(orientation[2])

        info_text = f'Step: {step_count}\n'
        info_text += f'Distanz: {distance:.2f}m\n'
        info_text += f'Höhe: {position[2]:.2f}m\n'
        info_text += f'Geschw.: {velocity_mag:.2f}m/s\n'
        info_text += f'Wind: {wind_mag_full:.2f}m/s\n'
        info_text += f'Roll: {roll_deg:.1f}°\n'
        info_text += f'Pitch: {pitch_deg:.1f}°\n'
        info_text += f'Yaw: {yaw_deg:.1f}°\n'
        info_text += f'Reward: {reward:.4f}'

        first_render = self._render_objects['info_text'] is None

        if first_render:
            self._render_objects['info_text'] = self.ax_top.text(
                0.02, 0.98,
                info_text,
                transform=self.ax_top.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                family='monospace'
            )
        else:
            self._render_objects['info_text'].set_text(info_text)

    def close(self):
        """Räumt Rendering-Ressourcen auf."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax_top = None
            self.ax_front = None
        if self.render_mode == "human":
            plt.ioff()  # Interaktiven Modus beenden

