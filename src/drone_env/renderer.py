"""
Renderer for DroneEnv - Visualization of drone and target.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Explicitly set backend for stable rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional


class DroneEnvRenderer:
    """
    Renderer class for DroneEnv.

    Visualizes the drone in two views:
    - Top View (bird's eye view of XY plane)
    - Front View (side view of XZ plane)
    """

    def __init__(self, render_mode: Optional[str] = None, space_side_length: float = 2.0):
        """
        Initializes the renderer.

        Args:
            render_mode: Rendering mode. Options:
                - "human": Interactive display for visualization
                - "rgb_array": Returns RGB array for video recording
                - None: No rendering
            space_side_length: Size of the observation space cube (meters).
                The renderer will show a grid slightly larger than this.
        """
        self.render_mode = render_mode
        self.space_side_length = space_side_length

        # Grid extends slightly beyond observation space (10% margin)
        self.grid_margin = 0.2  # 20% margin
        self.grid_limit = (space_side_length / 2) * (1 + self.grid_margin)

        # Matplotlib components
        self.fig = None
        self.ax_top = None
        self.ax_front = None
        self.ax_side = None
        self.ax_metrics = None

        # Current motor thrusts and episode time (for visualization only)
        self.current_motor_thrusts = np.zeros(4, dtype=np.float32)

        # Rendering objects for performance (reused across frames)
        self._render_objects = {
            'drone_circle_top': None,
            'drone_circle_front': None,
            'drone_circle_side': None,
            'rotor_lines_top': [],
            'rotor_lines_front': [],
            'rotor_lines_side': [],
            'rotor_circles_top': [],
            'rotor_circles_front': [],
            'rotor_circles_side': [],
            'tilt_arrow_top': None,
            'tilt_arrow_front': None,
            'tilt_arrow_side': None,
            'target_circle_top': None,
            'target_circle_front': None,
            'target_circle_side': None,
            'target_cross_top': [],
            'target_cross_front': [],
            'target_cross_side': [],
            'connection_line_top': None,
            'connection_line_front': None,
            'connection_line_side': None,
            'wind_arrow_top': None,
            'wind_arrow_front': None,
            'wind_arrow_side': None,
            'wind_arrow': None,
            'wind_text': None,
            'info_text': None,
            'ground_line': None,
            'ground_line_side': None,
            'motor_bars': [],
            'motor_bar_labels': [],
            'motor_bar_texts': [],
        }

    def initialize(self):
        """Initializes rendering components on first call."""
        if self.render_mode is None:
            return

        # For human mode: interactive mode
        if self.render_mode == "human":
            plt.ion()

        # Create 2x2 grid layout
        self.fig, ((self.ax_top, self.ax_side), (self.ax_front, self.ax_metrics)) = plt.subplots(
            2, 2, figsize=(16, 12)
        )
        self.fig.set_facecolor('white')
        self.fig.subplots_adjust(hspace=0.25, wspace=0.25)

        # ========== TOP VIEW (Bird's eye view XY) - TOP LEFT ==========
        self.ax_top.set_xlim(-self.grid_limit, self.grid_limit)
        self.ax_top.set_ylim(-self.grid_limit, self.grid_limit)
        self.ax_top.set_aspect('equal')
        self.ax_top.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        self.ax_top.set_xlabel('X (m)', fontsize=11)
        self.ax_top.set_ylabel('Y (m)', fontsize=11)
        self.ax_top.set_title('Top View', fontsize=12, fontweight='bold')
        self.ax_top.set_facecolor('#f0f0f0')

        # ========== FRONT VIEW (Side view XZ) - BOTTOM LEFT ==========
        self.ax_front.set_xlim(-self.grid_limit, self.grid_limit)
        self.ax_front.set_ylim(-self.grid_limit, self.grid_limit)
        self.ax_front.set_aspect('equal')
        self.ax_front.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        self.ax_front.set_xlabel('X (m)', fontsize=11)
        self.ax_front.set_ylabel('Z (Height) (m)', fontsize=11)
        self.ax_front.set_title('Front View', fontsize=12, fontweight='bold')
        self.ax_front.set_facecolor('#f0f0f0')

        # Ground line in front view
        self._render_objects['ground_line'] = self.ax_front.axhline(
            y=0, color='brown', linestyle='-', linewidth=2, alpha=0.5, label='Ground'
        )

        # ========== SIDE VIEW (YZ plane) - TOP RIGHT ==========
        self.ax_side.set_xlim(-self.grid_limit, self.grid_limit)
        self.ax_side.set_ylim(-self.grid_limit, self.grid_limit)
        self.ax_side.set_aspect('equal')
        self.ax_side.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        self.ax_side.set_xlabel('Y (m)', fontsize=11)
        self.ax_side.set_ylabel('Z (Height) (m)', fontsize=11)
        self.ax_side.set_title('Side View', fontsize=12, fontweight='bold')
        self.ax_side.set_facecolor('#f0f0f0')

        # Ground line in side view
        self._render_objects['ground_line_side'] = self.ax_side.axhline(
            y=0, color='brown', linestyle='-', linewidth=2, alpha=0.5, label='Ground'
        )

        # ========== METRICS PANEL - BOTTOM RIGHT ==========
        self.ax_metrics.set_xlim(0, 1)
        self.ax_metrics.set_ylim(0, 1)
        self.ax_metrics.axis('off')  # Hide axes for the metrics panel

    def render(self, env, **kwargs):
        """
        Renders the current scene.

        Args:
            env: DroneEnv instance containing all the state information needed for rendering.
            kwargs: Dictionary to log in info box

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode is None:
            return None

        # Extract state from environment
        position = env.drone.position
        velocity = env.drone.velocity
        orientation = env.drone.get_euler()
        target_position = env.target_position
        wind_vector = env.wind.get_vector()
        rotation_matrix = env.drone.get_rotation_matrix()
        rotor_positions = env.drone.rotor_positions
        step_count = env.step_count
        motor_thrusts = env.drone.motor_thrusts
        dt = env.dt

        # Store motor thrusts and calculate episode time
        if motor_thrusts is not None:
            self.current_motor_thrusts = motor_thrusts
        episode_time = step_count * dt

        # Create figure on first call
        first_render = self.fig is None
        if first_render:
            self.initialize()

        # ========== DRAW DRONE ==========
        self._render_drone(position, first_render)

        # ========== DRAW ROTORS ==========
        self._render_rotors(position, rotation_matrix, rotor_positions, first_render)

        # ========== DRAW TILT/ORIENTATION ==========
        self._render_orientation(position, rotation_matrix, first_render)

        # ========== DRAW TARGET ==========
        self._render_target(position, target_position, first_render)

        # ========== DRAW WIND VECTOR ==========
        self._render_wind(wind_vector, first_render)

        # ========== DRAW INFO BOX ==========
        self._render_info(
            position, velocity, orientation, target_position,
            wind_vector, step_count, episode_time, **kwargs
        )

        # ========== DRAW MOTOR POWER BARS ==========
        self._render_motor_bars(first_render)

        # Legends only on first render
        if first_render:
            self.ax_top.legend(loc='upper right', fontsize=9)
            self.ax_front.legend(loc='upper right', fontsize=9)
            self.ax_side.legend(loc='upper right', fontsize=9)

        # Perform rendering
        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.01)  # Pause for GUI update
            return None
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            height, width = self.fig.canvas.get_width_height()
            image = buf.reshape((height, width, 4))[:, :, :3]  # RGBA -> RGB
            return image

    def _render_drone(self, position: np.ndarray, first_render: bool):
        """Renders the drone body."""
        # TOP VIEW
        if first_render:
            self._render_objects['drone_circle_top'] = Circle(
                (position[0], position[1]),
                0.08,  # Reduced from 0.3 to 0.08m (8cm radius)
                color='#0066cc',
                alpha=0.9,
                zorder=5,
                label='Drone'
            )
            self.ax_top.add_patch(self._render_objects['drone_circle_top'])
        else:
            self._render_objects['drone_circle_top'].center = (position[0], position[1])

        # FRONT VIEW
        if first_render:
            self._render_objects['drone_circle_front'] = Circle(
                (position[0], position[2]),
                0.08,  # Reduced from 0.3 to 0.08m (8cm radius)
                color='#0066cc',
                alpha=0.9,
                zorder=5,
                label='Drone'
            )
            self.ax_front.add_patch(self._render_objects['drone_circle_front'])
        else:
            self._render_objects['drone_circle_front'].center = (position[0], position[2])

        # SIDE VIEW
        if first_render:
            self._render_objects['drone_circle_side'] = Circle(
                (position[1], position[2]),
                0.08,  # 8cm radius
                color='#0066cc',
                alpha=0.9,
                zorder=5,
                label='Drone'
            )
            self.ax_side.add_patch(self._render_objects['drone_circle_side'])
        else:
            self._render_objects['drone_circle_side'].center = (position[1], position[2])

    def _render_rotors(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        rotor_positions: np.ndarray,
        first_render: bool
    ):
        """Renders the drone's rotors."""
        rotor_colors = ['#ff6666', '#ff6666', '#66ff66', '#66ff66']  # Red: CW, Green: CCW
        rotor_scale = 1.2

        # Initialize lists on first render
        if first_render:
            self._render_objects['rotor_lines_top'] = []
            self._render_objects['rotor_lines_front'] = []
            self._render_objects['rotor_lines_side'] = []
            self._render_objects['rotor_circles_top'] = []
            self._render_objects['rotor_circles_front'] = []
            self._render_objects['rotor_circles_side'] = []

        for i, (rotor_pos_body, color) in enumerate(zip(rotor_positions, rotor_colors)):
            # Transform rotor position from body-frame to world-frame
            rotor_pos_world = rotation_matrix @ rotor_pos_body
            rotor_pos_world_scaled = rotor_pos_world * rotor_scale

            # World coordinates of rotors
            rotor_x = position[0] + rotor_pos_world_scaled[0]
            rotor_y = position[1] + rotor_pos_world_scaled[1]
            rotor_z = position[2] + rotor_pos_world_scaled[2]

            # --- TOP VIEW: Rotors in XY plane ---
            if first_render:
                line_top, = self.ax_top.plot(
                    [position[0], rotor_x],
                    [position[1], rotor_y],
                    color='#666666',
                    linewidth=1.5,  # Reduced from 2.5
                    zorder=4,
                    alpha=0.8
                )
                self._render_objects['rotor_lines_top'].append(line_top)

                rotor_circle_top = Circle(
                    (rotor_x, rotor_y),
                    0.04,  # Reduced from 0.15 to 0.04m (4cm radius)
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

            # --- FRONT VIEW: Rotors in XZ plane ---
            if first_render:
                line_front, = self.ax_front.plot(
                    [position[0], rotor_x],
                    [position[2], rotor_z],
                    color='#666666',
                    linewidth=1.5,  # Reduced from 2.5
                    zorder=4,
                    alpha=0.8
                )
                self._render_objects['rotor_lines_front'].append(line_front)

                rotor_circle_front = Circle(
                    (rotor_x, rotor_z),
                    0.04,  # Reduced from 0.15 to 0.04m (4cm radius)
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

            # --- SIDE VIEW: Rotors in YZ plane ---
            if first_render:
                line_side, = self.ax_side.plot(
                    [position[1], rotor_y],
                    [position[2], rotor_z],
                    color='#666666',
                    linewidth=1.5,
                    zorder=4,
                    alpha=0.8
                )
                self._render_objects['rotor_lines_side'].append(line_side)

                rotor_circle_side = Circle(
                    (rotor_y, rotor_z),
                    0.04,  # 4cm radius
                    color=color,
                    alpha=0.8,
                    zorder=6
                )
                self.ax_side.add_patch(rotor_circle_side)
                self._render_objects['rotor_circles_side'].append(rotor_circle_side)
            else:
                self._render_objects['rotor_lines_side'][i].set_data(
                    [position[1], rotor_y],
                    [position[2], rotor_z]
                )
                self._render_objects['rotor_circles_side'][i].center = (rotor_y, rotor_z)

    def _render_orientation(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        first_render: bool
    ):
        """Renders the drone's orientation/tilt."""
        # Normal vector of the drone (points "up" in body-frame)
        normal_body = np.array([0, 0, 1])
        normal_world = rotation_matrix @ normal_body

        # --- TOP VIEW: Projection of tilt onto XY plane ---
        tilt_x = normal_world[0]
        tilt_y = normal_world[1]
        tilt_magnitude = np.sqrt(tilt_x**2 + tilt_y**2)

        # Remove old arrow if present
        if self._render_objects['tilt_arrow_top'] is not None:
            self._render_objects['tilt_arrow_top'].remove()
            self._render_objects['tilt_arrow_top'] = None

        if tilt_magnitude > 0.01:
            tilt_scale = 1.5
            self._render_objects['tilt_arrow_top'] = self.ax_top.arrow(
                position[0], position[1],
                tilt_x * tilt_scale, tilt_y * tilt_scale,
                head_width=0.1,
                head_length=0.1,
                fc='#ff9900',
                ec='#ff9900',
                linewidth=2.5,
                zorder=7,
                alpha=0.9,
                label='Tilt' if first_render else ''
            )

        # --- FRONT VIEW: Tilt in XZ plane ---
        tilt_x_front = normal_world[0]
        tilt_z_front = normal_world[2]

        # Remove old arrow if present
        if self._render_objects['tilt_arrow_front'] is not None:
            self._render_objects['tilt_arrow_front'].remove()
            self._render_objects['tilt_arrow_front'] = None

        if abs(tilt_x_front) > 0.01 or abs(tilt_z_front - 1.0) > 0.01:
            tilt_scale_front = 1.5
            self._render_objects['tilt_arrow_front'] = self.ax_front.arrow(
                position[0], position[2],
                tilt_x_front * tilt_scale_front, tilt_z_front * tilt_scale_front,
                head_width=0.1,
                head_length=0.1,
                fc='#ff9900',
                ec='#ff9900',
                linewidth=2.5,
                zorder=7,
                alpha=0.9,
                label='Tilt' if first_render else ''
            )

        # --- SIDE VIEW: Tilt in YZ plane ---
        tilt_y_side = normal_world[1]
        tilt_z_side = normal_world[2]

        # Remove old arrow if present
        if self._render_objects['tilt_arrow_side'] is not None:
            self._render_objects['tilt_arrow_side'].remove()
            self._render_objects['tilt_arrow_side'] = None

        if abs(tilt_y_side) > 0.01 or abs(tilt_z_side - 1.0) > 0.01:
            tilt_scale_side = 1.5
            self._render_objects['tilt_arrow_side'] = self.ax_side.arrow(
                position[1], position[2],
                tilt_y_side * tilt_scale_side, tilt_z_side * tilt_scale_side,
                head_width=0.1,
                head_length=0.1,
                fc='#ff9900',
                ec='#ff9900',
                linewidth=2.5,
                zorder=7,
                alpha=0.9,
                label='Tilt' if first_render else ''
            )

    def _render_target(
        self,
        position: np.ndarray,
        target_position: np.ndarray,
        first_render: bool
    ):
        """Renders the target point."""
        # --- TOP VIEW: Target in XY plane ---
        if first_render:
            self._render_objects['target_circle_top'] = Circle(
                (target_position[0], target_position[1]),
                0.15,
                color='#00cc00',
                alpha=0.6,
                zorder=4,
                label='Target'
            )
            self.ax_top.add_patch(self._render_objects['target_circle_top'])

            # Target crosshair (Top View)
            cross_size = 0.1  # Reduced from 0.5 to 0.1m
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

            # Connection line to drone (Top View)
            self._render_objects['connection_line_top'], = self.ax_top.plot(
                [position[0], target_position[0]],
                [position[1], target_position[1]],
                'k--',
                alpha=0.4,
                linewidth=1.5,
                zorder=1
            )
        else:
            # Update positions
            self._render_objects['target_circle_top'].center = (target_position[0], target_position[1])

            cross_size = 0.1  # Reduced from 0.5 to 0.1m
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

        # --- FRONT VIEW: Target in XZ plane ---
        if first_render:
            self._render_objects['target_circle_front'] = Circle(
                (target_position[0], target_position[2]),
                0.15,  # Reduced from 1.0 to 0.15m (15cm radius)
                color='#00cc00',
                alpha=0.6,
                zorder=4,
                label='Target'
            )
            self.ax_front.add_patch(self._render_objects['target_circle_front'])

            # Target crosshair (Front View)
            cross_size = 0.1  # Reduced from 0.5 to 0.1m
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

            # Connection line to drone (Front View)
            self._render_objects['connection_line_front'], = self.ax_front.plot(
                [position[0], target_position[0]],
                [position[2], target_position[2]],
                'k--',
                alpha=0.4,
                linewidth=1.5,
                zorder=1
            )
        else:
            # Update positions
            self._render_objects['target_circle_front'].center = (target_position[0], target_position[2])

            cross_size = 0.1  # Reduced from 0.5 to 0.1m
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

        # --- SIDE VIEW: Target in YZ plane ---
        if first_render:
            self._render_objects['target_circle_side'] = Circle(
                (target_position[1], target_position[2]),
                0.15,
                color='#00cc00',
                alpha=0.6,
                zorder=4,
                label='Target'
            )
            self.ax_side.add_patch(self._render_objects['target_circle_side'])

            # Target crosshair (Side View)
            cross_size = 0.1
            line1, = self.ax_side.plot(
                [target_position[1] - cross_size, target_position[1] + cross_size],
                [target_position[2], target_position[2]],
                'g-', linewidth=2, zorder=5
            )
            line2, = self.ax_side.plot(
                [target_position[1], target_position[1]],
                [target_position[2] - cross_size, target_position[2] + cross_size],
                'g-', linewidth=2, zorder=5
            )
            self._render_objects['target_cross_side'] = [line1, line2]

            # Connection line to drone (Side View)
            self._render_objects['connection_line_side'], = self.ax_side.plot(
                [position[1], target_position[1]],
                [position[2], target_position[2]],
                'k--',
                alpha=0.4,
                linewidth=1.5,
                zorder=1
            )
        else:
            # Update positions
            self._render_objects['target_circle_side'].center = (target_position[1], target_position[2])

            cross_size = 0.1
            self._render_objects['target_cross_side'][0].set_data(
                [target_position[1] - cross_size, target_position[1] + cross_size],
                [target_position[2], target_position[2]]
            )
            self._render_objects['target_cross_side'][1].set_data(
                [target_position[1], target_position[1]],
                [target_position[2] - cross_size, target_position[2] + cross_size]
            )

            self._render_objects['connection_line_side'].set_data(
                [position[1], target_position[1]],
                [position[2], target_position[2]]
            )

    def _render_wind(self, wind_vector: np.ndarray, first_render: bool):
        """Renders the wind vector in all three view panels and the metrics panel."""
        wind_mag_full = np.linalg.norm(wind_vector)

        # --- TOP VIEW: Wind in XY plane ---
        # Remove old wind arrow if present
        if self._render_objects['wind_arrow_top'] is not None:
            self._render_objects['wind_arrow_top'].remove()
            self._render_objects['wind_arrow_top'] = None

        if wind_mag_full > 0.1:  # Only draw if wind is noticeable
            # Position wind indicator in corner of top view (world coordinates)
            wind_origin_x = -self.grid_limit * 0.75
            wind_origin_y = self.grid_limit * 0.75

            # Scale wind vector for visualization
            wind_scale = 0.3  # Scale factor for arrow size
            wind_dx = wind_vector[0] * wind_scale
            wind_dy = wind_vector[1] * wind_scale

            # Create arrow
            self._render_objects['wind_arrow_top'] = self.ax_top.arrow(
                wind_origin_x, wind_origin_y,
                wind_dx, wind_dy,
                head_width=0.15,
                head_length=0.1,
                fc='#cc0000',
                ec='#cc0000',
                linewidth=2.5,
                zorder=8,
                alpha=0.9,
                label='Wind' if first_render else ''
            )

        # --- FRONT VIEW: Wind in XZ plane ---
        # Remove old wind arrow if present
        if self._render_objects['wind_arrow_front'] is not None:
            self._render_objects['wind_arrow_front'].remove()
            self._render_objects['wind_arrow_front'] = None

        if wind_mag_full > 0.1:
            # Position wind indicator in corner of front view (world coordinates)
            wind_origin_x = -self.grid_limit * 0.75
            wind_origin_z = self.grid_limit * 0.75

            # Wind in XZ plane (X and Z components)
            wind_scale = 0.3
            wind_dx = wind_vector[0] * wind_scale
            wind_dz = wind_vector[2] * wind_scale

            # Create arrow
            self._render_objects['wind_arrow_front'] = self.ax_front.arrow(
                wind_origin_x, wind_origin_z,
                wind_dx, wind_dz,
                head_width=0.15,
                head_length=0.1,
                fc='#cc0000',
                ec='#cc0000',
                linewidth=2.5,
                zorder=8,
                alpha=0.9,
                label='Wind' if first_render else ''
            )

        # --- SIDE VIEW: Wind in YZ plane ---
        # Remove old wind arrow if present
        if self._render_objects['wind_arrow_side'] is not None:
            self._render_objects['wind_arrow_side'].remove()
            self._render_objects['wind_arrow_side'] = None

        if wind_mag_full > 0.1:
            # Position wind indicator in corner of side view (world coordinates)
            wind_origin_y = -self.grid_limit * 0.75
            wind_origin_z = self.grid_limit * 0.75

            # Wind in YZ plane (Y and Z components)
            wind_scale = 0.3
            wind_dy = wind_vector[1] * wind_scale
            wind_dz = wind_vector[2] * wind_scale

            # Create arrow
            self._render_objects['wind_arrow_side'] = self.ax_side.arrow(
                wind_origin_y, wind_origin_z,
                wind_dy, wind_dz,
                head_width=0.15,
                head_length=0.1,
                fc='#cc0000',
                ec='#cc0000',
                linewidth=2.5,
                zorder=8,
                alpha=0.9,
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
        episode_time: float,
        **kwargs
    ):
        """Renders the info box with status data in the metrics panel."""
        distance = np.linalg.norm(target_position - position)
        velocity_mag = np.linalg.norm(velocity)
        wind_mag_full = np.linalg.norm(wind_vector)

        # Convert angles to degrees
        roll_deg = np.rad2deg(orientation[0])
        pitch_deg = np.rad2deg(orientation[1])
        yaw_deg = np.rad2deg(orientation[2])

        info_text = f'Time: {episode_time:.2f}s\n'
        info_text += f'Step: {step_count}\n'
        info_text += f'Distance: {distance:.2f}m\n'
        info_text += f'Height: {position[2]:.2f}m\n'
        info_text += f'Velocity: {velocity_mag:.2f}m/s\n'
        info_text += f'Wind: {wind_mag_full:.2f}m/s\n'
        info_text += f'Roll: {roll_deg:.1f}°\n'
        info_text += f'Pitch: {pitch_deg:.1f}°\n'
        info_text += f'Yaw: {yaw_deg:.1f}°\n'
        for key, value in kwargs.items():
            info_text += f'{key}: {value}\n'

        first_render = self._render_objects['info_text'] is None

        if first_render:
            self._render_objects['info_text'] = self.ax_metrics.text(
                0.5, 0.95,  # Position in axes coordinates (centered, near top)
                info_text,
                fontsize=11,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                family='monospace'
            )
        else:
            self._render_objects['info_text'].set_text(info_text)

    def _render_motor_bars(self, first_render: bool):
        """Renders motor power bars in the metrics panel."""
        from matplotlib.patches import Rectangle

        # Bar parameters in axes coordinates
        bar_width = 0.08
        bar_max_height = 0.35
        bar_x_start = 0.3  # Start position in axes coordinates
        bar_x_spacing = 0.12  # Spacing between bars
        bar_x_positions = [bar_x_start + i * bar_x_spacing for i in range(4)]
        bar_y_base = 0.05  # Y position base in axes coordinates

        if first_render:
            # Create bars, labels, and text objects
            self._render_objects['motor_bars'] = []
            self._render_objects['motor_bar_labels'] = []
            self._render_objects['motor_bar_texts'] = []

            for i in range(4):
                # Create bar rectangle
                bar = Rectangle(
                    (bar_x_positions[i], bar_y_base),
                    bar_width,
                    0.0,  # Initial height of 0
                    facecolor='#ff0000',
                    edgecolor='#990000',
                    linewidth=2,
                    alpha=0.7,
                    clip_on=False
                )
                self.ax_metrics.add_patch(bar)
                self._render_objects['motor_bars'].append(bar)

                # Create label (M1, M2, M3, M4)
                label = self.ax_metrics.text(
                    bar_x_positions[i] + bar_width / 2,
                    bar_y_base - 0.03,
                    f'M{i+1}',
                    fontsize=10,
                    ha='center',
                    va='top',
                    fontweight='bold'
                )
                self._render_objects['motor_bar_labels'].append(label)

                # Create power percentage text
                power_text = self.ax_metrics.text(
                    bar_x_positions[i] + bar_width / 2,
                    bar_y_base + bar_max_height + 0.02,
                    '0%',
                    fontsize=9,
                    ha='center',
                    va='bottom',
                    family='monospace'
                )
                self._render_objects['motor_bar_texts'].append(power_text)

        # Update bar heights and text based on current motor thrusts
        for i in range(4):
            thrust = self.current_motor_thrusts[i]
            bar_height = thrust * bar_max_height

            # Update bar height
            self._render_objects['motor_bars'][i].set_height(bar_height)

            # Update power percentage text
            power_percent = int(thrust * 100)
            self._render_objects['motor_bar_texts'][i].set_text(f'{power_percent}%')

    def close(self):
        """Cleans up rendering resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax_top = None
            self.ax_front = None
            self.ax_side = None
            self.ax_metrics = None
        if self.render_mode == "human":
            plt.ioff()  # Exit interactive mode

