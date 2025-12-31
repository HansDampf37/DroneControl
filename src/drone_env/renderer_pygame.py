"""
Fast Pygame-based 3D Renderer for DroneEnv.

This renderer uses pygame for fast rendering of the drone in 3D space.
The drone is represented as 5 spheres (1 body + 4 rotors) with arrows
indicating the up direction and velocity direction.
"""
import numpy as np
import pygame
from typing import Optional, Tuple
from pygame import gfxdraw


class PyGameRenderer:
    """
    Fast pygame-based 3D renderer for drone visualization.

    Renders the drone as:
    - 5 spheres: 1 main body + 4 rotors
    - Arrow showing the "up" direction (drone orientation)
    - Arrow showing the velocity direction
    """

    def __init__(self, render_mode: Optional[str] = None, space_side_length: float = 2.0):
        """
        Initialize the pygame renderer.

        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            space_side_length: Size of the observation space cube (meters)
        """
        self.render_mode = render_mode
        self.space_side_length = space_side_length

        # Screen settings
        self.screen_width = 1200
        self.screen_height = 900
        self.screen = None
        self.clock = None

        # Camera settings
        self.camera_distance = space_side_length * 2.5
        self.camera_angle_h = 45  # Horizontal angle (degrees)
        self.camera_angle_v = 30  # Vertical angle (degrees)
        self.camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Camera offset

        # Mouse control state
        self.mouse_dragging = False
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.2  # degrees per pixel

        # Keyboard control state
        self.camera_speed = 0.1  # meters per frame

        # Colors
        self.BG_COLOR = (240, 240, 245)
        self.GRID_COLOR = (180, 180, 180)
        self.DRONE_BODY_COLOR = (0, 100, 200)
        self.ROTOR_CW_COLOR = (255, 100, 100)  # Red for clockwise
        self.ROTOR_CCW_COLOR = (100, 255, 100)  # Green for counter-clockwise
        self.TARGET_COLOR = (0, 200, 0)
        self.UP_ARROW_COLOR = (255, 150, 0)  # Orange
        self.VELOCITY_ARROW_COLOR = (255, 0, 255)  # Magenta
        self.GROUND_COLOR = (139, 90, 60)
        self.TEXT_COLOR = (0, 0, 0)

        # Font
        self.font = None
        self.font_small = None

        # Current motor thrusts for visualization
        self.current_motor_thrusts = np.zeros(4, dtype=np.float32)

    def initialize(self):
        """Initialize pygame components."""
        if self.render_mode is None:
            return

        pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Drone Control - 3D View")
        else:
            # For rgb_array mode, create a non-displayed surface
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

    def _project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project a 3D point to 2D screen coordinates using isometric-like projection.

        Args:
            point_3d: 3D point [x, y, z] in world coordinates

        Returns:
            Tuple of (screen_x, screen_y) coordinates
        """
        # Apply camera offset
        x, y, z = point_3d - self.camera_position

        # Apply camera rotation
        angle_h_rad = np.deg2rad(self.camera_angle_h)
        angle_v_rad = np.deg2rad(self.camera_angle_v)

        # Rotate around Z axis (horizontal rotation)
        x_rot = x * np.cos(angle_h_rad) - y * np.sin(angle_h_rad)
        y_rot = x * np.sin(angle_h_rad) + y * np.cos(angle_h_rad)
        z_rot = z

        # Rotate around X axis (vertical rotation)
        y_final = y_rot * np.cos(angle_v_rad) - z_rot * np.sin(angle_v_rad)
        z_final = y_rot * np.sin(angle_v_rad) + z_rot * np.cos(angle_v_rad)
        x_final = x_rot

        # Perspective projection
        scale = 200  # pixels per meter
        screen_x = int(self.screen_width / 2 + x_final * scale)
        screen_y = int(self.screen_height / 2 - z_final * scale - y_final * scale * 0.5)

        return screen_x, screen_y

    def _get_depth(self, point_3d: np.ndarray) -> float:
        """
        Calculate depth of a 3D point for z-ordering.

        Args:
            point_3d: 3D point [x, y, z]

        Returns:
            Depth value (higher = further away)
        """
        x, y, z = point_3d - self.camera_position
        angle_h_rad = np.deg2rad(self.camera_angle_h)
        angle_v_rad = np.deg2rad(self.camera_angle_v)

        # Rotate around Z axis
        x_rot = x * np.cos(angle_h_rad) - y * np.sin(angle_h_rad)
        y_rot = x * np.sin(angle_h_rad) + y * np.cos(angle_h_rad)

        # Rotate around X axis
        y_final = y_rot * np.cos(angle_v_rad) - z * np.sin(angle_v_rad)

        return y_final

    def _handle_keyboard_input(self):
        """Handle keyboard input for camera movement."""
        keys = pygame.key.get_pressed()

        # Calculate camera forward/right vectors based on horizontal angle
        angle_h_rad = np.deg2rad(self.camera_angle_h)

        # Forward direction (in XY plane)
        forward = np.array([
            np.sin(angle_h_rad),
            np.cos(angle_h_rad),
            0
        ])

        # Right direction (perpendicular to forward in XY plane)
        right = np.array([
            np.cos(angle_h_rad),
            -np.sin(angle_h_rad),
            0
        ])

        # Up direction
        up = np.array([0, 0, 1])

        # WASD for horizontal movement
        if keys[pygame.K_w]:
            self.camera_position += forward * self.camera_speed
        if keys[pygame.K_s]:
            self.camera_position -= forward * self.camera_speed
        if keys[pygame.K_a]:
            self.camera_position -= right * self.camera_speed
        if keys[pygame.K_d]:
            self.camera_position += right * self.camera_speed

        # Space/Shift for vertical movement
        if keys[pygame.K_SPACE]:
            self.camera_position += up * self.camera_speed
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            self.camera_position -= up * self.camera_speed

    def _handle_mouse_input(self, event):
        """Handle mouse input for camera rotation."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                self.mouse_dragging = True
                self.last_mouse_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.mouse_dragging = False
                self.last_mouse_pos = None

        elif event.type == pygame.MOUSEMOTION:
            if self.mouse_dragging and self.last_mouse_pos is not None:
                current_pos = pygame.mouse.get_pos()
                dx = current_pos[0] - self.last_mouse_pos[0]
                dy = current_pos[1] - self.last_mouse_pos[1]

                # Update camera angles
                self.camera_angle_h += dx * self.mouse_sensitivity
                self.camera_angle_v -= dy * self.mouse_sensitivity

                # Clamp vertical angle to avoid flipping
                self.camera_angle_v = np.clip(self.camera_angle_v, -89, 89)

                # Wrap horizontal angle
                self.camera_angle_h = self.camera_angle_h % 360

                self.last_mouse_pos = current_pos

    def _draw_sphere(self, position: np.ndarray, radius: float, color: Tuple[int, int, int]):
        """Draw a 3D sphere at the given position."""
        screen_x, screen_y = self._project_3d_to_2d(position)

        # Scale radius based on perspective (simple approach)
        screen_radius = int(radius * 200)  # 200 pixels per meter

        if screen_radius > 0 and 0 <= screen_x < self.screen_width and 0 <= screen_y < self.screen_height:
            # Draw filled circle with anti-aliasing
            gfxdraw.filled_circle(self.screen, screen_x, screen_y, screen_radius, color)
            gfxdraw.aacircle(self.screen, screen_x, screen_y, screen_radius, color)

            # Add shading for 3D effect
            highlight_color = tuple(min(255, c + 50) for c in color)
            highlight_radius = max(1, screen_radius // 3)
            highlight_offset_x = -screen_radius // 4
            highlight_offset_y = -screen_radius // 4
            gfxdraw.filled_circle(
                self.screen,
                screen_x + highlight_offset_x,
                screen_y + highlight_offset_y,
                highlight_radius,
                highlight_color
            )

    def _draw_line_3d(self, start: np.ndarray, end: np.ndarray, color: Tuple[int, int, int], width: int = 2):
        """Draw a line in 3D space."""
        start_2d = self._project_3d_to_2d(start)
        end_2d = self._project_3d_to_2d(end)
        pygame.draw.line(self.screen, color, start_2d, end_2d, width)

    def _draw_arrow_3d(self, start: np.ndarray, direction: np.ndarray, color: Tuple[int, int, int],
                       scale: float = 1.0, width: int = 3):
        """Draw a 3D arrow from start in the given direction."""
        end = start + direction * scale

        # Draw main line
        self._draw_line_3d(start, end, color, width)

        # Draw arrowhead
        if np.linalg.norm(direction) > 0.01:
            # Normalize direction
            dir_norm = direction / np.linalg.norm(direction)

            # Create perpendicular vectors for arrowhead
            if abs(dir_norm[2]) < 0.9:
                perp1 = np.cross(dir_norm, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(dir_norm, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(dir_norm, perp1)

            # Arrowhead geometry
            arrow_length = 0.15 * scale
            arrow_width = 0.1 * scale

            # Three points of the arrowhead
            tip = end
            base1 = end - dir_norm * arrow_length + perp1 * arrow_width
            base2 = end - dir_norm * arrow_length + perp2 * arrow_width
            base3 = end - dir_norm * arrow_length - perp1 * arrow_width
            base4 = end - dir_norm * arrow_length - perp2 * arrow_width

            self._draw_line_3d(tip, base1, color, width)
            self._draw_line_3d(tip, base2, color, width)
            self._draw_line_3d(tip, base3, color, width)
            self._draw_line_3d(tip, base4, color, width)

    def _draw_grid(self):
        """Draw a reference grid on the ground plane."""
        grid_size = int(self.space_side_length) + 2
        step = 0.5  # Grid spacing in meters

        # Draw grid lines
        for i in np.arange(-grid_size, grid_size + step, step):
            # Lines parallel to X axis
            start = np.array([i, -grid_size, 0])
            end = np.array([i, grid_size, 0])
            self._draw_line_3d(start, end, self.GRID_COLOR, 1)

            # Lines parallel to Y axis
            start = np.array([-grid_size, i, 0])
            end = np.array([grid_size, i, 0])
            self._draw_line_3d(start, end, self.GRID_COLOR, 1)

        # Draw ground plane (semi-transparent)
        # Draw thicker lines at the axes
        self._draw_line_3d(
            np.array([-grid_size, 0, 0]),
            np.array([grid_size, 0, 0]),
            self.GROUND_COLOR, 2
        )
        self._draw_line_3d(
            np.array([0, -grid_size, 0]),
            np.array([0, grid_size, 0]),
            self.GROUND_COLOR, 2
        )

    def _draw_drone(self, position: np.ndarray, rotation_matrix: np.ndarray, rotor_positions: np.ndarray):
        """Draw the drone with body and rotors."""
        rotor_scale = 1.2

        # Rotor colors (0,2 are CW, 1,3 are CCW)
        rotor_colors = [
            self.ROTOR_CW_COLOR,  # Motor 0 - CW
            self.ROTOR_CCW_COLOR,  # Motor 1 - CCW
            self.ROTOR_CW_COLOR,  # Motor 2 - CW
            self.ROTOR_CCW_COLOR   # Motor 3 - CCW
        ]

        # Draw rotors first (so they appear behind the body if needed)
        for i, (rotor_pos_body, color) in enumerate(zip(rotor_positions, rotor_colors)):
            # Transform rotor position to world frame
            rotor_pos_world = rotation_matrix @ rotor_pos_body
            rotor_pos_world_scaled = rotor_pos_world * rotor_scale
            rotor_world_position = position + rotor_pos_world_scaled

            # Draw rotor sphere (smaller than body)
            self._draw_sphere(rotor_world_position, 0.04, color)

            # Draw arm connecting to body
            self._draw_line_3d(position, rotor_world_position, (100, 100, 100), 2)

        # Draw main body
        self._draw_sphere(position, 0.08, self.DRONE_BODY_COLOR)

    def _draw_target(self, target_position: np.ndarray):
        """Draw the target position."""
        # Draw target sphere
        self._draw_sphere(target_position, 0.15, self.TARGET_COLOR)

        # Draw crosshair
        cross_size = 0.2
        self._draw_line_3d(
            target_position + np.array([-cross_size, 0, 0]),
            target_position + np.array([cross_size, 0, 0]),
            self.TARGET_COLOR, 2
        )
        self._draw_line_3d(
            target_position + np.array([0, -cross_size, 0]),
            target_position + np.array([0, cross_size, 0]),
            self.TARGET_COLOR, 2
        )
        self._draw_line_3d(
            target_position + np.array([0, 0, -cross_size]),
            target_position + np.array([0, 0, cross_size]),
            self.TARGET_COLOR, 2
        )

    def _draw_info_panel(self, position: np.ndarray, velocity: np.ndarray, orientation: np.ndarray,
                         target_position: np.ndarray, wind_vector: np.ndarray,
                         step_count: int, episode_time: float, **kwargs):
        """Draw information panel with drone stats."""
        distance = np.linalg.norm(target_position - position)
        velocity_mag = np.linalg.norm(velocity)
        wind_mag = np.linalg.norm(wind_vector)

        # Convert angles to degrees
        roll_deg = np.rad2deg(orientation[0])
        pitch_deg = np.rad2deg(orientation[1])
        yaw_deg = np.rad2deg(orientation[2])

        # Create info text
        info_lines = [
            f"Time: {episode_time:.2f}s  Step: {step_count}",
            f"Distance: {distance:.2f}m  Height: {position[2]:.2f}m",
            f"Velocity: {velocity_mag:.2f}m/s  Wind: {wind_mag:.2f}m/s",
            f"Roll: {roll_deg:.1f}°  Pitch: {pitch_deg:.1f}°  Yaw: {yaw_deg:.1f}°",
        ]

        # Add kwargs
        for key, value in kwargs.items():
            info_lines.append(f"{key}: {value}")

        # Draw background panel
        panel_x = 10
        panel_y = 10
        panel_width = 450
        line_height = 25
        panel_height = len(info_lines) * line_height + 20

        # Draw semi-transparent background
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill((255, 255, 200))
        self.screen.blit(panel_surface, (panel_x, panel_y))

        # Draw border
        pygame.draw.rect(self.screen, (0, 0, 0), (panel_x, panel_y, panel_width, panel_height), 2)

        # Draw text
        for i, line in enumerate(info_lines):
            text_surface = self.font_small.render(line, True, self.TEXT_COLOR)
            self.screen.blit(text_surface, (panel_x + 10, panel_y + 10 + i * line_height))

    def _draw_motor_bars(self):
        """Draw motor thrust bars."""
        bar_x = self.screen_width - 150
        bar_y = self.screen_height - 250
        bar_width = 25
        bar_max_height = 150
        bar_spacing = 35

        for i in range(4):
            thrust = self.current_motor_thrusts[i]
            bar_height = int(thrust * bar_max_height)

            # Draw bar background
            pygame.draw.rect(
                self.screen,
                (200, 200, 200),
                (bar_x + i * bar_spacing, bar_y, bar_width, bar_max_height),
                0
            )

            # Draw thrust level
            color = (255, int(255 * (1 - thrust)), 0)  # Red to yellow gradient
            pygame.draw.rect(
                self.screen,
                color,
                (bar_x + i * bar_spacing, bar_y + bar_max_height - bar_height, bar_width, bar_height),
                0
            )

            # Draw border
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (bar_x + i * bar_spacing, bar_y, bar_width, bar_max_height),
                2
            )

            # Draw label
            label = self.font_small.render(f"M{i+1}", True, self.TEXT_COLOR)
            label_rect = label.get_rect(center=(bar_x + i * bar_spacing + bar_width // 2, bar_y + bar_max_height + 15))
            self.screen.blit(label, label_rect)

            # Draw percentage
            pct_text = self.font_small.render(f"{int(thrust * 100)}%", True, self.TEXT_COLOR)
            pct_rect = pct_text.get_rect(center=(bar_x + i * bar_spacing + bar_width // 2, bar_y - 10))
            self.screen.blit(pct_text, pct_rect)

    def _draw_coordinate_system(self):
        """Draw a 3D coordinate system indicator in the bottom-left corner."""
        # Position in bottom-left corner
        origin_x = 80
        origin_y = self.screen_height - 80
        axis_length = 50

        # Define axis vectors in world space
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # Apply camera rotation to axes
        angle_h_rad = np.deg2rad(self.camera_angle_h)
        angle_v_rad = np.deg2rad(self.camera_angle_v)

        def rotate_axis(axis):
            # Rotate around Z axis
            x_rot = axis[0] * np.cos(angle_h_rad) - axis[1] * np.sin(angle_h_rad)
            y_rot = axis[0] * np.sin(angle_h_rad) + axis[1] * np.cos(angle_h_rad)
            z_rot = axis[2]

            # Rotate around X axis
            y_final = y_rot * np.cos(angle_v_rad) - z_rot * np.sin(angle_v_rad)
            z_final = y_rot * np.sin(angle_v_rad) + z_rot * np.cos(angle_v_rad)

            return np.array([x_rot, y_final, z_final])

        # Rotate axes
        x_rotated = rotate_axis(x_axis)
        y_rotated = rotate_axis(y_axis)
        z_rotated = rotate_axis(z_axis)

        # Project to 2D (simple orthographic for the indicator)
        def project_axis(axis_3d):
            return (
                int(origin_x + axis_3d[0] * axis_length),
                int(origin_y - axis_3d[2] * axis_length - axis_3d[1] * axis_length * 0.5)
            )

        x_2d = project_axis(x_rotated)
        y_2d = project_axis(y_rotated)
        z_2d = project_axis(z_rotated)

        # Draw background circle
        pygame.draw.circle(self.screen, (255, 255, 255), (origin_x, origin_y), 65, 0)
        pygame.draw.circle(self.screen, (0, 0, 0), (origin_x, origin_y), 65, 2)

        # Draw axes with labels (draw in order of depth for proper occlusion)
        axes_data = [
            (x_rotated, x_2d, (255, 0, 0), 'X'),    # Red
            (y_rotated, y_2d, (0, 255, 0), 'Y'),    # Green
            (z_rotated, z_2d, (0, 100, 255), 'Z')   # Blue
        ]

        # Sort by depth (y component after rotation, negative = further)
        axes_data.sort(key=lambda a: a[0][1])

        # Draw axes
        for axis_3d, axis_2d, color, label in axes_data:
            # Draw axis line
            pygame.draw.line(self.screen, color, (origin_x, origin_y), axis_2d, 3)

            # Draw arrowhead
            direction = np.array([axis_2d[0] - origin_x, axis_2d[1] - origin_y])
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
                perp = np.array([-direction[1], direction[0]])

                arrow_size = 8
                arrow_base = np.array(axis_2d) - direction * arrow_size
                arrow_left = arrow_base + perp * arrow_size * 0.5
                arrow_right = arrow_base - perp * arrow_size * 0.5

                pygame.draw.polygon(self.screen, color, [
                    axis_2d,
                    (int(arrow_left[0]), int(arrow_left[1])),
                    (int(arrow_right[0]), int(arrow_right[1]))
                ])

            # Draw label
            label_offset = 15
            label_pos = (
                int(axis_2d[0] + direction[0] * label_offset if length > 0 else axis_2d[0]),
                int(axis_2d[1] + direction[1] * label_offset if length > 0 else axis_2d[1])
            )
            label_surface = self.font.render(label, True, color)
            label_rect = label_surface.get_rect(center=label_pos)
            self.screen.blit(label_surface, label_rect)

        # Draw origin point
        pygame.draw.circle(self.screen, (0, 0, 0), (origin_x, origin_y), 4, 0)

    def render(self, env, **kwargs):
        """
        Render the current scene.

        Args:
            env: DroneEnv instance containing state information
            kwargs: Additional info to display

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode is None:
            return None

        # Initialize on first call
        if self.screen is None:
            self.initialize()

        # Handle pygame events (for window closing and mouse control)
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
                # Handle mouse input
                self._handle_mouse_input(event)

            # Handle keyboard input (continuous)
            self._handle_keyboard_input()

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

        # Store motor thrusts
        if motor_thrusts is not None:
            self.current_motor_thrusts = motor_thrusts
        episode_time = step_count * dt

        # Clear screen
        self.screen.fill(self.BG_COLOR)

        # Draw grid
        self._draw_grid()

        # Draw connection line from drone to target
        self._draw_line_3d(position, target_position, (150, 150, 150), 1)

        # Draw target
        self._draw_target(target_position)

        # Draw drone
        self._draw_drone(position, rotation_matrix, rotor_positions)

        # Draw orientation arrow (up direction)
        normal_body = np.array([0, 0, 1])
        normal_world = rotation_matrix @ normal_body
        self._draw_arrow_3d(position, normal_world, self.UP_ARROW_COLOR, scale=0.5, width=3)

        # Draw velocity arrow
        if np.linalg.norm(velocity) > 0.1:
            vel_normalized = velocity / np.linalg.norm(velocity)
            self._draw_arrow_3d(position, vel_normalized, self.VELOCITY_ARROW_COLOR, scale=0.5, width=3)

        # Draw wind indicator
        if np.linalg.norm(wind_vector) > 0.1:
            wind_pos = np.array([-self.space_side_length * 0.8, self.space_side_length * 0.8, self.space_side_length * 0.8])
            wind_normalized = wind_vector / np.linalg.norm(wind_vector)
            self._draw_arrow_3d(wind_pos, wind_normalized, (200, 0, 0), scale=0.5, width=3)

        # Draw UI elements
        self._draw_info_panel(position, velocity, orientation, target_position, wind_vector,
                             step_count, episode_time, **kwargs)
        self._draw_motor_bars()

        # Draw coordinate system indicator
        self._draw_coordinate_system()

        # Update display
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)  # Cap at 60 FPS
            return None
        elif self.render_mode == "rgb_array":
            # Convert pygame surface to RGB array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    def close(self):
        """Clean up pygame resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

