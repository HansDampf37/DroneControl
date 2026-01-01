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
        self.screen_width = 1920
        self.screen_height = 1080
        self.screen = None
        self.clock = None

        # Camera settings
        self.camera_distance = space_side_length * 2.5
        self.camera_angle_h = 45  # Horizontal angle (degrees)
        self.camera_angle_v = -30  # Vertical angle (degrees)
        self.camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Camera offset
        self.fov = 60  # Field of view in degrees
        self.near_clip = 0.1  # Near clipping plane
        self.far_clip = 100.0  # Far clipping plane

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
        self.WIND_COLOR = (200, 0, 0)  # Red
        self.DROP_LINE_COLOR = (128, 128, 128)  # Gray

        # Font
        self.font = None
        self.font_small = None
        self.font_large = None

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
        self.font_large = pygame.font.Font(None, 36)
        self.font = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

    def _get_camera_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get camera basis vectors (forward, right, up).

        Returns:
            Tuple of (forward, right, up) unit vectors
        """
        angle_h_rad = np.deg2rad(self.camera_angle_h)
        angle_v_rad = np.deg2rad(self.camera_angle_v)

        # Forward direction
        forward = np.array([
            np.sin(angle_h_rad) * np.cos(angle_v_rad),
            np.cos(angle_h_rad) * np.cos(angle_v_rad),
            np.sin(angle_v_rad)
        ])

        # Right direction (perpendicular to forward)
        right = np.array([
            np.cos(angle_h_rad),
            -np.sin(angle_h_rad),
            0
        ])

        # Up direction (perpendicular to both)
        up = np.cross(right, forward)

        return forward, right, up

    def _project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project a 3D point to 2D screen coordinates using perspective projection.

        Args:
            point_3d: 3D point [x, y, z] in world coordinates

        Returns:
            Tuple of (screen_x, screen_y) coordinates
        """
        # Get camera vectors
        forward, right, up = self._get_camera_vectors()

        # Camera position in world space
        camera_pos = self.camera_position - forward * self.camera_distance

        # Transform point to camera space
        point_relative = point_3d - camera_pos

        # Project onto camera basis vectors
        x_cam = np.dot(point_relative, right)
        y_cam = np.dot(point_relative, up)
        z_cam = np.dot(point_relative, forward)

        # Perspective division
        if z_cam < self.near_clip:
            # Point is behind camera or too close
            return -1000, -1000  # Off-screen

        # Calculate perspective scale factor
        aspect_ratio = self.screen_width / self.screen_height
        fov_rad = np.deg2rad(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)

        # Perspective projection
        x_ndc = (x_cam / z_cam) * f / aspect_ratio
        y_ndc = (y_cam / z_cam) * f

        # Convert from NDC [-1, 1] to screen coordinates
        screen_x = int((x_ndc + 1.0) * 0.5 * self.screen_width)
        screen_y = int((1.0 - y_ndc) * 0.5 * self.screen_height)

        return screen_x, screen_y

    def _get_depth(self, point_3d: np.ndarray) -> float:
        """
        Calculate depth of a 3D point for z-ordering.

        Args:
            point_3d: 3D point [x, y, z]

        Returns:
            Depth value (higher = further away)
        """
        forward, _, _ = self._get_camera_vectors()
        camera_pos = self.camera_position - forward * self.camera_distance
        point_relative = point_3d - camera_pos

        return np.dot(point_relative, forward)

    def _get_perspective_scale(self, point_3d: np.ndarray) -> float:
        """
        Get perspective scale factor for a 3D point.

        Args:
            point_3d: 3D point [x, y, z]

        Returns:
            Scale factor (smaller for distant objects)
        """
        depth = self._get_depth(point_3d)
        if depth < self.near_clip:
            return 0.0

        fov_rad = np.deg2rad(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)

        # Scale decreases with distance
        return f / depth * 200  # 200 is base scale in pixels

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
        """Draw a 3D sphere at the given position with perspective."""
        screen_x, screen_y = self._project_3d_to_2d(position)

        # Check if point is visible
        if screen_x < -500 or screen_x > self.screen_width + 500:
            return
        if screen_y < -500 or screen_y > self.screen_height + 500:
            return

        # Scale radius based on perspective
        scale = self._get_perspective_scale(position)
        screen_radius = int(radius * scale)

        if screen_radius > 0:
            # Draw filled circle with anti-aliasing
            gfxdraw.filled_circle(self.screen, screen_x, screen_y, screen_radius, color)
            gfxdraw.aacircle(self.screen, screen_x, screen_y, screen_radius, color)

            # Add shading for 3D effect
            highlight_color = tuple(min(255, c + 50) for c in color)
            highlight_radius = max(1, screen_radius // 3)
            highlight_offset_x = -screen_radius // 4
            highlight_offset_y = -screen_radius // 4

            if highlight_radius > 0:
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

    def _draw_drop_line(self, position: np.ndarray, color: Tuple[int, int, int] = (128, 128, 128),
                        width: int = 1, dashed: bool = True, dash_length: float = 0.1):
        """
        Draw a vertical line from a position down to the XY-plane (z=0).

        Args:
            position: 3D position [x, y, z]
            color: Line color (default: gray)
            width: Line width
            dashed: If True, draw a dashed line
            dash_length: Length of each dash segment in meters (default: 0.1m)
        """
        # Start point (object position)
        start = position.copy()
        # End point (projection on XY-plane)
        end = position.copy()
        end[2] = 0.0  # Set z to 0 (ground plane)

        if dashed:
            # Calculate total line length
            total_length = float(np.linalg.norm(end - start))

            if total_length < 1e-6:
                return  # Don't draw if line is too short

            # Calculate number of segments based on fixed dash length
            num_dashes = max(1, int(total_length / dash_length))
            segment_length = total_length / (2 * num_dashes)  # Account for gaps

            # Draw dashed line with consistent segment lengths
            direction = (end - start) / total_length
            current_pos = 0.0

            while current_pos < total_length:
                # Draw a dash
                segment_start = start + direction * current_pos
                segment_end_pos = min(current_pos + segment_length, total_length)
                segment_end = start + direction * segment_end_pos

                self._draw_line_3d(segment_start, segment_end, color, width)

                # Skip a gap (same length as dash)
                current_pos += 2 * segment_length
        else:
            # Draw solid line
            self._draw_line_3d(start, end, color, width)

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

    def _draw_drone(self, position: np.ndarray, rotation_matrix: np.ndarray, rotor_positions: np.ndarray, wind_vector: np.ndarray = None):
        """Draw the drone with body and rotors."""
        rotor_scale = 1.2

        # Rotor colors - diagonal motors have the same rotation direction
        # Motor layout (looking from above):
        #   0(CW)     2(CCW)
        #        \ | /
        #         \|/
        #        --+--
        #         /|\
        #        / | \
        #   3(CCW)    1(CW)
        rotor_colors = [
            self.ROTOR_CW_COLOR,   # Motor 0 - CW (front-right)
            self.ROTOR_CW_COLOR,   # Motor 1 - CW (rear-right)
            self.ROTOR_CCW_COLOR,  # Motor 2 - CCW (front-left)
            self.ROTOR_CCW_COLOR,  # Motor 3 - CCW (rear-left)

        ]

        # Draw main body
        self._draw_sphere(position, 0.2, self.DRONE_BODY_COLOR)

        # Draw rotors first (so they appear behind the body if needed)
        for i, (rotor_pos_body, color) in enumerate(zip(rotor_positions, rotor_colors)):
            # Transform rotor position to world frame
            rotor_pos_world = rotation_matrix @ rotor_pos_body
            rotor_pos_world_scaled = rotor_pos_world * rotor_scale
            rotor_world_position = position + rotor_pos_world_scaled

            # Draw rotor sphere (smaller than body)
            self._draw_sphere(rotor_world_position, 0.1, color)

            # Draw arm connecting to body
            self._draw_line_3d(position, rotor_world_position, (100, 100, 100), 2)

        # Draw wind direction arrow on the drone
        if wind_vector is not None and np.linalg.norm(wind_vector) > 0.1:
            wind_normalized = wind_vector / np.linalg.norm(wind_vector)
            self._draw_arrow_3d(position, wind_normalized, self.WIND_COLOR, scale=0.4, width=2)

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

    def _draw_wind_indicator(self, wind_vector: np.ndarray):
        """Draw wind direction indicator in the top-right corner."""
        # Position in top-right corner
        origin_x = self.screen_width - 100
        origin_y = 100
        arrow_scale = 40  # Base scale for arrow

        # Draw background circle
        pygame.draw.circle(self.screen, (255, 255, 255), (origin_x, origin_y), 50)
        pygame.draw.circle(self.screen, (0, 0, 0), (origin_x, origin_y), 50, 2)

        # Calculate wind magnitude and direction
        wind_mag = np.linalg.norm(wind_vector)

        if wind_mag > 0.1:
            # Normalize wind vector
            wind_normalized = wind_vector / wind_mag

            # Apply camera rotation to wind vector
            angle_h_rad = np.deg2rad(self.camera_angle_h)
            angle_v_rad = np.deg2rad(self.camera_angle_v)

            # Rotate around Z axis
            x_rot = wind_normalized[0] * np.cos(angle_h_rad) - wind_normalized[1] * np.sin(angle_h_rad)
            y_rot = wind_normalized[0] * np.sin(angle_h_rad) + wind_normalized[1] * np.cos(angle_h_rad)
            z_rot = wind_normalized[2]

            # Rotate around X axis
            y_final = y_rot * np.cos(angle_v_rad) - z_rot * np.sin(angle_v_rad)
            z_final = y_rot * np.sin(angle_v_rad) + z_rot * np.cos(angle_v_rad)

            # Project to 2D (simple orthographic for the indicator)
            arrow_end_x = int(origin_x + x_rot * arrow_scale)
            arrow_end_y = int(origin_y - z_final * arrow_scale - y_final * arrow_scale * 0.5)

            # Draw arrow
            pygame.draw.line(self.screen, self.WIND_COLOR, (origin_x, origin_y),
                            (arrow_end_x, arrow_end_y), 4)

            # Draw arrowhead
            direction = np.array([arrow_end_x - origin_x, arrow_end_y - origin_y], dtype=float)
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
                perp = np.array([-direction[1], direction[0]])

                arrow_size = 10
                arrow_base = np.array([arrow_end_x, arrow_end_y]) - direction * arrow_size
                arrow_left = arrow_base + perp * arrow_size * 0.5
                arrow_right = arrow_base - perp * arrow_size * 0.5

                pygame.draw.polygon(self.screen, self.WIND_COLOR, [
                    (arrow_end_x, arrow_end_y),
                    (int(arrow_left[0]), int(arrow_left[1])),
                    (int(arrow_right[0]), int(arrow_right[1]))
                ])

        # Draw wind speed label
        wind_text = self.font_small.render(f"{wind_mag:.1f}m/s", True, self.TEXT_COLOR)
        wind_rect = wind_text.get_rect(center=(origin_x, origin_y + 65))
        self.screen.blit(wind_text, wind_rect)

        # Draw "Wind" label
        label_text = self.font_small.render("Wind", True, self.TEXT_COLOR)
        label_rect = label_text.get_rect(center=(origin_x, origin_y - 65))
        self.screen.blit(label_text, label_rect)

    def _draw_legend(self):
        """Draw a legend explaining the visual elements."""
        legend_x = self.screen_width - 200
        legend_y = 200  # Upper right corner, below wind indicator
        legend_width = 190
        line_height = 30

        legend_items = [
            ("Drone Body", self.DRONE_BODY_COLOR),
            ("CW Rotor", self.ROTOR_CW_COLOR),
            ("CCW Rotor", self.ROTOR_CCW_COLOR),
            ("Target", self.TARGET_COLOR),
            ("Up Direction", self.UP_ARROW_COLOR),
            ("Velocity", self.VELOCITY_ARROW_COLOR),
            ("Drop Line", self.DROP_LINE_COLOR),
            ("Wind", self.WIND_COLOR),
        ]

        legend_height = len(legend_items) * line_height + 40

        # Draw background
        panel_surface = pygame.Surface((legend_width, legend_height))
        panel_surface.set_alpha(200)
        panel_surface.fill((255, 255, 255))
        self.screen.blit(panel_surface, (legend_x, legend_y))

        # Draw border
        pygame.draw.rect(self.screen, (0, 0, 0),
                        (legend_x, legend_y, legend_width, legend_height), 2)

        # Draw title
        title = self.font_small.render("Legend", True, self.TEXT_COLOR)
        self.screen.blit(title, (legend_x + 10, legend_y + 10))

        # Draw legend items
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + 40 + i * line_height

            # Draw color indicator
            if label in ["Up Direction", "Velocity", "Drop Line"]:
                # Draw line for arrows/lines
                pygame.draw.line(self.screen, color,
                               (legend_x + 10, y_pos + 10),
                               (legend_x + 30, y_pos + 10), 3)
            else:
                # Draw circle for objects
                pygame.draw.circle(self.screen, color,
                                 (legend_x + 20, y_pos + 10), 8)

            # Draw label text
            text = self.font_small.render(label, True, self.TEXT_COLOR)
            self.screen.blit(text, (legend_x + 40, y_pos + 2))

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

        # Draw drop lines to ground plane (gray, dashed, consistent segment lengths)
        self._draw_drop_line(position, width=1, dashed=True)  # Drone drop line
        self._draw_drop_line(target_position, width=1, dashed=True)  # Target drop line

        # Draw connection line from drone to target
        self._draw_line_3d(position, target_position, (150, 150, 150), 1)

        # Draw target
        self._draw_target(target_position)

        # Draw drone
        self._draw_drone(position, rotation_matrix, rotor_positions, wind_vector)

        # Draw orientation arrow (up direction)
        normal_body = np.array([0, 0, 1])
        normal_world = rotation_matrix @ normal_body
        self._draw_arrow_3d(position, normal_world, self.UP_ARROW_COLOR, scale=0.5, width=3)

        # Draw velocity arrow
        if np.linalg.norm(velocity) > 0.1:
            vel_normalized = velocity / np.linalg.norm(velocity)
            self._draw_arrow_3d(position, vel_normalized, self.VELOCITY_ARROW_COLOR, scale=0.5, width=3)

        # Draw UI elements
        self._draw_info_panel(position, velocity, orientation, target_position, wind_vector,
                             step_count, episode_time, **kwargs)
        self._draw_motor_bars()
        self._draw_coordinate_system()
        self._draw_wind_indicator(wind_vector)
        self._draw_legend()

        # Update display
        if self.render_mode == "human":
            pygame.display.flip()
            # Match rendering rate to simulation timestep for real-time visualization
            target_fps = 1.0 / dt if dt > 0 else 60
            self.clock.tick(target_fps)
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

