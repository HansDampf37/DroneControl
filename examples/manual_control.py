"""
Manual test for drone control with Pygame renderer.

Controls:
- Keys 1-4: Increase thrust for Motor 1-4 (hold to continue increasing)
- Keys 5-8: Decrease thrust for Motor 1-4 (hold to continue decreasing)
- Release key: Thrust returns to hover state
- Key ESC or X: Exit
- Key R: Reset drone position and orientation
- Mouse Drag: Rotate camera view
- Arrow Keys/WASD: Move camera position

The visualization shows the drone in 3D.
Motor configuration (X-configuration):
  Motor 2 (front-left, CCW)
         ○
        / \\
       /   \\
Motor 0 ○   ○ Motor 3
(front-  \\   / (rear-
right,   \\ /  right,
CW)        ○   CW)
    Motor 1 (rear-left, CCW)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv
import pygame
import time


class ManualDroneController:
    """Interactive controller for manual drone control."""

    def __init__(self):
        # dt=0.01 corresponds to 100 FPS (1/0.01 = 100 Hz)
        self.dt = 0.01  # Timestep in seconds

        # Create environment with pygame renderer
        self.env = DroneEnv(
            max_steps=10000,
            render_mode="human",
            renderer_type="pygame",  # Use pygame renderer
            enable_crash_detection=False,
            dt=self.dt,
            use_wind=False,
        )

        # Calculate hover thrust (equal thrust on all motors to balance weight)
        # For a 1kg drone, weight = mass * g = 1.0 * 9.81 = 9.81 N
        # Each motor needs to provide 1/4 of the weight
        # hover_thrust_per_motor = (mass * gravity) / (4 * max_thrust_per_motor)
        # Assuming max_thrust_per_motor ~ 4.0 N (from drone.py defaults)
        self.hover_thrust = 0.495  # Approximate hover thrust for 1kg drone
        self.env.drone.enable_pendulum = True
        self.env.drone.pendulum_k = 1

        # Motor thrust states (start at hover)
        self.motor_thrusts = np.array([self.hover_thrust] * 4, dtype=np.float32)

        # Thrust change rate (per second)
        self.thrust_change_rate = 0.5  # Change by 0.5 (50%) per second

        # Track which keys are currently pressed
        self.keys_pressed = set()

        # Simulation state
        self.running = True
        self.paused = False

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0

        # FPS control
        self.target_fps = 1.0 / self.dt  # Target FPS based on dt
        self.target_frame_time = self.dt  # Time per frame in seconds
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_display_time = time.time()
        self.actual_fps = 0.0

        # Reset environment
        self.obs, self.info = self.env.reset()
        self.episode_count = 1
        print("=" * 60)
        print("MANUAL DRONE CONTROL (Pygame Renderer)")
        print("=" * 60)
        print(f"\nSIMULATION PARAMETERS:")
        print(f"  Timestep (dt): {self.dt}s")
        print(f"  Target FPS: {self.target_fps:.0f}")
        print(f"  Frame time: {self.target_frame_time*1000:.1f}ms")
        print(f"  Hover thrust: {self.hover_thrust:.2f}")
        print("\nCONTROLS:")
        print("  1-4       : Increase thrust for Motor 1-4 (hold)")
        print("  5-8       : Decrease thrust for Motor 1-4 (hold)")
        print("  Release   : Return to hover thrust")
        print("  Mouse Drag: Rotate camera")
        print("  Arrow/WASD: Move camera")
        print("  R         : Reset drone")
        print("  X or ESC  : Exit")
        print("\nMOTOR CONFIGURATION (X-configuration):")
        print("     2(FL,CCW)")
        print("      ○")
        print("     / \\")
        print("  0 ○   ○ 3")
        print("   (FR,CW) (RR,CW)")
        print("     \\ /")
        print("      ○")
        print("     1(RL,CCW)")
        print("\nFL=front-left, FR=front-right")
        print("RL=rear-left, RR=rear-right")
        print("CW=clockwise, CCW=counter-clockwise")
        print("=" * 60)
        print("Control the drone in the Pygame window!\n")

    def handle_pygame_events(self):
        """Handle pygame events for motor control and forward camera events to renderer."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\nExiting...")
                self.running = False
                return

            # Forward mouse events to renderer for camera control
            elif event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION]:
                self.env.renderer._handle_mouse_input(event)

            elif event.type == pygame.KEYDOWN:
                # Track pressed keys
                key_name = pygame.key.name(event.key)
                self.keys_pressed.add(key_name)

                # Reset
                if event.key == pygame.K_r:
                    self.reset()

                # Exit
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_x:
                    print("\nExiting...")
                    self.running = False

            elif event.type == pygame.KEYUP:
                # Track released keys
                key_name = pygame.key.name(event.key)
                self.keys_pressed.discard(key_name)

        # Also handle camera keyboard input (WASD, arrows, space, shift) through renderer
        self.env.renderer._handle_keyboard_input()

    def update_motor_thrusts(self):
        """Update motor thrusts based on currently pressed keys."""
        # Calculate thrust change for this frame
        thrust_delta = self.thrust_change_rate * self.dt

        # Check which motors should change thrust
        thrust_changes = np.zeros(4, dtype=np.float32)

        # Keys 1-4: Increase thrust
        if '1' in self.keys_pressed:
            thrust_changes[0] = thrust_delta
        if '2' in self.keys_pressed:
            thrust_changes[1] = thrust_delta
        if '3' in self.keys_pressed:
            thrust_changes[2] = thrust_delta
        if '4' in self.keys_pressed:
            thrust_changes[3] = thrust_delta

        # Keys 5-8: Decrease thrust
        if '5' in self.keys_pressed:
            thrust_changes[0] = -thrust_delta
        if '6' in self.keys_pressed:
            thrust_changes[1] = -thrust_delta
        if '7' in self.keys_pressed:
            thrust_changes[2] = -thrust_delta
        if '8' in self.keys_pressed:
            thrust_changes[3] = -thrust_delta

        # Apply changes
        self.motor_thrusts += thrust_changes

        # Clamp thrusts to valid range [0, 1]
        self.motor_thrusts = np.clip(self.motor_thrusts, 0.0, 1.0)

        # If no keys are pressed for a motor, push it back toward hover thrust
        hover_return_rate = 2.0  # Return to hover faster than manual control
        for i in range(4):
            key_increase = str(i + 1) in self.keys_pressed
            key_decrease = str(i + 5) in self.keys_pressed

            if not key_increase and not key_decrease:
                # Push toward hover thrust
                diff = self.hover_thrust - self.motor_thrusts[i]
                self.motor_thrusts[i] += diff * hover_return_rate * self.dt

    def reset(self):
        """Reset the drone to initial state."""
        self.obs, self.info = self.env.reset()
        self.motor_thrusts = np.array([self.hover_thrust] * 4, dtype=np.float32)
        self.keys_pressed.clear()
        self.episode_count += 1
        print(f"\n>>> RESET: Starting Episode {self.episode_count} <<<")

    def run(self):
        """Main control loop."""
        # Initial render to create pygame window
        self.env.render(skip_event_handling=True)

        step = 0
        last_info_time = time.time()
        self.last_frame_time = time.time()

        try:
            while self.running:
                frame_start_time = time.time()

                # Handle pygame events (must be called before updating motor thrusts)
                self.handle_pygame_events()

                if not self.running:
                    break

                # Update motor thrusts based on key states
                self.update_motor_thrusts()

                # Step simulation with current motor thrusts
                self.obs, reward, terminated, truncated, self.info = self.env.step(self.motor_thrusts)

                # Check if episode ended
                if terminated or truncated:
                    reason = "terminated" if terminated else "truncated"
                    print(f"\n{'='*60}")
                    print(f"EPISODE {self.episode_count} ENDED ({reason.upper()})!")
                    print(f"  Total steps in episode: {step}")
                    print(f"  Total steps overall: {self.total_steps + step}")
                    if terminated:
                        print(f"  Reason: Episode terminated (likely crash or bounds violation)")
                    else:
                        print(f"  Reason: Episode truncated (max steps reached)")
                    print(f"{'='*60}")

                    # Update total steps
                    self.total_steps += step
                    step = 0

                    # Automatically start new episode
                    print(f"Starting new episode...")
                    self.reset()
                    continue

                # Render visualization (skip event handling since we do it ourselves)
                self.env.render(skip_event_handling=True)

                # Calculate how long to wait for correct FPS
                frame_elapsed = time.time() - frame_start_time
                time_to_wait = self.target_frame_time - frame_elapsed

                if time_to_wait > 0:
                    time.sleep(time_to_wait)

                # FPS tracking
                self.frame_count += 1
                current_time = time.time()

                # Calculate actual FPS every second
                if current_time - self.fps_display_time >= 1.0:
                    self.actual_fps = self.frame_count / (current_time - self.fps_display_time)
                    self.frame_count = 0
                    self.fps_display_time = current_time

                # Display info every 2 seconds
                if current_time - last_info_time > 2.0:
                    pos = self.info['position']
                    roll_deg = np.rad2deg(self.env.drone.get_euler()[0])
                    pitch_deg = np.rad2deg(self.env.drone.get_euler()[1])
                    yaw_deg = np.rad2deg(self.env.drone.get_euler()[2])
                    angular_vel = np.rad2deg(self.env.drone.angular_velocity)
                    vel = self.env.drone.velocity

                    print(f"\nStatus (Step {step}):")
                    print(f"  Position: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
                    print(f"  Roll:  {roll_deg:6.1f}°")
                    print(f"  Pitch: {pitch_deg:6.1f}°")
                    print(f"  Yaw:   {yaw_deg:6.1f}°")
                    print(f"  Velocity: [{vel[0]:6.2f}, {vel[1]:6.2f}, {vel[2]:6.2f}] m/s")
                    print(f"  Angular Velocity: [{angular_vel[0]:6.2f}, {angular_vel[1]:6.2f}, {angular_vel[2]:6.2f}] deg/s")
                    print(f"  Motor Thrusts: [{self.motor_thrusts[0]:.2f}, {self.motor_thrusts[1]:.2f}, {self.motor_thrusts[2]:.2f}, {self.motor_thrusts[3]:.2f}]")
                    print(f"  FPS: {self.actual_fps:.1f} / {self.target_fps:.0f} (target)")

                    last_info_time = current_time

                step += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.env.close()
            print("\nManual control ended.")


if __name__ == "__main__":
    controller = ManualDroneController()
    controller.run()

