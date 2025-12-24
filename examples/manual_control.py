"""
Manual test for drone control.

Controls:
- Keys 1-4: Set/toggle Motor 1-4 to 100%
- Keys Q/W/E/R: Set/toggle Motor 1-4 to 50%
- Key 0: All motors OFF
- Key SPACE: All motors to 25% (hover attempt)
- Key ESC or X: Exit
- Key R: Reset

The visualization shows the drone from above.
Motor configuration (X-formation):
  Motor 2 (front-left, CCW)
         ○
        / \
       /   \
Motor 0 ○   ○ Motor 3
(front-  \   / (rear-
right,   \ /  right,
CW)        ○   CW)
    Motor 1 (rear-left, CW)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv
import matplotlib.pyplot as plt
import time


class ManualDroneController:
    """Interactive controller for manual drone control."""

    def __init__(self):
        # dt=0.1 corresponds to 10 FPS (1/0.01 = 100 Hz)
        self.dt = 0.1  # Timestep in seconds

        self.env = DroneEnv(
            max_steps=10000,
            render_mode="human",
            enable_crash_detection=False,
            dt=self.dt,
        )

        # Motor states
        self.motor_powers = np.array([0.0, 0.0, 0.0, 0.0])
        self.motor_active = np.array([False, False, False, False])

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
        print("MANUAL DRONE CONTROL")
        print("=" * 60)
        print(f"\nSIMULATION PARAMETERS:")
        print(f"  Timestep (dt): {self.dt}s")
        print(f"  Target FPS: {self.target_fps:.0f}")
        print(f"  Frame time: {self.target_frame_time*1000:.1f}ms")
        print("\nCONTROLS:")
        print("  1-4     : Set/toggle Motor 1-4 to 100%")
        print("  Q/W/E/R : Set/toggle Motor 1-4 to 50%")
        print("  0       : All motors OFF")
        print("  SPACE   : All motors to 25% (hover)")
        print("  X or ESC : Exit")
        print("  R       : Reset (position & orientation)")
        print("\nMOTOR CONFIGURATION (X-formation):")
        print("     2(FL)")
        print("      ○")
        print("     / \\")
        print("  0 ○   ○ 3")
        print("   (FR)  (RR)")
        print("     \\ /")
        print("      ○")
        print("     1(RL)")
        print("\nFL=front-left(CCW), FR=front-right(CW)")
        print("RL=rear-left(CW), RR=rear-right(CW)")
        print("=" * 60)
        print("Press keys in the Matplotlib window!\n")

    def on_key_press(self, event):
        """Handle keyboard inputs."""
        key = event.key

        # Motors 1-4 to 100%
        motor_power = 1
        percent = f"{int(motor_power*100)}%"
        if key == '1':
            self.motor_active[0] = not self.motor_active[0]
            self.motor_powers[0] = motor_power if self.motor_active[0] else 0.0
            print(f"Motor 0 (front-right): {'ON (' + percent + f')' if self.motor_active[0] else 'OFF'}")

        elif key == '2':
            self.motor_active[1] = not self.motor_active[1]
            self.motor_powers[1] = motor_power if self.motor_active[1] else 0.0
            print(f"Motor 1 (rear-left): {'ON (' + percent + f')' if self.motor_active[1] else 'OFF'}")

        elif key == '3':
            self.motor_active[2] = not self.motor_active[2]
            self.motor_powers[2] = motor_power if self.motor_active[2] else 0.0
            print(f"Motor 2 (front-left):  {'ON (' + percent + f')' if self.motor_active[2] else 'OFF'}")

        elif key == '4':
            self.motor_active[3] = not self.motor_active[3]
            self.motor_powers[3] = motor_power if self.motor_active[3] else 0.0
            print(f"Motor 3 (rear-right): {'ON (' + percent + f')' if self.motor_active[3] else 'OFF'}")

        # Motors 1-4 to 50%
        elif key == 'q':
            self.motor_active[0] = not self.motor_active[0]
            self.motor_powers[0] = 0.5 if self.motor_active[0] else 0.0
            print(f"Motor 0 (front-right): {'ON (50%)' if self.motor_active[0] else 'OFF'}")

        elif key == 'w':
            self.motor_active[1] = not self.motor_active[1]
            self.motor_powers[1] = 0.5 if self.motor_active[1] else 0.0
            print(f"Motor 1 (rear-left): {'ON (50%)' if self.motor_active[1] else 'OFF'}")

        elif key == 'e':
            self.motor_active[2] = not self.motor_active[2]
            self.motor_powers[2] = 0.5 if self.motor_active[2] else 0.0
            print(f"Motor 2 (front-left):  {'ON (50%)' if self.motor_active[2] else 'OFF'}")

        elif key == 'r':
            # Special case: R can either set Motor 4 to 50% or reset
            # Use Shift+R for reset
            if event.key == 'R':  # Shift+R
                self.reset()
            else:  # Lowercase r
                self.motor_active[3] = not self.motor_active[3]
                self.motor_powers[3] = 0.5 if self.motor_active[3] else 0.0
                print(f"Motor 3 (rear-right): {'ON (50%)' if self.motor_active[3] else 'OFF'}")

        # All motors off
        elif key == '0':
            self.motor_powers[:] = 0.0
            self.motor_active[:] = False
            print("All motors OFF")

        # Hover (all to 25%)
        elif key == ' ':
            self.motor_powers[:] = 0.25
            self.motor_active[:] = True
            print("All motors to 25% (hover attempt)")

        # Reset
        elif key == 'R':  # Shift+R
            self.reset()

        # Exit
        elif key in ['x', 'X', 'escape']:
            print("\nExiting...")
            self.running = False
            plt.close('all')

    def reset(self):
        """Reset the drone to initial state."""
        self.obs, self.info = self.env.reset()
        self.motor_powers[:] = 0.0
        self.motor_active[:] = False
        self.episode_count += 1
        print(f"\n>>> RESET: Starting Episode {self.episode_count} <<<\n")

    def run(self):
        """Main control loop."""
        # Connect keyboard event
        if self.env.renderer.fig is not None:
            self.env.renderer.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initial render to create figure
        self.env.render()

        # Connect after first render
        if self.env.renderer.fig is not None:
            self.env.renderer.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        step = 0
        last_info_time = time.time()
        self.last_frame_time = time.time()

        try:
            while self.running:
                frame_start_time = time.time()

                # Step simulation with current motor powers
                self.obs, reward, terminated, truncated, self.info = self.env.step(self.motor_powers)

                # Check if episode ended (failed)
                if terminated or truncated:
                    reason = "terminated" if terminated else "truncated"
                    print(f"\n{'='*60}")
                    print(f"EPISODE {self.episode_count} FAILED ({reason.upper()})!")
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

                # Render visualization (environment has its own FPS control)
                self.env.render()

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
                    roll_deg = np.rad2deg(self.env.drone.orientation[0])
                    pitch_deg = np.rad2deg(self.env.drone.orientation[1])
                    yaw_deg = np.rad2deg(self.env.drone.orientation[2])
                    angular_vel = np.rad2deg(self.env.drone.angular_velocity)
                    vel = self.env.drone.velocity

                    print(self.obs)

                    print(f"\nStatus (Step {step}):")
                    print(f"  Position: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
                    print(f"  Roll:  {roll_deg:6.1f}°")
                    print(f"  Pitch: {pitch_deg:6.1f}°")
                    print(f"  Yaw:   {yaw_deg:6.1f}°")
                    print(f"  Velocity: [{vel[0]:6.2f}, {vel[1]:6.2f}, {vel[2]:6.2f}]m/s")
                    print(f"  Angular Velocity: [{angular_vel[0]:6.2f}, {angular_vel[1]:6.2f}, {angular_vel[2]:6.2f}]deg/s")
                    print(f"  Motors: [{self.motor_powers[0]:.2f}, {self.motor_powers[1]:.2f}, "
                          f"{self.motor_powers[2]:.2f}, {self.motor_powers[3]:.2f}]")
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

