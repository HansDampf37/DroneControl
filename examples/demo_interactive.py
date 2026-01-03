#!/usr/bin/env python3
"""
Interactive Demo for Drone Environment

This script provides a comprehensive demonstration of the drone environment with
multiple modes:
1. Manual Control - Fly the drone manually using keyboard controls
2. Random Agent - Watch a random agent fly the drone
3. Hovering Agent - Watch a simple proportional controller try to reach targets

Features:
- Interactive 3D visualization with Pygame renderer
- Camera controls (mouse drag to rotate, WASD to move)
- Real-time drone dynamics and physics
- Visual feedback with coordinate axes and drone orientation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv
import pygame
import time
import argparse


class InteractiveDroneDemo:
    """Interactive demo controller for the drone environment."""

    def __init__(self, mode='manual', dt=0.01, use_wind=False):
        """
        Initialize the interactive demo.

        Args:
            mode: Demo mode - 'manual', 'random', or 'hover'
            dt: Simulation timestep in seconds (default: 0.01 = 100 Hz)
            use_wind: Whether to enable wind simulation
        """
        self.mode = mode
        self.dt = dt

        # Create environment with pygame renderer
        self.env = DroneEnv(
            max_steps=10000,
            render_mode="human",
            renderer_type="pygame",
            enable_crash_detection=(mode == 'manual'),
            dt=dt,
            use_wind=use_wind,
            wind_strength_range=(0.0, 3.0) if use_wind else (0.0, 0.0),
            target_change_interval=500 if mode != 'manual' else None
        )

        # Mode-specific initialization
        if mode == 'manual':
            self._init_manual_mode()
        elif mode == 'hover':
            self.base_thrust = 0.59  # Hover thrust for 1kg drone

        # Simulation state
        self.running = True
        self.paused = False
        self.episode_count = 0
        self.total_reward = 0
        self.step_count = 0

        # FPS tracking
        self.target_fps = 1.0 / self.dt
        self.target_frame_time = self.dt
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_display_time = time.time()
        self.actual_fps = 0.0

        # Reset environment
        self.obs, self.info = self.env.reset()
        self.episode_count = 1

        self._print_instructions()

        # Initialize pygame by rendering once
        # This ensures pygame is initialized before we start handling events
        self.env.render()

    def _init_manual_mode(self):
        """Initialize manual control mode."""
        self.hover_thrust = 0.495  # Approximate hover thrust
        self.motor_thrusts = np.array([self.hover_thrust] * 4, dtype=np.float32)
        self.thrust_change_rate = 0.5  # Change by 50% per second
        self.keys_pressed = set()

    def _print_instructions(self):
        """Print mode-specific instructions."""
        print("=" * 70)
        print(f"DRONE ENVIRONMENT INTERACTIVE DEMO - {self.mode.upper()} MODE")
        print("=" * 70)
        print(f"\n📊 SIMULATION PARAMETERS:")
        print(f"  • Timestep (dt): {self.dt}s ({1/self.dt:.0f} Hz)")
        print(f"  • Target FPS: {self.target_fps:.0f}")
        print(f"  • Wind: {'Enabled' if self.env.wind.enabled else 'Disabled'}")

        if self.mode == 'manual':
            print(f"  • Hover thrust: {self.hover_thrust:.3f}")
            print("\n🎮 MANUAL CONTROLS:")
            print("  Motor Control:")
            print("    1-4       : Increase thrust for Motor 1-4 (hold)")
            print("    5-8       : Decrease thrust for Motor 1-4 (hold)")
            print("    Release   : Return to hover thrust")
            print("\n  Motor Layout (X-configuration):")
            print("       2 (FL,CCW)")
            print("          ○")
            print("         / \\")
            print("      0 ○   ○ 3")
            print("     (FR,CW) (RR,CW)")
            print("         \\ /")
            print("          ○")
            print("       1 (RL,CCW)")
            print("    FL=Front-Left, FR=Front-Right")
            print("    RL=Rear-Left, RR=Rear-Right")

        print("\n📹 CAMERA CONTROLS:")
        print("  Mouse:")
        print("    • Click & Drag : Rotate camera view")
        print("  Keyboard:")
        print("    • W/↑         : Move camera forward")
        print("    • S/↓         : Move camera backward")
        print("    • A/←         : Move camera left")
        print("    • D/→         : Move camera right")
        print("    • Space       : Move camera up")
        print("    • Shift       : Move camera down")

        print("\n⌨️  OTHER CONTROLS:")
        print("    • R           : Reset episode")
        print("    • ESC/X       : Exit")

        print("\n📐 VISUALIZATION:")
        print("  • Coordinate axes (RGB = XYZ) in bottom-left corner")
        print("  • Green sphere: Target position")
        print("  • Drone: Central sphere + 4 motor spheres")
        print("  • Blue arrow: Drone orientation (normal vector)")
        print("  • Yellow arrow: Velocity vector")
        print("=" * 70)
        print()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n👋 Exiting...")
                self.running = False
                return

            # Forward mouse events to renderer
            elif event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION]:
                self.env.renderer._handle_mouse_input(event)

            elif event.type == pygame.KEYDOWN:
                if self.mode == 'manual':
                    key_name = pygame.key.name(event.key)
                    self.keys_pressed.add(key_name)

                if event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_x:
                    print("\n👋 Exiting...")
                    self.running = False

            elif event.type == pygame.KEYUP:
                if self.mode == 'manual':
                    key_name = pygame.key.name(event.key)
                    self.keys_pressed.discard(key_name)

        # Handle camera keyboard input
        self.env.renderer._handle_keyboard_input()

    def get_action(self):
        """Get action based on current mode."""
        if self.mode == 'manual':
            return self._get_manual_action()
        elif self.mode == 'random':
            return self._get_random_action()
        elif self.mode == 'hover':
            return self._get_hover_action()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_manual_action(self):
        """Get action from manual keyboard control."""
        # Calculate thrust change for this frame
        thrust_delta = self.thrust_change_rate * self.dt
        thrust_changes = np.zeros(4, dtype=np.float32)

        # Keys 1-4: Increase thrust
        for i in range(4):
            if str(i + 1) in self.keys_pressed:
                thrust_changes[i] = thrust_delta
            elif str(i + 5) in self.keys_pressed:
                thrust_changes[i] = -thrust_delta

        # Apply changes
        self.motor_thrusts += thrust_changes
        self.motor_thrusts = np.clip(self.motor_thrusts, 0.0, 1.0)

        # Return to hover when no keys pressed
        hover_return_rate = 2.0
        for i in range(4):
            key_increase = str(i + 1) in self.keys_pressed
            key_decrease = str(i + 5) in self.keys_pressed
            if not key_increase and not key_decrease:
                diff = self.hover_thrust - self.motor_thrusts[i]
                self.motor_thrusts[i] += diff * hover_return_rate * self.dt

        return self.motor_thrusts.copy()

    def _get_random_action(self):
        """Get random action with hover bias."""
        hover_thrust = 0.5
        noise = np.random.uniform(-0.3, 0.3, 4)
        return np.clip(hover_thrust + noise, 0.0, 1.0)

    def _get_hover_action(self):
        """Get action from simple proportional controller."""
        rel_pos = self.obs[0:3]  # Relative position to target
        velocity = self.obs[3:6]  # Linear velocity

        # Simple PD-like control for vertical position
        z_error = -rel_pos[2]
        z_control = z_error * 0.5
        z_damping = -velocity[2] * 0.1

        thrust_adjustment = z_control + z_damping
        action = np.clip(self.base_thrust + thrust_adjustment, 0.0, 1.0)
        action = np.array([action] * 4, dtype=np.float32)

        # Add small random perturbations
        action += np.random.uniform(-0.05, 0.05, 4)
        return np.clip(action, 0.0, 1.0)

    def reset(self):
        """Reset the environment."""
        print(f"\n🔄 Episode {self.episode_count} Statistics:")
        print(f"  Steps: {self.step_count}")
        print(f"  Total Reward: {self.total_reward:.2f}")
        print(f"  Avg Reward/Step: {self.total_reward/max(1, self.step_count):.4f}")
        print(f"  Duration: {self.step_count * self.dt:.2f}s")

        if self.mode == 'manual':
            self.motor_thrusts = np.array([self.hover_thrust] * 4, dtype=np.float32)
            self.keys_pressed.clear()

        self.obs, self.info = self.env.reset()
        self.episode_count += 1
        self.total_reward = 0
        self.step_count = 0

        print(f"\n✨ Episode {self.episode_count} Started")
        print(f"  Target: {self.info['target_position']}")
        print(f"  Initial Distance: {self.info['distance_to_target']:.2f}m")

    def run(self):
        """Main loop."""
        try:
            while self.running:
                frame_start = time.time()

                # Handle events
                self.handle_events()

                if not self.running:
                    break

                # Get action and step
                action = self.get_action()
                self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                self.total_reward += reward
                self.step_count += 1

                # Render
                self.env.render()

                # FPS tracking
                self.frame_count += 1
                if time.time() - self.fps_display_time >= 1.0:
                    self.actual_fps = self.frame_count
                    self.frame_count = 0
                    self.fps_display_time = time.time()

                # Print status every 100 steps
                if self.step_count % 100 == 0:
                    print(f"Step {self.step_count:4d} | "
                          f"Reward: {reward:6.3f} | "
                          f"Distance: {self.info['distance_to_target']:5.2f}m | "
                          f"FPS: {self.actual_fps:3.0f}")

                # Reset on episode end
                if terminated or truncated:
                    reason = "Terminated" if terminated else "Truncated"
                    print(f"\n⚠️  Episode ended: {reason}")
                    if 'crashed' in self.info and self.info['crashed']:
                        print("  Reason: Crash detected")
                    if 'out_of_bounds' in self.info and self.info['out_of_bounds']:
                        print("  Reason: Out of bounds")
                    self.reset()

                # Frame timing
                frame_time = time.time() - frame_start
                sleep_time = max(0, self.target_frame_time - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                self.last_frame_time = time.time()

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        except pygame.error as e:
            print(f"\n⚠️  Pygame error: {e}")
        finally:
            try:
                self.env.close()
            except:
                pass  # Ignore errors during cleanup
            print("\n✅ Demo ended. Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive demo for the drone environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Manual control mode
  python demo_interactive.py --mode manual
  
  # Random agent with wind
  python demo_interactive.py --mode random --wind
  
  # Hovering controller at 60 FPS
  python demo_interactive.py --mode hover --fps 60
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['manual', 'random', 'hover'],
        default='manual',
        help='Demo mode (default: manual)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=100,
        help='Target simulation FPS (default: 100)'
    )
    parser.add_argument(
        '--wind',
        action='store_true',
        help='Enable wind simulation'
    )

    args = parser.parse_args()

    dt = 1.0 / args.fps
    demo = InteractiveDroneDemo(mode=args.mode, dt=dt, use_wind=args.wind)
    demo.run()


if __name__ == '__main__':
    main()

