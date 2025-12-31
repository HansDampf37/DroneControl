#!/usr/bin/env python3
"""
Interactive demo for the pygame renderer with camera controls.

This script demonstrates the new interactive camera controls:
- Mouse: Click and drag to rotate the camera
- Keyboard: WASD to move, Space/Shift for up/down
- Coordinate system indicator in bottom-left corner
"""
import numpy as np
from src.drone_env import DroneEnv


def interactive_demo():
    """Run an interactive demo with camera controls."""
    print("=" * 70)
    print("INTERACTIVE PYGAME RENDERER DEMO")
    print("=" * 70)
    print()
    print("🎮 CONTROLS:")
    print("  🖱️  Mouse:")
    print("      • Click and drag: Rotate camera view")
    print()
    print("  ⌨️  Keyboard:")
    print("      • W: Move camera forward")
    print("      • S: Move camera backward")
    print("      • A: Move camera left")
    print("      • D: Move camera right")
    print("      • Space: Move camera up")
    print("      • Shift: Move camera down")
    print()
    print("  📐 Coordinate System:")
    print("      • Red (X), Green (Y), Blue (Z) axes in bottom-left")
    print("      • Rotates with camera for orientation reference")
    print()
    print("Close the window to exit.")
    print("=" * 70)
    print()

    # Create environment with pygame renderer
    env = DroneEnv(
        render_mode="human",
        renderer_type="pygame",
        max_steps=5000,
        use_wind=True,
        wind_strength_range=(0.0, 2.0),
        target_change_interval=500  # Change target every 500 steps
    )

    # Reset environment
    obs, info = env.reset()

    total_reward = 0
    episode_count = 0
    step_count = 0

    # Simple hovering controller for more stable flight
    base_thrust = 0.59

    try:
        while True:
            # Simple proportional controller
            rel_pos = obs[0:3]
            velocity = obs[3:6]

            # Height control
            z_error = -rel_pos[2]
            z_control = z_error * 0.5
            z_damping = -velocity[2] * 0.1

            thrust_adjustment = z_control + z_damping
            action = np.clip(base_thrust + thrust_adjustment, 0.0, 1.0)
            action = np.array([action] * 4, dtype=np.float32)

            # Add small random perturbations for interesting movement
            action += np.random.uniform(-0.05, 0.05, 4)
            action = np.clip(action, 0.0, 1.0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Render with interactive controls
            env.render()

            # Print camera position every 100 steps
            if step_count % 100 == 0:
                cam_pos = env.renderer.camera_position
                cam_angle_h = env.renderer.camera_angle_h
                cam_angle_v = env.renderer.camera_angle_v
                print(f"Step {step_count}: Camera at ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}), "
                      f"Angles: ({cam_angle_h:.1f}°, {cam_angle_v:.1f}°)")

            # Reset on episode end
            if terminated or truncated:
                episode_count += 1
                print(f"\nEpisode {episode_count} finished: {step_count} steps, reward: {total_reward:.2f}")

                # Reset for next episode
                obs, info = env.reset()
                total_reward = 0
                step_count = 0

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    finally:
        env.close()
        print("Demo ended.")


if __name__ == "__main__":
    interactive_demo()

