#!/usr/bin/env python3
"""
Demo script for the pygame-based 3D renderer.

This script demonstrates the new fast pygame renderer with a simple
manual control interface or random agent.
"""
import numpy as np
from src.drone_env import DroneEnv


def demo_random_agent():
    """Run a random agent with the pygame renderer."""
    print("Starting Pygame 3D Renderer Demo...")
    print("The drone will fly with random actions.")
    print("Close the window to exit.")
    print()

    # Create environment with pygame renderer
    env = DroneEnv(
        render_mode="human",
        renderer_type="pygame",  # Use the new fast renderer
        max_steps=2000,
        use_wind=True,
        wind_strength_range=(0.0, 3.0)
    )

    # Reset environment
    obs, info = env.reset()

    total_reward = 0
    episode_count = 0
    step_count = 0

    try:
        while True:
            # Simple random action with some bias towards hovering
            # This creates more interesting movements than pure random
            hover_thrust = 0.5
            noise = np.random.uniform(-0.3, 0.3, 4)
            action = np.clip(hover_thrust + noise, 0.0, 1.0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Render
            env.render()

            # Reset on episode end
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} finished: {step_count} steps, reward: {total_reward:.2f}")

                # Reset for next episode
                obs, info = env.reset()
                total_reward = 0
                step_count = 0

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        env.close()
        print("Demo ended.")


def demo_hovering_agent():
    """Run a simple hovering agent that tries to maintain altitude."""
    print("Starting Pygame 3D Renderer Demo (Hovering Agent)...")
    print("The drone will attempt to hover and reach targets.")
    print("Close the window to exit.")
    print()

    # Create environment with pygame renderer
    env = DroneEnv(
        render_mode="human",
        renderer_type="pygame",
        max_steps=2000,
        use_wind=True,
        wind_strength_range=(0.0, 2.0),
        target_change_interval=500  # Change target every 500 steps
    )

    # Reset environment
    obs, info = env.reset()

    total_reward = 0
    episode_count = 0
    step_count = 0

    try:
        while True:
            # Simple proportional controller for hovering
            # Extract relative position from observation
            rel_pos = obs[0:3]  # Relative position to target
            velocity = obs[3:6]  # Linear velocity

            # Simple PD-like control
            # Increase thrust if below target, decrease if above
            base_thrust = 0.59  # Approximate hover thrust for 1kg drone

            # Proportional control on height difference
            z_error = -rel_pos[2]  # Negative because we want to reduce distance
            z_control = z_error * 0.5

            # Add some damping based on vertical velocity
            z_damping = -velocity[2] * 0.1

            # Apply same adjustment to all motors (simplified)
            thrust_adjustment = z_control + z_damping
            action = np.clip(base_thrust + thrust_adjustment, 0.0, 1.0)
            action = np.array([action] * 4, dtype=np.float32)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Render
            env.render()

            # Reset on episode end
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} finished: {step_count} steps, reward: {total_reward:.2f}")

                # Reset for next episode
                obs, info = env.reset()
                total_reward = 0
                step_count = 0

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        env.close()
        print("Demo ended.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "hover":
        demo_hovering_agent()
    else:
        demo_random_agent()

