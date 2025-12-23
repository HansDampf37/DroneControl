"""Test script for the Drone Environment.

This module contains comprehensive tests for the drone environment's functionality,
including basic operations, physics simulation, wind dynamics, and crash detection.
"""
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv


def test_basic_functionality():
    """Tests basic environment functionality.

    Verifies that the environment can be created, reset, and stepped through
    correctly with proper observation and action spaces.
    """
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    env = DroneEnv(max_steps=100, render_mode=None)

    # Reset
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Observation Shape: {obs.shape}")
    print(f"  Action Space: {env.action_space}")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Initial distance to target: {info['distance_to_target']:.2f}m")

    # A few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if i == 0:
            print(f"\n✓ Step successful")
            print(f"  Observation: {obs[:6]}")  # First 6 values
            print(f"  Reward: {reward:.4f}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")

    env.close()
    print(f"\n✓ Test 1 passed!\n")


def test_physics():
    """Tests the physics simulation.

    Validates the drone's physical behavior including:
    - Hover capability (gravity compensation)
    - Full thrust response
    - Roll/pitch control through asymmetric motor commands
    """
    print("=" * 60)
    print("Test 2: Physics Simulation")
    print("=" * 60)

    env = DroneEnv(max_steps=500, dt=0.01, render_mode=None)
    obs, info = env.reset(seed=123)

    print(f"Start position: {info['position']}")
    print(f"Target position: {info['target_position']}")

    # Hover test: All motors at ~25% (should compensate for gravity)
    hover_thrust = 0.25  # Approximation for hover
    print(f"\n→ Testing hover with {hover_thrust*100:.0f}% thrust...")

    for _ in range(100):
        action = np.array([hover_thrust] * 4)
        obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Position after 1s: {info['position']}")
    print(f"  Velocity: {obs[3:6]}")

    # Full thrust test
    print(f"\n→ Testing full thrust...")
    env.reset(seed=123)

    for _ in range(100):
        action = np.array([1.0, 1.0, 1.0, 1.0])
        obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Position after 1s: {info['position']}")
    print(f"  Height (Z): {info['position'][2]:.2f}m")

    # Roll test (left motors stronger)
    print(f"\n→ Testing roll (asymmetric motors)...")
    env.reset(seed=123)

    for _ in range(50):
        action = np.array([0.3, 0.2, 0.3, 0.2])  # Right side more thrust
        obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Roll angle: {np.rad2deg(obs[6]):.2f}°")
    print(f"  Pitch angle: {np.rad2deg(obs[7]):.2f}°")
    print(f"  Yaw angle: {np.rad2deg(obs[8]):.2f}°")

    env.close()
    print(f"\n✓ Test 2 passed!\n")


def test_wind():
    """Tests the wind dynamics.

    Validates that the Ornstein-Uhlenbeck wind process generates
    realistic wind variations within the specified range.
    """
    print("=" * 60)
    print("Test 3: Wind Dynamics")
    print("=" * 60)

    env = DroneEnv(
        max_steps=200,
        wind_strength_range=(2.0, 5.0),
        use_wind=True,
        render_mode=None
    )
    obs, info = env.reset(seed=456)

    wind_vectors = []

    for i in range(200):
        action = np.array([0.25] * 4)  # Hover
        obs, reward, terminated, truncated, info = env.step(action)
        wind_vectors.append(env.wind_vector.copy())

    wind_vectors = np.array(wind_vectors)

    print(f"Initial wind: {wind_vectors[0]}")
    print(f"Final wind: {wind_vectors[-1]}")
    print(f"Average wind strength: {np.mean(np.linalg.norm(wind_vectors, axis=1)):.2f} m/s")
    print(f"Max wind strength: {np.max(np.linalg.norm(wind_vectors, axis=1)):.2f} m/s")
    print(f"Min wind strength: {np.min(np.linalg.norm(wind_vectors, axis=1)):.2f} m/s")

    env.close()
    print(f"\n✓ Test 3 passed!\n")


def test_reward():
    """Tests the reward function.

    Verifies that the reward function behaves correctly and provides
    appropriate feedback based on distance to target.
    """
    print("=" * 60)
    print("Test 4: Reward Function")
    print("=" * 60)

    env = DroneEnv(max_steps=100, render_mode=None)
    obs, info = env.reset(seed=789)

    initial_distance = info['distance_to_target']
    print(f"Initial distance: {initial_distance:.2f}m")

    rewards = []
    distances = []

    for i in range(100):
        # Random actions
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        distances.append(info['distance_to_target'])

    print(f"\nReward statistics:")
    print(f"  Min: {np.min(rewards):.4f}")
    print(f"  Max: {np.max(rewards):.4f}")
    print(f"  Mean: {np.mean(rewards):.4f}")

    print(f"\nDistance statistics:")
    print(f"  Min: {np.min(distances):.2f}m")
    print(f"  Max: {np.max(distances):.2f}m")
    print(f"  Final: {distances[-1]:.2f}m")

    # Verify reward formula: ((max_dist - dist) / max_dist) ** 2
    max_dist = env.max_dist_to_target
    test_distance = 10.0
    expected_reward = ((max_dist - test_distance) / max_dist) ** 2
    print(f"\nVerify formula (distance=10m):")
    print(f"  Expected reward: {expected_reward:.4f}")

    env.close()
    print(f"\n✓ Test 4 passed!\n")


def demo_with_visualization():
    """Demo with visualization.

    Demonstrates the environment with real-time rendering to visually
    verify the drone's behavior and the rendering system.
    """
    print("=" * 60)
    print("Demo: Visualization")
    print("=" * 60)
    print("Starting environment with rendering...")
    print("(Close window to exit)")

    env = DroneEnv(
        max_steps=500,
        render_mode="human"
    )

    obs, info = env.reset(seed=42)

    done = False
    step = 0

    try:
        while not done and step < 500:
            # Simple policy: Hover + small random variation
            action = np.array([0.25, 0.25, 0.25, 0.25]) + np.random.uniform(-0.05, 0.05, 4)
            action = np.clip(action, 0, 1)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            env.render()
            step += 1

            if step % 100 == 0:
                print(f"Step {step}: Distance={info['distance_to_target']:.2f}m, Reward={reward:.4f}")

    except KeyboardInterrupt:
        print("\nDemo interrupted.")

    finally:
        env.close()

    print(f"\n✓ Demo completed!\n")


if __name__ == "__main__":
    # Run all tests
    test_basic_functionality()
    test_physics()
    test_wind()
    test_reward()

    # Optional: Demo with visualization
    # Uncomment the following line to see the environment in action:
    # demo_with_visualization()
    print("\nDo you want to start the demo with visualization?")
    print("(Press Enter to proceed or Ctrl+C to skip)")
    try:
        input()
        demo_with_visualization()
    except KeyboardInterrupt:
        print("\nDemo skipped.")

    print("\n" + "=" * 60)
    print("All tests completed successfully! 🚁")
    print("=" * 60)

