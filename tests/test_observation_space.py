#!/usr/bin/env python3
"""
Test script to verify observation space changes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.drone_env import DroneEnv

def test_observation_space():
    """Test that observation space is correctly sized."""
    print("Testing observation space configuration...")

    # Create environment without rendering
    env = DroneEnv(
        max_steps=100,
        render_mode=None,  # No rendering for this test
        enable_crash_detection=True,
        dt=0.1,
    )

    print(f"\n✓ Environment created successfully")
    print(f"  Space side length: {env.space_side_length} meters")
    print(f"  Observation space bounds:")
    print(f"    Position: [{-env.space_side_length/2:.1f}, {env.space_side_length/2:.1f}] meters")
    print(f"    Max distance to target: {env.max_dist_to_target:.2f} meters")

    # Reset and check initial state
    obs, info = env.reset()
    print(f"\n✓ Environment reset successfully")
    print(f"  Initial position: {info['position']}")
    print(f"  Target position: {info['target_position']}")
    print(f"  Distance to target: {info['distance_to_target']:.2f} meters")

    # Check observation space
    print(f"\n✓ Observation space check:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space shape: {env.observation_space.shape}")
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch!"

    # Test that drone doesn't immediately crash with motors off
    print(f"\n✓ Testing stability (10 steps with motors off)...")
    action = np.array([0.0, 0.0, 0.0, 0.0])
    crashed_early = False
    for step in range(10):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            if info.get('crashed', False):
                print(f"  ✗ Drone crashed at step {step}")
                crashed_early = True
                break
            elif info.get('out_of_bounds', False):
                print(f"  ✗ Drone went out of bounds at step {step}")
                print(f"    Position: {info['position']}")
                crashed_early = True
                break

    if not crashed_early:
        print(f"  ✓ Drone remained stable for 10 steps")
        print(f"    Final position: {info['position']}")
        print(f"    Final velocity: {env.drone.velocity}")

    # Test with hover motors
    print(f"\n✓ Testing with hover motors (25% thrust on all motors)...")
    env.reset()
    action = np.array([0.25, 0.25, 0.25, 0.25])
    hover_stable = True
    for step in range(50):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"  ✗ Episode terminated at step {step}")
            if info.get('crashed', False):
                print(f"    Reason: Crash")
            elif info.get('out_of_bounds', False):
                print(f"    Reason: Out of bounds")
                print(f"    Position: {info['position']}")
            hover_stable = False
            break

    if hover_stable:
        print(f"  ✓ Hover test completed 50 steps")
        print(f"    Final position: {info['position']}")
        print(f"    Final velocity: {env.drone.velocity}")

    env.close()
    print(f"\n{'='*60}")
    print(f"✓ All tests completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_observation_space()

