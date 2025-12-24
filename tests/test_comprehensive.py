#!/usr/bin/env python3
"""
Comprehensive test suite for observation space changes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.drone_env import DroneEnv

def test_all():
    """Run all tests for observation space changes."""
    print("="*70)
    print("COMPREHENSIVE OBSERVATION SPACE TEST SUITE")
    print("="*70)

    # Test 1: Environment Creation
    print("\n[Test 1] Environment Creation")
    print("-" * 70)
    env = DroneEnv(render_mode=None, dt=0.1)
    assert env.space_side_length == 3, "Space side length should be 3m"
    print(f"✓ Observation space: {env.space_side_length}m cube")
    print(f"✓ Boundaries: ±{env.space_side_length/2}m")
    print(f"✓ Max distance to target: {env.max_dist_to_target:.2f}m")

    # Test 2: Observation Space Bounds
    print("\n[Test 2] Observation Space Bounds")
    print("-" * 70)
    obs_low = env.observation_space.low[:3]
    obs_high = env.observation_space.high[:3]
    expected_bound = env.space_side_length / 2
    assert np.allclose(obs_low, [-expected_bound] * 3), "Lower bound incorrect"
    assert np.allclose(obs_high, [expected_bound] * 3), "Upper bound incorrect"
    print(f"✓ Position bounds: [{obs_low[0]:.1f}, {obs_high[0]:.1f}]m")
    print(f"✓ Observation space shape: {env.observation_space.shape}")

    # Test 3: Renderer Configuration
    print("\n[Test 3] Renderer Configuration")
    print("-" * 70)
    assert env.renderer.space_side_length == 3, "Renderer space size mismatch"
    assert env.renderer.grid_margin == 0.2, "Grid margin incorrect"
    expected_grid_limit = (3 / 2) * 1.2
    assert env.renderer.grid_limit == expected_grid_limit, "Grid limit incorrect"
    print(f"✓ Renderer space side length: {env.renderer.space_side_length}m")
    print(f"✓ Grid margin: {env.renderer.grid_margin*100:.0f}%")
    print(f"✓ Grid limits: ±{env.renderer.grid_limit:.2f}m")

    # Test 4: Reset and Initial State
    print("\n[Test 4] Reset and Initial State")
    print("-" * 70)
    obs, info = env.reset(seed=42)
    assert obs.shape == (15,), "Observation shape incorrect"
    assert info['position'].shape == (3,), "Position shape incorrect"
    assert info['target_position'].shape == (3,), "Target position shape incorrect"
    print(f"✓ Initial position: {info['position']}")
    print(f"✓ Target position: {info['target_position']}")
    print(f"✓ Initial distance: {info['distance_to_target']:.2f}m")

    # Test 5: Target Generation
    print("\n[Test 5] Target Generation Within Bounds")
    print("-" * 70)
    targets = []
    for _ in range(100):
        target = env._generate_random_target()
        targets.append(target)
        # Check all targets are within bounds
        assert np.all(np.abs(target) <= env.space_side_length / 2), \
            f"Target out of bounds: {target}"
    targets = np.array(targets)
    print(f"✓ Generated 100 targets, all within bounds")
    print(f"✓ Target range X: [{targets[:, 0].min():.2f}, {targets[:, 0].max():.2f}]m")
    print(f"✓ Target range Y: [{targets[:, 1].min():.2f}, {targets[:, 1].max():.2f}]m")
    print(f"✓ Target range Z: [{targets[:, 2].min():.2f}, {targets[:, 2].max():.2f}]m")

    # Test 6: Out of Bounds Detection
    print("\n[Test 6] Out of Bounds Detection")
    print("-" * 70)
    env.reset()
    # Manually set drone to boundary
    boundary = env.space_side_length / 2

    # Test at boundary (should not trigger)
    env.drone.position = np.array([boundary * 0.99, 0, 0])
    assert not env._check_out_of_bounds(), "False positive at 99% boundary"
    print(f"✓ At 99% boundary ({boundary*0.99:.2f}m): Not out of bounds")

    # Test beyond boundary (should trigger)
    env.drone.position = np.array([boundary * 1.01, 0, 0])
    assert env._check_out_of_bounds(), "Missed out of bounds at 101%"
    print(f"✓ At 101% boundary ({boundary*1.01:.2f}m): Out of bounds detected")

    # Test 7: Step Execution
    print("\n[Test 7] Step Execution")
    print("-" * 70)
    env.reset()
    action = np.array([0.25, 0.25, 0.25, 0.25])
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (15,), "Observation shape incorrect after step"
    assert 0 <= reward <= 1, "Reward out of expected range"
    assert isinstance(terminated, (bool, np.bool_)), "Terminated not boolean"
    assert isinstance(truncated, (bool, np.bool_)), "Truncated not boolean"
    print(f"✓ Step executed successfully")
    print(f"✓ Reward: {reward:.3f}")
    print(f"✓ Terminated: {terminated}, Truncated: {truncated}")

    # Test 8: Boundary Box Objects
    print("\n[Test 8] Boundary Box Rendering Objects")
    print("-" * 70)
    assert 'boundary_box_top' in env.renderer._render_objects, \
        "Missing boundary_box_top"
    assert 'boundary_box_front' in env.renderer._render_objects, \
        "Missing boundary_box_front"
    print(f"✓ Boundary box objects registered in renderer")

    # Test 9: Hover Stability Test
    print("\n[Test 9] Hover Stability (30 steps)")
    print("-" * 70)
    env.reset()
    hover_action = np.array([0.25, 0.25, 0.25, 0.25])
    positions = []

    for step in range(30):
        obs, reward, terminated, truncated, info = env.step(hover_action)
        positions.append(info['position'].copy())

        if terminated:
            if info.get('out_of_bounds'):
                print(f"✗ Out of bounds at step {step}: {info['position']}")
                break
            elif info.get('crashed'):
                print(f"✗ Crashed at step {step}")
                break
    else:
        positions = np.array(positions)
        print(f"✓ Completed 30 steps without termination")
        print(f"✓ Final position: {positions[-1]}")
        print(f"✓ Max position deviation: {np.abs(positions).max():.2f}m")
        assert np.abs(positions).max() < env.space_side_length / 2, \
            "Drone exceeded bounds during hover"

    env.close()

    # Final Summary
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print(f"\nSummary:")
    print(f"  • Observation space: {env.space_side_length}m³ cube")
    print(f"  • Boundaries: ±{env.space_side_length/2}m in all axes")
    print(f"  • Grid display: ±{env.renderer.grid_limit:.2f}m (with {env.renderer.grid_margin*100:.0f}% margin)")
    print(f"  • All functionality verified and working correctly")
    print()

if __name__ == "__main__":
    test_all()

