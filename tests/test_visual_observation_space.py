#!/usr/bin/env python3
"""
Visual test to verify observation space boundary rendering.
This will open a matplotlib window showing the drone with the observation space boundary.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.drone_env import DroneEnv
import time

def test_visual_boundaries():
    """Test that boundary box is visible in the renderer."""
    print("Testing visual observation space boundaries...")
    print("This will open a matplotlib window.")
    print("The red dashed box shows the observation space limits.")
    print("Press Ctrl+C in terminal to exit.\n")

    # Create environment with rendering enabled
    env = DroneEnv(
        max_steps=500,
        render_mode="human",
        enable_crash_detection=True,
        dt=0.1,
    )

    print(f"Observation space: {env.space_side_length}m cube")
    print(f"Grid limits: ±{env.renderer.grid_limit:.2f}m")
    print(f"Boundary: ±{env.space_side_length/2}m\n")

    # Reset environment
    obs, info = env.reset()

    # Hover action to keep drone stable
    hover_action = np.array([0.25, 0.25, 0.25, 0.25])

    try:
        for step in range(500):
            obs, reward, terminated, truncated, info = env.step(hover_action)
            env.render()

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                if info.get('crashed'):
                    print("  Reason: Crash detected")
                elif info.get('out_of_bounds'):
                    print("  Reason: Out of bounds")
                    print(f"  Position: {info['position']}")
                else:
                    print("  Reason: Max steps")
                break

            time.sleep(0.05)  # Slow down for visualization

            # Print status every 50 steps
            if step % 50 == 0:
                pos = info['position']
                print(f"Step {step:3d}: Position [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] | "
                      f"Distance: {info['distance_to_target']:.2f}m | Reward: {reward:.3f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        env.close()
        print("\nVisual test completed.")

if __name__ == "__main__":
    test_visual_boundaries()

