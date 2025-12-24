#!/usr/bin/env python3
"""
Quick visual test to verify the new rendering sizes.
Shows the drone with improved scaling for the 3-meter observation space.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.drone_env import DroneEnv

def test_visual_sizes():
    """Test that visual elements are properly scaled."""
    print("="*70)
    print("VISUAL SCALING TEST")
    print("="*70)
    print("\nRendering with new scaled sizes:")
    print("  • Drone body: 0.08m radius (8cm)")
    print("  • Drone arm length: 0.10m (10cm)")
    print("  • Rotor circles: 0.04m radius (4cm)")
    print("  • Rotor scale: 1.2x")
    print("  • Target circle: 0.15m radius (15cm)")
    print("  • Target crosshair: 0.1m (10cm)")
    print("\nRendering 50 steps with hover control...")
    print("Close the matplotlib window to end the test.\n")

    # Create environment with rendering
    env = DroneEnv(
        max_steps=500,
        render_mode="human",
        dt=0.1,
    )

    # Reset
    obs, info = env.reset(seed=42)

    # Hover action
    hover = np.array([0.25, 0.25, 0.25, 0.25])

    try:
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(hover)
            env.render()

            if step % 10 == 0:
                pos = info['position']
                print(f"Step {step:2d}: Pos=[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] | "
                      f"Dist={info['distance_to_target']:.2f}m | Reward={reward:.3f}")

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()

    print("\n" + "="*70)
    print("Visual test completed!")
    print("="*70)
    print("\nThe rendering should now show:")
    print("  ✓ Smaller drone body proportional to 3m observation space")
    print("  ✓ Smaller rotors with shorter arms (10cm)")
    print("  ✓ Smaller target marker")
    print("  ✓ Red boundary box at ±1.5m")
    print("  ✓ Grid extending to ±1.8m")

if __name__ == "__main__":
    test_visual_sizes()

