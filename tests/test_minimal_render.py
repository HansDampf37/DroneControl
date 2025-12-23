"""Minimal rendering test."""
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv

print("Creating environment...")
env = DroneEnv(max_steps=100, render_mode="human")

print("Reset...")
obs, info = env.reset(seed=42)
print(f"Target: {info['target_position']}")

print("\nStarting rendering loop (20 steps)...")
print("The window should open and display the drone.\n")

for step in range(20):
    # Simple hover action
    action = np.array([0.25, 0.25, 0.25, 0.25])

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step+1:2d}: Pos=[{info['position'][0]:6.2f}, {info['position'][1]:6.2f}, {info['position'][2]:6.2f}]")

    # RENDERING - this is the critical part
    env.render()

    if terminated or truncated:
        break

print("\nClosing environment...")
env.close()
print("Done!")

