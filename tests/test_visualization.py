"""Test of the new visualization with rotors and tilt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv

print("Testing new visualization with rotors and tilt...")
print("Close the window with Ctrl+C\n")

env = DroneEnv(max_steps=500, render_mode="human")
obs, info = env.reset(seed=42)

print(f"Target: {info['target_position']}")
print("\nThe visualization shows:")
print("  - Blue center: Drone position")
print("  - 4 circles: Rotors (Red=CW, Green=CCW)")
print("  - Gray lines: Connections to rotors (3D → 2D projected)")
print("  - Orange arrow: Tilt direction (projection of normal)")
print("  - Info box: Including Roll/Pitch/Yaw in degrees")
print("\nThe rotor arms adapt to Roll, Pitch AND Yaw!")
print("Observe how the X-shape changes with tilt.\n")

try:
    for step in range(500):
        # Different maneuvers to show tilt
        if step < 50:
            # Hover - no tilt, symmetric X
            action = np.array([0.25, 0.25, 0.25, 0.25])
        elif step < 100:
            # Strong roll right (arms tilt right)
            action = np.array([0.15, 0.35, 0.35, 0.15])
        elif step < 150:
            # Roll left (arms tilt left)
            action = np.array([0.35, 0.15, 0.15, 0.35])
        elif step < 200:
            # Pitch forward (arms tilt forward)
            action = np.array([0.15, 0.35, 0.15, 0.35])
        elif step < 250:
            # Pitch backward (arms tilt backward)
            action = np.array([0.35, 0.15, 0.35, 0.15])
        elif step < 350:
            # Combined: Roll + Pitch (diagonal tilt)
            action = np.array([0.15, 0.35, 0.25, 0.25])
        elif step < 400:
            # Yaw rotation while hovering (X rotates)
            action = np.array([0.28, 0.28, 0.22, 0.22])
        else:
            # Back to hover
            action = np.array([0.25, 0.25, 0.25, 0.25])

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Info every 50 steps
        if (step + 1) % 50 == 0:
            roll_deg = np.rad2deg(obs[6])
            pitch_deg = np.rad2deg(obs[7])
            yaw_deg = np.rad2deg(obs[8])
            print(f"Step {step+1:3d}: Roll={roll_deg:6.1f}°, Pitch={pitch_deg:6.1f}°, Yaw={yaw_deg:6.1f}°")

        if terminated or truncated:
            break

except KeyboardInterrupt:
    print("\nTest aborted.")

finally:
    env.close()
    print("\nVisualization test completed!")

