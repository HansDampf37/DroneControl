"""Debug script for rendering issues."""
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv
import matplotlib.pyplot as plt

print("=" * 60)
print("RENDERING DEBUG")
print("=" * 60)

print("\n1. Create environment...")
env = DroneEnv(max_steps=50, render_mode="human")
print(f"   env.fig is None: {env.fig is None}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")

print("\n2. Reset environment...")
obs, info = env.reset(seed=42)
print(f"   Position: {info['position']}")
print(f"   Target: {info['target_position']}")
print(f"   env.fig is None: {env.fig is None}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")

print("\n3. First render() call...")
env.render()
print(f"   env.fig is None: {env.fig is None}")
print(f"   env.fig number: {env.fig.number if env.fig else 'None'}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")
print(f"   Number of figures: {len(plt.get_fignums())}")

print("\n4. Second render() call...")
env.step(np.array([0.25, 0.25, 0.25, 0.25]))
env.render()
print(f"   env.fig is None: {env.fig is None}")
print(f"   env.fig number: {env.fig.number if env.fig else 'None'}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")
print(f"   Number of figures: {len(plt.get_fignums())}")

print("\n5. Third render() call...")
env.step(np.array([0.25, 0.25, 0.25, 0.25]))
env.render()
print(f"   env.fig is None: {env.fig is None}")
print(f"   env.fig number: {env.fig.number if env.fig else 'None'}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")
print(f"   Number of figures: {len(plt.get_fignums())}")

print("\n6. Keep window open for 3 seconds...")
import time
time.sleep(3)

print("\n7. Close environment...")
env.close()
print(f"   env.fig is None: {env.fig is None}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")

print("\n" + "=" * 60)
print("DEBUG COMPLETED")
print("=" * 60)
print("\nIf more than 1 figure was created, there's a problem!")
print("The figure should remain the same across all render() calls.")

