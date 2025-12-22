"""Minimaler Rendering Test."""
import sys
from pathlib import Path

# Füge src/ zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv

print("Erstelle Environment...")
env = DroneEnv(max_steps=100, render_mode="human")

print("Reset...")
obs, info = env.reset(seed=42)
print(f"Ziel: {info['target_position']}")

print("\nStarte Rendering-Loop (20 Steps)...")
print("Das Fenster sollte sich öffnen und die Drohne anzeigen.\n")

for step in range(20):
    # Einfache Hover-Action
    action = np.array([0.25, 0.25, 0.25, 0.25])

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step+1:2d}: Pos=[{info['position'][0]:6.2f}, {info['position'][1]:6.2f}, {info['position'][2]:6.2f}]")

    # RENDERING - das ist der kritische Teil
    env.render()

    if terminated or truncated:
        break

print("\nSchließe Environment...")
env.close()
print("Fertig!")

