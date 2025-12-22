"""Debug-Skript für Rendering-Probleme."""
import sys
from pathlib import Path

# Füge src/ zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv
import matplotlib.pyplot as plt

print("=" * 60)
print("RENDERING DEBUG")
print("=" * 60)

print("\n1. Erstelle Environment...")
env = DroneEnv(max_steps=50, render_mode="human")
print(f"   env.fig is None: {env.fig is None}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")

print("\n2. Reset Environment...")
obs, info = env.reset(seed=42)
print(f"   Position: {info['position']}")
print(f"   Ziel: {info['target_position']}")
print(f"   env.fig is None: {env.fig is None}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")

print("\n3. Erster render() Aufruf...")
env.render()
print(f"   env.fig is None: {env.fig is None}")
print(f"   env.fig number: {env.fig.number if env.fig else 'None'}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")
print(f"   Anzahl Figures: {len(plt.get_fignums())}")

print("\n4. Zweiter render() Aufruf...")
env.step(np.array([0.25, 0.25, 0.25, 0.25]))
env.render()
print(f"   env.fig is None: {env.fig is None}")
print(f"   env.fig number: {env.fig.number if env.fig else 'None'}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")
print(f"   Anzahl Figures: {len(plt.get_fignums())}")

print("\n5. Dritter render() Aufruf...")
env.step(np.array([0.25, 0.25, 0.25, 0.25]))
env.render()
print(f"   env.fig is None: {env.fig is None}")
print(f"   env.fig number: {env.fig.number if env.fig else 'None'}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")
print(f"   Anzahl Figures: {len(plt.get_fignums())}")

print("\n6. Lasse das Fenster 3 Sekunden offen...")
import time
time.sleep(3)

print("\n7. Close Environment...")
env.close()
print(f"   env.fig is None: {env.fig is None}")
print(f"   plt.get_fignums(): {plt.get_fignums()}")

print("\n" + "=" * 60)
print("DEBUG ABGESCHLOSSEN")
print("=" * 60)
print("\nWenn mehr als 1 Figure erstellt wurde, gibt es ein Problem!")
print("Die Figure sollte während aller render() Aufrufe gleich bleiben.")

