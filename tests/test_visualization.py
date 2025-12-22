"""Test der neuen Visualisierung mit Rotoren und Neigung."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv

print("Teste neue Visualisierung mit Rotoren und Neigung...")
print("Schließe das Fenster mit Ctrl+C\n")

env = DroneEnv(max_steps=500, render_mode="human")
obs, info = env.reset(seed=42)

print(f"Ziel: {info['target_position']}")
print("\nDie Visualisierung zeigt:")
print("  - Blaues Zentrum: Drohnen-Position")
print("  - 4 Kreise: Rotoren (Rot=CW, Grün=CCW)")
print("  - Graue Linien: Verbindungen zu Rotoren (3D → 2D projiziert)")
print("  - Oranger Pfeil: Neigungsrichtung (Projektion der Normalen)")
print("  - Info-Box: Inkl. Roll/Pitch/Yaw in Grad")
print("\nDie Rotor-Arme passen sich an Roll, Pitch UND Yaw an!")
print("Beobachte wie sich die X-Form verändert bei Neigung.\n")

try:
    for step in range(500):
        # Verschiedene Manöver um Neigung zu zeigen
        if step < 50:
            # Hover - keine Neigung, symmetrisches X
            action = np.array([0.25, 0.25, 0.25, 0.25])
        elif step < 100:
            # Starker Roll rechts (Arme kippen nach rechts)
            action = np.array([0.15, 0.35, 0.35, 0.15])
        elif step < 150:
            # Roll links (Arme kippen nach links)
            action = np.array([0.35, 0.15, 0.15, 0.35])
        elif step < 200:
            # Pitch vorwärts (Arme kippen nach vorne)
            action = np.array([0.15, 0.35, 0.15, 0.35])
        elif step < 250:
            # Pitch rückwärts (Arme kippen nach hinten)
            action = np.array([0.35, 0.15, 0.35, 0.15])
        elif step < 350:
            # Kombiniert: Roll + Pitch (diagonale Neigung)
            action = np.array([0.15, 0.35, 0.25, 0.25])
        elif step < 400:
            # Yaw-Rotation bei Hover (X dreht sich)
            action = np.array([0.28, 0.28, 0.22, 0.22])
        else:
            # Zurück zu Hover
            action = np.array([0.25, 0.25, 0.25, 0.25])

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Info alle 50 Steps
        if (step + 1) % 50 == 0:
            roll_deg = np.rad2deg(obs[6])
            pitch_deg = np.rad2deg(obs[7])
            yaw_deg = np.rad2deg(obs[8])
            print(f"Step {step+1:3d}: Roll={roll_deg:6.1f}°, Pitch={pitch_deg:6.1f}°, Yaw={yaw_deg:6.1f}°")

        if terminated or truncated:
            break

except KeyboardInterrupt:
    print("\nTest abgebrochen.")

finally:
    env.close()
    print("\nVisualisierungs-Test abgeschlossen!")

