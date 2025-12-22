"""Schneller Test für Rendering-Probleme."""
import sys
from pathlib import Path

# Füge src/ zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv


def test_rendering():
    """Testet das Rendering mit verschiedenen Aktionen."""
    print("Teste Rendering...")
    print("Schließe das Fenster oder drücke Ctrl+C zum Beenden\n")

    env = DroneEnv(max_steps=200, render_mode="human")

    obs, info = env.reset(seed=42)
    print(f"Start-Position: {info['position']}")
    print(f"Ziel-Position: {info['target_position']}")
    print(f"Initiale Distanz: {info['distance_to_target']:.2f}m\n")

    try:
        for step in range(200):
            # Wechsle zwischen verschiedenen Aktionen für sichtbare Bewegung
            if step < 50:
                # Hover
                action = np.array([0.25, 0.25, 0.25, 0.25])
            elif step < 100:
                # Leichte Rotation (mehr Yaw)
                action = np.array([0.3, 0.2, 0.2, 0.3])
            elif step < 150:
                # Vorwärts kippen
                action = np.array([0.3, 0.3, 0.2, 0.2])
            else:
                # Zurück zu Hover
                action = np.array([0.25, 0.25, 0.25, 0.25])

            obs, reward, terminated, truncated, info = env.step(action)

            # Rendering
            env.render()

            # Info alle 20 Steps
            if (step + 1) % 20 == 0:
                print(f"Step {step+1:3d}: "
                      f"Pos=[{info['position'][0]:6.2f}, {info['position'][1]:6.2f}, {info['position'][2]:6.2f}], "
                      f"Dist={info['distance_to_target']:6.2f}m, "
                      f"Reward={reward:.4f}")

            if terminated or truncated:
                print("Episode beendet!")
                break

            # Die Pause ist jetzt in env.render() integriert (plt.pause)

    except KeyboardInterrupt:
        print("\nTest abgebrochen.")

    finally:
        env.close()
        print("\nRendering-Test abgeschlossen!")


if __name__ == "__main__":
    test_rendering()

