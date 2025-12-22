"""Test-Skript für das Drohnen-Environment."""
import sys
from pathlib import Path

# Füge src/ zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv


def test_basic_functionality():
    """Testet grundlegende Environment-Funktionalität."""
    print("=" * 60)
    print("Test 1: Grundlegende Funktionalität")
    print("=" * 60)

    env = DroneEnv(max_steps=100, render_mode=None)

    # Reset
    obs, info = env.reset(seed=42)
    print(f"✓ Reset erfolgreich")
    print(f"  Observation Shape: {obs.shape}")
    print(f"  Action Space: {env.action_space}")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Initiale Distanz zum Ziel: {info['distance_to_target']:.2f}m")

    # Ein paar Steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if i == 0:
            print(f"\n✓ Step erfolgreich")
            print(f"  Observation: {obs[:6]}")  # Erste 6 Werte
            print(f"  Reward: {reward:.4f}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")

    env.close()
    print(f"\n✓ Test 1 bestanden!\n")


def test_physics():
    """Testet die Physik-Simulation."""
    print("=" * 60)
    print("Test 2: Physik-Simulation")
    print("=" * 60)

    env = DroneEnv(max_steps=500, dt=0.01, render_mode=None)
    obs, info = env.reset(seed=123)

    print(f"Start-Position: {info['position']}")
    print(f"Ziel-Position: {info['target_position']}")

    # Hover-Test: Alle Motoren auf ~25% (sollte Gravitation kompensieren)
    hover_thrust = 0.25  # Approximation für Hover
    print(f"\n→ Teste Hover mit {hover_thrust*100:.0f}% Thrust...")

    for _ in range(100):
        action = np.array([hover_thrust] * 4)
        obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Position nach 1s: {info['position']}")
    print(f"  Geschwindigkeit: {obs[3:6]}")

    # Full-Thrust-Test
    print(f"\n→ Teste Full-Thrust...")
    env.reset(seed=123)

    for _ in range(100):
        action = np.array([1.0, 1.0, 1.0, 1.0])
        obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Position nach 1s: {info['position']}")
    print(f"  Höhe (Z): {info['position'][2]:.2f}m")

    # Roll-Test (linke Motoren stärker)
    print(f"\n→ Teste Roll (asymmetrische Motoren)...")
    env.reset(seed=123)

    for _ in range(50):
        action = np.array([0.3, 0.2, 0.3, 0.2])  # Rechts mehr Thrust
        obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Roll-Winkel: {np.rad2deg(obs[6]):.2f}°")
    print(f"  Pitch-Winkel: {np.rad2deg(obs[7]):.2f}°")
    print(f"  Yaw-Winkel: {np.rad2deg(obs[8]):.2f}°")

    env.close()
    print(f"\n✓ Test 2 bestanden!\n")


def test_wind():
    """Testet die Wind-Dynamik."""
    print("=" * 60)
    print("Test 3: Wind-Dynamik")
    print("=" * 60)

    env = DroneEnv(
        max_steps=200,
        wind_strength_range=(2.0, 5.0),
        render_mode=None
    )
    obs, info = env.reset(seed=456)

    wind_vectors = []

    for i in range(200):
        action = np.array([0.25] * 4)  # Hover
        obs, reward, terminated, truncated, info = env.step(action)
        wind_vectors.append(obs[12:15].copy())

    wind_vectors = np.array(wind_vectors)

    print(f"Initialer Wind: {wind_vectors[0]}")
    print(f"Finaler Wind: {wind_vectors[-1]}")
    print(f"Durchschnittliche Windstärke: {np.mean(np.linalg.norm(wind_vectors, axis=1)):.2f} m/s")
    print(f"Max Windstärke: {np.max(np.linalg.norm(wind_vectors, axis=1)):.2f} m/s")
    print(f"Min Windstärke: {np.min(np.linalg.norm(wind_vectors, axis=1)):.2f} m/s")

    env.close()
    print(f"\n✓ Test 3 bestanden!\n")


def test_reward():
    """Testet die Reward-Funktion."""
    print("=" * 60)
    print("Test 4: Reward-Funktion")
    print("=" * 60)

    env = DroneEnv(max_steps=100, render_mode=None)
    obs, info = env.reset(seed=789)

    initial_distance = info['distance_to_target']
    print(f"Initiale Distanz: {initial_distance:.2f}m")

    rewards = []
    distances = []

    for i in range(100):
        # Zufällige Aktionen
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        distances.append(info['distance_to_target'])

    print(f"\nReward-Statistiken:")
    print(f"  Min: {np.min(rewards):.4f}")
    print(f"  Max: {np.max(rewards):.4f}")
    print(f"  Mean: {np.mean(rewards):.4f}")

    print(f"\nDistanz-Statistiken:")
    print(f"  Min: {np.min(distances):.2f}m")
    print(f"  Max: {np.max(distances):.2f}m")
    print(f"  Final: {distances[-1]:.2f}m")

    # Verifiziere Reward-Formel
    test_distance = 10.0
    expected_reward = 1.0 / (1.0 + test_distance)
    print(f"\nVerifiziere Formel (Distanz=10m):")
    print(f"  Erwarteter Reward: {expected_reward:.4f}")

    env.close()
    print(f"\n✓ Test 4 bestanden!\n")


def demo_with_visualization():
    """Demo mit Visualisierung."""
    print("=" * 60)
    print("Demo: Visualisierung")
    print("=" * 60)
    print("Starte Environment mit Rendering...")
    print("(Fenster schließen zum Beenden)")

    env = DroneEnv(
        max_steps=500,
        render_mode="human"
    )

    obs, info = env.reset(seed=42)

    done = False
    step = 0

    try:
        while not done and step < 500:
            # Einfache Policy: Hover + kleine zufällige Variation
            action = np.array([0.25, 0.25, 0.25, 0.25]) + np.random.uniform(-0.05, 0.05, 4)
            action = np.clip(action, 0, 1)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            env.render()
            step += 1

            if step % 100 == 0:
                print(f"Step {step}: Distanz={info['distance_to_target']:.2f}m, Reward={reward:.4f}")

    except KeyboardInterrupt:
        print("\nDemo unterbrochen.")

    finally:
        env.close()

    print(f"\n✓ Demo beendet!\n")


if __name__ == "__main__":
    # Führe alle Tests aus
    test_basic_functionality()
    test_physics()
    test_wind()
    test_reward()

    # Optional: Demo mit Visualisierung
    print("\nMöchtest du die Demo mit Visualisierung starten?")
    print("(Drücke Enter zum Fortfahren oder Ctrl+C zum Überspringen)")
    try:
        input()
        demo_with_visualization()
    except KeyboardInterrupt:
        print("\nDemo übersprungen.")

    print("\n" + "=" * 60)
    print("Alle Tests erfolgreich abgeschlossen! 🚁")
    print("=" * 60)

