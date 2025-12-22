"""Test der Crash-Detektion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv


def test_crash_z_threshold():
    """Test: Crash durch zu niedrige Z-Koordinate."""
    print("=" * 60)
    print("Test 1: Crash bei niedriger Z-Koordinate")
    print("=" * 60)

    env = DroneEnv(
        max_steps=1000,
        enable_crash_detection=True,
        crash_z_threshold=-5.0,
        render_mode=None
    )

    obs, info = env.reset(seed=42)
    print(f"Start-Position: {info['position']}")

    # Lasse die Drohne fallen (kein Thrust)
    crashed = False
    for step in range(200):
        action = np.array([0.0, 0.0, 0.0, 0.0])  # Kein Thrust
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            print(f"Step {step:3d}: Z={info['position'][2]:6.2f}m, Crashed={info['crashed']}")

        if terminated:
            crashed = True
            print(f"\n✓ Crash detektiert bei Step {step}, Z={info['position'][2]:.2f}m")
            break

    if not crashed:
        print(f"\n✗ Kein Crash detektiert! Finale Z-Position: {info['position'][2]:.2f}m")

    env.close()
    print()
    return crashed


def test_crash_tilt():
    """Test: Crash durch extreme Neigung."""
    print("=" * 60)
    print("Test 2: Crash bei extremer Neigung")
    print("=" * 60)

    env = DroneEnv(
        max_steps=1000,
        enable_crash_detection=True,
        crash_z_threshold=-50.0,  # Sehr niedrig, damit nur Tilt triggert
        crash_tilt_threshold=80.0,
        render_mode=None
    )

    obs, info = env.reset(seed=123)
    print(f"Start-Position: {info['position']}")

    # Erzwinge extreme Neigung (sehr asymmetrischer Thrust)
    crashed = False
    for step in range(200):
        action = np.array([0.0, 0.5, 0.5, 0.0])  # Extrem asymmetrisch
        obs, reward, terminated, truncated, info = env.step(action)

        roll_deg = np.rad2deg(obs[6])
        pitch_deg = np.rad2deg(obs[7])

        if step % 20 == 0:
            print(f"Step {step:3d}: Roll={roll_deg:6.1f}°, Pitch={pitch_deg:6.1f}°, Crashed={info['crashed']}")

        if terminated:
            crashed = True
            print(f"\n✓ Crash detektiert bei Step {step}")
            print(f"  Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°")
            break

    if not crashed:
        print(f"\n✗ Kein Crash detektiert! Finale Neigung: Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°")

    env.close()
    print()
    return crashed


def test_no_crash_disabled():
    """Test: Keine Crash-Detektion wenn deaktiviert."""
    print("=" * 60)
    print("Test 3: Crash-Detektion deaktiviert")
    print("=" * 60)

    env = DroneEnv(
        max_steps=200,
        enable_crash_detection=False,  # Deaktiviert
        render_mode=None
    )

    obs, info = env.reset(seed=42)

    # Lasse die Drohne fallen
    crashed = False
    for step in range(200):
        action = np.array([0.0, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            crashed = True
            break

    if not crashed:
        print(f"✓ Korrekt: Kein Crash detektiert (Crash-Detektion deaktiviert)")
        print(f"  Finale Z-Position: {info['position'][2]:.2f}m")
    else:
        print(f"✗ Fehler: Crash detektiert obwohl deaktiviert!")

    env.close()
    print()
    return not crashed


def test_normal_flight():
    """Test: Kein Crash bei normalem Hover."""
    print("=" * 60)
    print("Test 4: Normaler Hover (kein Crash)")
    print("=" * 60)

    env = DroneEnv(
        max_steps=100,
        enable_crash_detection=True,
        render_mode=None
    )

    obs, info = env.reset(seed=789)

    # Hover
    crashed = False
    for step in range(100):
        action = np.array([0.25, 0.25, 0.25, 0.25])
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            crashed = True
            print(f"✗ Unerwarteter Crash bei Step {step}")
            print(f"  Z={info['position'][2]:.2f}m")
            break

    if not crashed:
        print(f"✓ Korrekt: Kein Crash bei normalem Hover")
        print(f"  Finale Z-Position: {info['position'][2]:.2f}m")
        print(f"  Alle {step+1} Steps erfolgreich")

    env.close()
    print()
    return not crashed


def test_crash_info():
    """Test: 'crashed' Flag in info."""
    print("=" * 60)
    print("Test 5: 'crashed' Info-Flag")
    print("=" * 60)

    env = DroneEnv(
        max_steps=1000,
        enable_crash_detection=True,
        crash_z_threshold=-5.0,
        render_mode=None
    )

    obs, info = env.reset(seed=42)

    # Prüfe dass 'crashed' in info vorhanden ist
    action = np.array([0.0, 0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)

    if 'crashed' in info:
        print(f"✓ 'crashed' Flag vorhanden in info")
        print(f"  Wert: {info['crashed']}")
    else:
        print(f"✗ 'crashed' Flag fehlt in info!")

    env.close()
    print()
    return 'crashed' in info


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CRASH-DETEKTION TESTS")
    print("=" * 60 + "\n")

    results = []

    # Führe alle Tests aus
    results.append(("Z-Threshold Crash", test_crash_z_threshold()))
    results.append(("Tilt Crash", test_crash_tilt()))
    results.append(("Deaktiviert", test_no_crash_disabled()))
    results.append(("Normaler Hover", test_normal_flight()))
    results.append(("Info-Flag", test_crash_info()))

    # Zusammenfassung
    print("=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nErgebnis: {passed}/{total} Tests bestanden")

    if passed == total:
        print("\n🎉 Alle Tests erfolgreich!")
    else:
        print(f"\n⚠️  {total - passed} Test(s) fehlgeschlagen")

