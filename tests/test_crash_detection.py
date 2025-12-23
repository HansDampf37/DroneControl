"""Test for crash detection system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv


def test_crash_z_threshold():
    """Test: Crash detection by low z-coordinate threshold."""
    print("=" * 60)
    print("Test 1: Crash at low Z-coordinate")
    print("=" * 60)

    env = DroneEnv(
        max_steps=1000,
        enable_crash_detection=True,
        crash_z_vel_threshold=-5.0,
        render_mode=None
    )

    obs, info = env.reset(seed=42)
    print(f"Start position: {info['position']}")

    # Let the drone fall (no thrust)
    crashed = False
    for step in range(200):
        action = np.array([0.0, 0.0, 0.0, 0.0])  # No thrust
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            print(f"Step {step:3d}: Z={info['position'][2]:6.2f}m, Crashed={info['crashed']}")

        if terminated:
            crashed = True
            print(f"\n✓ Crash detected at step {step}, Z={info['position'][2]:.2f}m")
            break

    if not crashed:
        print(f"\n✗ No crash detected! Final Z-position: {info['position'][2]:.2f}m")

    env.close()
    print()
    return crashed


def test_crash_tilt():
    """Test: Crash detection by extreme tilt angle."""
    print("=" * 60)
    print("Test 2: Crash at extreme tilt")
    print("=" * 60)

    env = DroneEnv(
        max_steps=1000,
        enable_crash_detection=True,
        crash_z_vel_threshold=-50.0,  # Very low, so only tilt triggers
        crash_tilt_threshold=80.0,
        render_mode=None
    )

    obs, info = env.reset(seed=123)
    print(f"Start position: {info['position']}")

    # Force extreme tilt (very asymmetric thrust)
    crashed = False
    for step in range(200):
        action = np.array([0.0, 0.5, 0.5, 0.0])  # Extremely asymmetric
        obs, reward, terminated, truncated, info = env.step(action)

        roll_deg = np.rad2deg(obs[6])
        pitch_deg = np.rad2deg(obs[7])

        if step % 20 == 0:
            print(f"Step {step:3d}: Roll={roll_deg:6.1f}°, Pitch={pitch_deg:6.1f}°, Crashed={info['crashed']}")

        if terminated:
            crashed = True
            print(f"\n✓ Crash detected at step {step}")
            print(f"  Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°")
            break

    if not crashed:
        print(f"\n✗ No crash detected! Final tilt: Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°")

    env.close()
    print()
    return crashed


def test_no_crash_disabled():
    """Test: No crash detection when disabled."""
    print("=" * 60)
    print("Test 3: Crash detection disabled")
    print("=" * 60)

    env = DroneEnv(
        max_steps=200,
        enable_crash_detection=False,  # Disabled
        render_mode=None
    )

    obs, info = env.reset(seed=42)

    # Let the drone fall
    crashed = False
    for step in range(200):
        action = np.array([0.0, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            crashed = True
            break

    if not crashed:
        print(f"✓ Correct: No crash detected (crash detection disabled)")
        print(f"  Final Z-position: {info['position'][2]:.2f}m")
    else:
        print(f"✗ Error: Crash detected despite being disabled!")

    env.close()
    print()
    return not crashed


def test_normal_flight():
    """Test: No crash during normal hovering."""
    print("=" * 60)
    print("Test 4: Normal hover (no crash)")
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
            print(f"✗ Unexpected crash at step {step}")
            print(f"  Z={info['position'][2]:.2f}m")
            break

    if not crashed:
        print(f"✓ Correct: No crash during normal hover")
        print(f"  Final Z-position: {info['position'][2]:.2f}m")
        print(f"  All {step+1} steps successful")

    env.close()
    print()
    return not crashed


def test_crash_info():
    """Test: 'crashed' flag in info dictionary."""
    print("=" * 60)
    print("Test 5: 'crashed' Info Flag")
    print("=" * 60)

    env = DroneEnv(
        max_steps=1000,
        enable_crash_detection=True,
        crash_z_vel_threshold=-5.0,
        render_mode=None
    )

    obs, info = env.reset(seed=42)

    # Check that 'crashed' is present in info
    action = np.array([0.0, 0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)

    if 'crashed' in info:
        print(f"✓ 'crashed' flag present in info")
        print(f"  Value: {info['crashed']}")
    else:
        print(f"✗ 'crashed' flag missing in info!")

    env.close()
    print()
    return 'crashed' in info


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CRASH DETECTION TESTS")
    print("=" * 60 + "\n")

    results = []

    # Run all tests
    results.append(("Z-Threshold Crash", test_crash_z_threshold()))
    results.append(("Tilt Crash", test_crash_tilt()))
    results.append(("Disabled", test_no_crash_disabled()))
    results.append(("Normal Hover", test_normal_flight()))
    results.append(("Info-Flag", test_crash_info()))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests successful!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

