#!/usr/bin/env python3
"""
Schneller Test für die optimierte Rendering-Performance.
"""

from src.drone_env.env import DroneEnv
import numpy as np
import time

def quick_test():
    print("\n" + "=" * 60)
    print("🚁 QUICK RENDERING PERFORMANCE TEST")
    print("=" * 60)

    # Test 1: Baseline ohne Rendering
    print("\n1. Baseline (ohne Rendering)...")
    env = DroneEnv(render_mode=None, max_steps=100)
    obs, info = env.reset()

    start = time.time()
    for _ in range(100):
        action = np.random.uniform(0.4, 0.8, 4)
        obs, reward, terminated, truncated, info = env.step(action)
    elapsed = time.time() - start
    baseline_fps = 100 / elapsed
    print(f"   ✅ {baseline_fps:.0f} steps/sec ({elapsed*10:.1f}ms per step)")
    env.close()

    # Test 2: Mit Rendering
    print("\n2. Mit Rendering (human mode)...")
    env = DroneEnv(render_mode='human', max_steps=100)
    obs, info = env.reset()

    start = time.time()
    for i in range(30):
        action = np.random.uniform(0.4, 0.8, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    elapsed = time.time() - start
    render_fps = 30 / elapsed
    print(f"   ✅ {render_fps:.1f} FPS ({elapsed/30*1000:.0f}ms per frame)")
    env.close()

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("ERGEBNIS")
    print("=" * 60)
    print(f"Simulation:  {baseline_fps:>6.0f} steps/sec")
    print(f"Rendering:   {render_fps:>6.1f} FPS")
    overhead = (1 - render_fps/baseline_fps) * 100
    print(f"Overhead:    {overhead:>6.1f}%")
    print("=" * 60)

    # Bewertung
    if render_fps >= 10:
        print("✅ EXCELLENT - Rendering ist gut optimiert!")
    elif render_fps >= 7:
        print("✓  GOOD - Rendering ist akzeptabel")
    else:
        print("⚠  SLOW - Rendering könnte optimiert werden")

    print("\nOptimierungen aktiv:")
    print("  ✓ Objekt-Wiederverwendung")
    print("  ✓ Update statt Clear")
    print("  ✓ Bedingte Darstellung")
    print("  ✓ Two-View Layout (Top + Front)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    quick_test()

