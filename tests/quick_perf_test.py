#!/usr/bin/env python3
"""
Quick test for optimized rendering performance.
"""

from src.drone_env.env import DroneEnv
import numpy as np
import time

def quick_test():
    print("\n" + "=" * 60)
    print("🚁 QUICK RENDERING PERFORMANCE TEST")
    print("=" * 60)

    # Test 1: Baseline without rendering
    print("\n1. Baseline (without rendering)...")
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

    # Test 2: With rendering
    print("\n2. With rendering (human mode)...")
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

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Simulation:  {baseline_fps:>6.0f} steps/sec")
    print(f"Rendering:   {render_fps:>6.1f} FPS")
    overhead = (1 - render_fps/baseline_fps) * 100
    print(f"Overhead:    {overhead:>6.1f}%")
    print("=" * 60)

    # Evaluation
    if render_fps >= 10:
        print("✅ EXCELLENT - Rendering is well optimized!")
    elif render_fps >= 7:
        print("✓  GOOD - Rendering is acceptable")
    else:
        print("⚠  SLOW - Rendering could be optimized")

    print("\nActive optimizations:")
    print("  ✓ Object reuse")
    print("  ✓ Update instead of clear")
    print("  ✓ Conditional rendering")
    print("  ✓ Two-View layout (Top + Front)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    quick_test()

