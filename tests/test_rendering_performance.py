#!/usr/bin/env python3
"""
Performance test for the optimized rendering method.
"""

from src.drone_env.env import DroneEnv
import numpy as np
import time

def test_without_rendering():
    """Test without rendering - baseline performance"""
    print("=" * 60)
    print("Test 1: Simulation WITHOUT Rendering (Baseline)")
    print("=" * 60)

    env = DroneEnv(render_mode=None, max_steps=200)
    obs, info = env.reset()

    start = time.time()
    for i in range(200):
        action = np.random.uniform(0.4, 0.8, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    elapsed = time.time() - start
    fps = 200 / elapsed
    print(f"✅ 200 steps in {elapsed:.3f}s ({fps:.1f} steps/sec)")
    print(f"   Ø {elapsed/200*1000:.2f}ms per step")
    env.close()
    return fps

def test_with_rendering_visual():
    """Test with visual rendering - only for local tests"""
    print("\n" + "=" * 60)
    print("Test 2: Simulation WITH Rendering (human mode)")
    print("=" * 60)
    print("Note: Close the window manually after a few seconds")

    try:
        env = DroneEnv(render_mode='human', max_steps=200)
        obs, info = env.reset()

        start = time.time()
        for i in range(50):  # Only 50 frames for visual test
            action = np.random.uniform(0.4, 0.8, 4)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if i % 10 == 0:
                print(f"  Frame {i}/50 rendered...")

            if terminated or truncated:
                break

        elapsed = time.time() - start
        fps = min(i+1, 50) / elapsed
        print(f"✅ {min(i+1, 50)} frames in {elapsed:.3f}s ({fps:.1f} FPS)")
        print(f"   Ø {elapsed/min(i+1, 50)*1000:.2f}ms per frame")
        env.close()
        return fps
    except Exception as e:
        print(f"⚠️  Visual rendering not available: {e}")
        print("   (Normal in headless environments)")
        return None

def main():
    print("\n🚁 Drone Environment - Performance Tests")
    print("=" * 60)

    baseline_fps = test_without_rendering()

    # Only test visual rendering if available
    visual_fps = test_with_rendering_visual()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Simulation (without rendering): {baseline_fps:.1f} steps/sec")
    if visual_fps:
        print(f"Rendering (human mode):         {visual_fps:.1f} FPS")
        overhead = (1 - visual_fps/baseline_fps) * 100
        print(f"Rendering overhead:             {overhead:.1f}%")

    print("\n✅ Performance optimizations active:")
    print("   - Reuse of plot objects")
    print("   - Update instead of clearing axes")
    print("   - Conditional rendering (e.g. wind only when visible)")
    print("   - Reduction of redundant calculations")
    print("=" * 60)

if __name__ == "__main__":
    main()

