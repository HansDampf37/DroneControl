"""Quick test for rendering issues."""
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv


def test_rendering():
    """Tests rendering with different actions."""
    print("Testing rendering...")
    print("Close the window or press Ctrl+C to exit\n")

    env = DroneEnv(max_steps=200, render_mode="human")

    obs, info = env.reset(seed=42)
    print(f"Start position: {info['position']}")
    print(f"Target position: {info['target_position']}")
    print(f"Initial distance: {info['distance_to_target']:.2f}m\n")

    try:
        for step in range(200):
            # Switch between different actions for visible movement
            if step < 50:
                # Hover
                action = np.array([0.25, 0.25, 0.25, 0.25])
            elif step < 100:
                # Slight rotation (more yaw)
                action = np.array([0.3, 0.2, 0.2, 0.3])
            elif step < 150:
                # Tilt forward
                action = np.array([0.3, 0.3, 0.2, 0.2])
            else:
                # Back to hover
                action = np.array([0.25, 0.25, 0.25, 0.25])

            obs, reward, terminated, truncated, info = env.step(action)

            # Rendering
            env.render()

            # Info every 20 steps
            if (step + 1) % 20 == 0:
                print(f"Step {step+1:3d}: "
                      f"Pos=[{info['position'][0]:6.2f}, {info['position'][1]:6.2f}, {info['position'][2]:6.2f}], "
                      f"Dist={info['distance_to_target']:6.2f}m, "
                      f"Reward={reward:.4f}")

            if terminated or truncated:
                print("Episode ended!")
                break

            # Pause is now integrated in env.render() (plt.pause)

    except KeyboardInterrupt:
        print("\nTest aborted.")

    finally:
        env.close()
        print("\nRendering test completed!")


if __name__ == "__main__":
    test_rendering()

