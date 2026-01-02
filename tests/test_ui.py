#!/usr/bin/env python3
"""
Test the new layout with info box, motor bars, and wind arrow outside the plot.
"""
import numpy as np

from src.drone_env import ThrustChangeController
from src.drone_env.env import DroneEnv


def test_new_layout():
    # Create environment with rendering and wind enabled
    env = ThrustChangeController(
            DroneEnv(
            max_steps=300,
            dt=0.02,
            render_mode="human",
            use_wind=True,
            wind_strength_range=(2.0, 5.0), # Moderate wind
            enable_crash_detection=False,
            enable_out_of_bounds_detection=True,
        )
    )

    try:
        while True:
            # Reset environment
            env.reset()

            # Control pattern: vary motors to show different patterns
            for step in range(3000):
                time = step * env.dt

                # Create a rotating pattern
                phase = time * 40
                motor_1 = 0.01 * np.sin(phase)
                motor_2 = 0.01 * np.sin(phase + np.pi/2)
                motor_3 = 0.01 * np.sin(phase + np.pi)
                motor_4 = 0.01 * np.sin(phase + 3*np.pi/2)

                action = np.array([motor_1, motor_2, motor_3, motor_4], dtype=np.float32)

                # Step and render
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()

                if step % 50 == 0:
                    wind = env.wind.get_vector()
                    print(f"Step {step}: Time={time:.2f}s, Wind=[{wind[0]:.2f}, {wind[1]:.2f}, {wind[2]:.2f}] m/s")

                if terminated or truncated:
                    print(f"Episode ended at step {step}")
                    break

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        env.close()
        print("\nTest completed!")
        print("All UI elements should now be positioned outside the plot area on the right.")


if __name__ == "__main__":
    test_new_layout()

