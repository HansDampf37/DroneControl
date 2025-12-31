"""
Test script for the new pygame renderer.

Demonstrates the performance improvement of the pygame-based 3D renderer
over the matplotlib-based renderer.
"""
import time
import numpy as np
from src.drone_env import DroneEnv


def test_renderer(renderer_type="pygame", num_steps=200):
    """
    Test a renderer by running the environment for a certain number of steps.

    Args:
        renderer_type: "pygame" or "matplotlib"
        num_steps: Number of steps to run

    Returns:
        Average FPS
    """
    print(f"\nTesting {renderer_type} renderer...")

    # Create environment with the specified renderer
    env = DroneEnv(
        render_mode="human",
        renderer_type=renderer_type,
        max_steps=num_steps,
        use_wind=True
    )

    # Reset environment
    obs, info = env.reset()

    # Track timing
    start_time = time.time()
    frame_times = []

    # Run simulation
    for step in range(num_steps):
        frame_start = time.time()

        # Random action (for testing purposes)
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render
        env.render()

        frame_end = time.time()
        frame_times.append(frame_end - frame_start)

        if terminated or truncated:
            obs, info = env.reset()

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    avg_fps = num_steps / total_time
    avg_frame_time = np.mean(frame_times) * 1000  # in ms
    min_frame_time = np.min(frame_times) * 1000
    max_frame_time = np.max(frame_times) * 1000

    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average frame time: {avg_frame_time:.2f}ms")
    print(f"  Min frame time: {min_frame_time:.2f}ms")
    print(f"  Max frame time: {max_frame_time:.2f}ms")

    env.close()

    return avg_fps


def compare_renderers():
    """Compare the performance of both renderers."""
    print("=" * 60)
    print("Renderer Performance Comparison")
    print("=" * 60)

    num_steps = 200

    # Test pygame renderer
    pygame_fps = test_renderer("pygame", num_steps)

    # Wait a bit between tests
    time.sleep(2)

    # Test matplotlib renderer
    matplotlib_fps = test_renderer("matplotlib", num_steps)

    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"Pygame FPS: {pygame_fps:.2f}")
    print(f"Matplotlib FPS: {matplotlib_fps:.2f}")
    print(f"Speedup: {pygame_fps / matplotlib_fps:.2f}x faster")
    print("=" * 60)


if __name__ == "__main__":
    # You can run individual tests or the comparison

    # Option 1: Just test pygame renderer
    test_renderer("pygame", num_steps=300)

    # Option 2: Compare both renderers
    # compare_renderers()

