"""Example: Random Agent in Drone Environment."""
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv


def run_random_agent(episodes=5, max_steps=500, render=False):
    """
    Runs a random agent in the environment.

    Args:
        episodes: Number of episodes to run. Default is 5.
        max_steps: Maximum steps per episode. Default is 500.
        render: Whether to visualize the environment. Default is False.
    """
    render_mode = "human" if render else None
    env = DroneEnv(max_steps=max_steps, render_mode=render_mode)

    episode_rewards = []
    episode_distances = []

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        distances = []

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{episodes}")
        print(f"{'='*60}")
        print(f"Initial distance to target: {info['distance_to_target']:.2f}m")
        print(f"Target position: {info['target_position']}")

        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            distances.append(info['distance_to_target'])

            if render:
                env.render()

            # Show progress
            if (step + 1) % 100 == 0:
                avg_distance = np.mean(distances[-100:])
                print(f"  Step {step + 1:3d}: Avg distance (last 100)={avg_distance:.2f}m, "
                      f"Reward={reward:.4f}")

            if terminated or truncated:
                break

        # Episode statistics
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)
        final_distance = distances[-1]

        print(f"\nEpisode {episode + 1} ended after {len(distances)} steps:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Average Distance: {avg_distance:.2f}m")
        print(f"  Minimum Distance: {min_distance:.2f}m")
        print(f"  Final Distance: {final_distance:.2f}m")

        episode_rewards.append(total_reward)
        episode_distances.append(avg_distance)

    env.close()

    # Summary across all episodes
    print(f"\n{'='*60}")
    print(f"Summary over {episodes} episodes:")
    print(f"{'='*60}")
    print(f"Average Total Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Distance: {np.mean(episode_distances):.2f} ± {np.std(episode_distances):.2f}m")
    print(f"Best Total Reward: {np.max(episode_rewards):.2f}")
    print(f"Best Average Distance: {np.min(episode_distances):.2f}m")


def run_hover_agent(episodes=3, max_steps=500, render=False):
    """
    Runs a simple hover agent (all motors at ~25%).

    This agent only tries to hover without flying to the target.

    Args:
        episodes: Number of episodes to run. Default is 3.
        max_steps: Maximum steps per episode. Default is 500.
        render: Whether to visualize the environment. Default is False.
    """
    render_mode = "human" if render else None
    env = DroneEnv(max_steps=max_steps, render_mode=render_mode)

    print(f"\n{'='*60}")
    print(f"Hover Agent (Baseline)")
    print(f"{'='*60}")

    episode_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0

        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"Initial distance: {info['distance_to_target']:.2f}m")

        for step in range(max_steps):
            # Hover: All motors at ~25% (compensates for gravity)
            action = np.array([0.25, 0.25, 0.25, 0.25])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

            if terminated or truncated:
                break

        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final Distance: {info['distance_to_target']:.2f}m")
        print(f"  Final Height: {info['position'][2]:.2f}m")

        episode_rewards.append(total_reward)

    env.close()

    print(f"\nHover Agent Average: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Drone Environment with Random/Hover Agent')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=5000, help='Max steps per episode')
    parser.add_argument('--render', action='store_true', help='Enable visualization')
    parser.add_argument('--agent', type=str, default='random', choices=['random', 'hover'],
                        help='Agent type: random or hover')

    args = parser.parse_args()

    print("Drone RL Environment - Agent Demo")
    print(f"Agent: {args.agent.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.steps}")
    print(f"Rendering: {'ON' if args.render else 'OFF'}")

    if args.agent == 'random':
        run_random_agent(episodes=args.episodes, max_steps=args.steps, render=args.render)
    else:
        run_hover_agent(episodes=args.episodes, max_steps=args.steps, render=args.render)

