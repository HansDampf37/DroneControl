"""Beispiel: Random Agent im Drohnen-Environment."""
import sys
from pathlib import Path

# Füge src/ zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv


def run_random_agent(episodes=5, max_steps=500, render=False):
    """
    Führt einen Random Agent im Environment aus.

    Args:
        episodes: Anzahl der Episoden
        max_steps: Maximale Steps pro Episode
        render: Ob das Environment visualisiert werden soll
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
        print(f"Start-Distanz zum Ziel: {info['distance_to_target']:.2f}m")
        print(f"Ziel-Position: {info['target_position']}")

        for step in range(max_steps):
            # Random Action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            distances.append(info['distance_to_target'])

            if render:
                env.render()

            # Zeige Fortschritt
            if (step + 1) % 100 == 0:
                avg_distance = np.mean(distances[-100:])
                print(f"  Step {step + 1:3d}: Avg-Distanz (letzte 100)={avg_distance:.2f}m, "
                      f"Reward={reward:.4f}")

            if terminated or truncated:
                break

        # Episode-Statistiken
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)
        final_distance = distances[-1]

        print(f"\nEpisode {episode + 1} beendet nach {len(distances)} Steps:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Durchschnittliche Distanz: {avg_distance:.2f}m")
        print(f"  Minimale Distanz: {min_distance:.2f}m")
        print(f"  Finale Distanz: {final_distance:.2f}m")

        episode_rewards.append(total_reward)
        episode_distances.append(avg_distance)

    env.close()

    # Zusammenfassung über alle Episoden
    print(f"\n{'='*60}")
    print(f"Zusammenfassung über {episodes} Episoden:")
    print(f"{'='*60}")
    print(f"Durchschnittlicher Total Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Durchschnittliche Distanz: {np.mean(episode_distances):.2f} ± {np.std(episode_distances):.2f}m")
    print(f"Bester Total Reward: {np.max(episode_rewards):.2f}")
    print(f"Beste durchschnittliche Distanz: {np.min(episode_distances):.2f}m")


def run_hover_agent(episodes=3, max_steps=500, render=False):
    """
    Führt einen simplen Hover-Agent aus (alle Motoren auf ~25%).
    Dieser Agent versucht nur zu schweben, ohne zum Ziel zu fliegen.
    """
    render_mode = "human" if render else None
    env = DroneEnv(max_steps=max_steps, render_mode=render_mode)

    print(f"\n{'='*60}")
    print(f"Hover-Agent (Baseline)")
    print(f"{'='*60}")

    episode_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0

        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"Start-Distanz: {info['distance_to_target']:.2f}m")

        for step in range(max_steps):
            # Hover: Alle Motoren auf ~25% (kompensiert Gravitation)
            action = np.array([0.25, 0.25, 0.25, 0.25])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

            if terminated or truncated:
                break

        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Finale Distanz: {info['distance_to_target']:.2f}m")
        print(f"  Finale Höhe: {info['position'][2]:.2f}m")

        episode_rewards.append(total_reward)

    env.close()

    print(f"\nHover-Agent Durchschnitt: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Drohnen-Environment mit Random/Hover Agent')
    parser.add_argument('--episodes', type=int, default=5, help='Anzahl der Episoden')
    parser.add_argument('--steps', type=int, default=500, help='Max Steps pro Episode')
    parser.add_argument('--render', action='store_true', help='Aktiviere Visualisierung')
    parser.add_argument('--agent', type=str, default='random', choices=['random', 'hover'],
                        help='Agent-Typ: random oder hover')

    args = parser.parse_args()

    print("Drohnen-RL Environment - Agent Demo")
    print(f"Agent: {args.agent.upper()}")
    print(f"Episoden: {args.episodes}")
    print(f"Max Steps: {args.steps}")
    print(f"Rendering: {'ON' if args.render else 'OFF'}")

    if args.agent == 'random':
        run_random_agent(episodes=args.episodes, max_steps=args.steps, render=args.render)
    else:
        run_hover_agent(episodes=args.episodes, max_steps=args.steps, render=args.render)

