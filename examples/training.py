"""
Beispiel-Training mit Ray RLlib.

Installation:
    pip install ray[rllib] gymnasium torch

Ausführung:
    python training.py --algorithm PPO --timesteps 100000
"""
import sys
from pathlib import Path

# Füge src/ zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import os
from src.drone_env import RLlibDroneEnv


def train_with_rllib(algorithm='PPO', total_timesteps=100000, save_path='../models/drone_model'):
    """
    Trainiert einen RL-Agenten mit Ray RLlib.

    Args:
        algorithm: RL-Algorithmus ('PPO', 'SAC', 'APPO')
        total_timesteps: Anzahl der Trainings-Steps
        save_path: Pfad zum Speichern des Modells
    """
    try:
        from ray import tune, air
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.algorithms.sac import SACConfig
        from ray.rllib.algorithms.appo import APPOConfig
        import ray
    except ImportError:
        print("ERROR: Ray RLlib nicht installiert!")
        print("Installation mit: pip install ray[rllib] torch")
        return

    # Ray initialisieren
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_gpus=0)

    # Konvertiere zu absolutem Pfad (RLlib 2.5+ benötigt absoluten Pfad)
    save_path = os.path.abspath(save_path)

    # Erstelle Verzeichnisse
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('../logs', exist_ok=True)

    print(f"Erstelle {algorithm}-Konfiguration...")

    # Environment-Konfiguration
    env_config = {
        "max_steps": 500,
        "render_mode": None,
        "enable_crash_detection": True,
    }

    # Algorithmus-Konfiguration
    if algorithm == 'PPO':
        config = (
            PPOConfig()
            .environment(RLlibDroneEnv, env_config=env_config)
            .framework("torch")
            .training(
                lr=3e-4,
                train_batch_size=2048,
                minibatch_size=64,
                num_epochs=10,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
            )
            .env_runners(num_env_runners=2, num_envs_per_env_runner=1)
            .resources(num_gpus=0)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=5,
                evaluation_config={"explore": False},
            )
        )
    elif algorithm == 'SAC':
        config = (
            SACConfig()
            .environment(RLlibDroneEnv, env_config=env_config)
            .framework("torch")
            .training(
                lr=3e-4,
                train_batch_size=256,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": 100000,
                },
                tau=0.005,
                gamma=0.99,
                target_entropy="auto",
            )
            .env_runners(num_env_runners=2)  # Geändert von rollouts
            .resources(num_gpus=0)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=5,
            )
        )
    elif algorithm == 'APPO':
        config = (
            APPOConfig()
            .environment(RLlibDroneEnv, env_config=env_config)
            .framework("torch")
            .training(
                lr=3e-4,
                train_batch_size=2048,
                gamma=0.99,
            )
            .env_runners(num_env_runners=4)  # Geändert von rollouts
            .resources(num_gpus=0)
        )
    else:
        print(f"ERROR: Unbekannter Algorithmus '{algorithm}'")
        print("Verfügbare Algorithmen: PPO, SAC, APPO")
        ray.shutdown()
        return

    # Training
    print(f"\nStarte Training für {total_timesteps} Timesteps...")
    print("=" * 60)

    # Berechne Anzahl der Iterationen
    steps_per_iteration = config.train_batch_size
    num_iterations = max(1, total_timesteps // steps_per_iteration)

    algo = None
    try:
        # Erstelle Algorithmus
        algo = config.build()

        # Training-Loop
        for i in range(num_iterations):
            result = algo.train()

            # Logging
            if i % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Iteration {i}/{num_iterations}")
                print(f"{'='*60}")

                # RLlib 2.5+ verwendet 'env_runners' statt 'episode_*'
                env_runners = result.get('env_runners', {})
                print(f"Episode Reward Mean: {env_runners.get('episode_return_mean', 0.0):.2f}")
                print(f"Episode Reward Min:  {env_runners.get('episode_return_min', 0.0):.2f}")
                print(f"Episode Reward Max:  {env_runners.get('episode_return_max', 0.0):.2f}")
                print(f"Episode Length Mean: {env_runners.get('episode_len_mean', 0.0):.1f}")
                print(f"Episodes:            {env_runners.get('num_episodes', 0):.0f}")
                print(f"Env Steps Sampled:   {env_runners.get('num_env_steps_sampled', 0):.0f}")
                print(f"Env Steps Lifetime:  {result.get('num_env_steps_sampled_lifetime', 0):.0f}")

            # Checkpoint speichern
            if i % 50 == 0 and i > 0:
                checkpoint_dir = algo.save(save_path)
                print(f"\nCheckpoint gespeichert: {checkpoint_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining unterbrochen!")

    # Finales Modell speichern
    if algo is not None:
        final_checkpoint = algo.save(save_path)
        print(f"\nFinales Modell gespeichert: {final_checkpoint}")
        algo.stop()

    ray.shutdown()

    return algo


def evaluate_model(model_path, episodes=5, render=True, algorithm='PPO'):
    """
    Evaluiert ein trainiertes Modell.

    Args:
        model_path: Pfad zum gespeicherten Checkpoint
        episodes: Anzahl der Test-Episoden
        render: Ob visualisiert werden soll
        algorithm: Welcher Algorithmus verwendet wurde
    """
    try:
        from ray.rllib.algorithms.ppo import PPO
        from ray.rllib.algorithms.sac import SAC
        from ray.rllib.algorithms.appo import APPO
        from src.drone_env import DroneEnv
        import ray
    except ImportError:
        print("ERROR: Ray RLlib nicht installiert!")
        return

    # Ray initialisieren
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_gpus=0)

    # Lade Algorithmus
    model_path   = os.path.abspath(model_path)
    print(f"Lade Modell von {model_path}...")

    algo_class = {'PPO': PPO, 'SAC': SAC, 'APPO': APPO}.get(algorithm)
    if algo_class is None:
        print(f"ERROR: Unbekannter Algorithmus '{algorithm}'")
        ray.shutdown()
        return

    try:
        algo = algo_class.from_checkpoint(model_path)
        print(f"Modell erfolgreich geladen als {algorithm}")
    except Exception as e:
        print(f"ERROR: Konnte Modell nicht laden: {e}")
        ray.shutdown()
        return

    # Environment
    render_mode = "human" if render else None
    env = DroneEnv(max_steps=500, render_mode=render_mode)

    # Hole RLModule für Inferenz (neue API in RLlib 2.5+)
    rl_module = algo.get_module()

    episode_rewards = []
    episode_distances = []

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        distances = []

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{episodes}")
        print(f"Start-Distanz: {info['distance_to_target']:.2f}m")

        done = False
        step = 0

        while not done:
            # Verwende RLModule.forward_inference für die neue API
            import torch

            # Konvertiere Observation zu Tensor und füge Batch-Dimension hinzu
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            # Forward pass durch RLModule
            with torch.no_grad():
                output = rl_module.forward_inference({"obs": obs_tensor})

            # Extrahiere Action (deterministisch)
            if "actions" in output:
                action = output["actions"].cpu().numpy()[0]
            elif "action_dist_inputs" in output:
                # Für deterministische Policy: verwende Mittelwert
                # action_dist_inputs enthält [mean, log_std] konkateniert
                # Wir brauchen nur die erste Hälfte (mean)
                action_dist = output["action_dist_inputs"].cpu().numpy()[0]
                action_size = len(action_dist) // 2
                action = action_dist[:action_size]  # Nur Mittelwert
            else:
                raise ValueError(f"Unerwartetes Output-Format: {output.keys()}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            distances.append(info['distance_to_target'])
            step += 1

            if render:
                env.render()

            if step % 100 == 0:
                print(f"  Step {step}: Distanz={info['distance_to_target']:.2f}m")

        min_dist = np.min(distances)
        avg_dist = np.mean(distances)
        final_dist = distances[-1]

        print(f"\nErgebnisse:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Durchschn. Distanz: {avg_dist:.2f}m")
        print(f"  Min. Distanz: {min_dist:.2f}m")
        print(f"  Finale Distanz: {final_dist:.2f}m")
        if info.get('crashed'):
            print(f"  ⚠️  Episode endete mit Crash!")

        episode_rewards.append(total_reward)
        episode_distances.append(avg_dist)

    env.close()

    # Zusammenfassung
    print(f"\n{'='*60}")
    print(f"Zusammenfassung über {episodes} Episoden:")
    print(f"{'='*60}")
    print(f"Durchschn. Total Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Durchschn. Distanz: {np.mean(episode_distances):.2f} ± {np.std(episode_distances):.2f}m")
    print(f"Beste Episode Reward: {np.max(episode_rewards):.2f}")

    # Cleanup
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drohnen-RL Training mit Ray RLlib')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Modus: train oder eval')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'APPO'],
                        help='RL-Algorithmus')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Anzahl der Trainings-Timesteps')
    parser.add_argument('--model-path', type=str, default='models/drone_model',
                        help='Pfad zum Modell (zum Speichern/Laden)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Anzahl der Eval-Episoden')
    parser.add_argument('--no-render', action='store_true',
                        help='Deaktiviere Rendering bei Evaluation')

    args = parser.parse_args()

    if args.mode == 'train':
        print("=" * 60)
        print("Drohnen-RL Training mit Ray RLlib")
        print("=" * 60)
        print(f"Algorithmus: {args.algorithm}")
        print(f"Timesteps: {args.timesteps}")
        print(f"Modell-Pfad: {args.model_path}")
        print("=" * 60)

        train_with_rllib(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            save_path=args.model_path
        )

    elif args.mode == 'eval':
        print("=" * 60)
        print("Drohnen-RL Evaluation mit Ray RLlib")
        print("=" * 60)
        print(f"Algorithmus: {args.algorithm}")
        print(f"Modell-Pfad: {args.model_path}")
        print(f"Episoden: {args.episodes}")
        print(f"Rendering: {'OFF' if args.no_render else 'ON'}")
        print("=" * 60)

        evaluate_model(
            model_path=args.model_path,
            episodes=args.episodes,
            render=not args.no_render,
            algorithm=args.algorithm
        )

