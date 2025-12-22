"""
Beispiel-Training mit Stable-Baselines3.

Installation:
    pip install stable-baselines3[extra]

Ausführung:
    python training.py --algorithm PPO --timesteps 100000
"""
import sys
from pathlib import Path

# Füge src/ zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from src.drone_env import DroneEnv


def train_with_sb3(algorithm='PPO', total_timesteps=100000, save_path='models/drone_model'):
    """
    Trainiert einen RL-Agenten mit Stable-Baselines3.

    Args:
        algorithm: RL-Algorithmus ('PPO', 'SAC', 'TD3')
        total_timesteps: Anzahl der Trainings-Steps
        save_path: Pfad zum Speichern des Modells
    """
    try:
        from stable_baselines3 import PPO, SAC, TD3
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        import os
    except ImportError:
        print("ERROR: Stable-Baselines3 nicht installiert!")
        print("Installation mit: pip install stable-baselines3[extra]")
        return

    # Environment erstellen
    print(f"Erstelle Drohnen-Environment...")
    env = DroneEnv(max_steps=500, render_mode=None)

    # Eval-Environment
    eval_env = DroneEnv(max_steps=500, render_mode=None)

    # Modell erstellen
    print(f"Erstelle {algorithm}-Modell...")

    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
        )
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
        )
    else:
        print(f"ERROR: Unbekannter Algorithmus '{algorithm}'")
        return

    # Callbacks
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best',
        log_path='./logs/eval',
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/checkpoints',
        name_prefix=f'{algorithm.lower()}_drone'
    )

    # Training
    print(f"\nStarte Training für {total_timesteps} Timesteps...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining unterbrochen!")

    # Modell speichern
    model.save(save_path)
    print(f"\nModell gespeichert unter: {save_path}")

    # Cleanup
    env.close()
    eval_env.close()

    return model


def evaluate_model(model_path, episodes=5, render=True):
    """
    Evaluiert ein trainiertes Modell.

    Args:
        model_path: Pfad zum gespeicherten Modell
        episodes: Anzahl der Test-Episoden
        render: Ob visualisiert werden soll
    """
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except ImportError:
        print("ERROR: Stable-Baselines3 nicht installiert!")
        return

    # Versuche verschiedene Algorithmen zu laden
    model = None
    for algorithm in [PPO, SAC, TD3]:
        try:
            model = algorithm.load(model_path)
            print(f"Modell geladen als {algorithm.__name__}")
            break
        except:
            continue

    if model is None:
        print(f"ERROR: Konnte Modell nicht laden von {model_path}")
        return

    # Environment
    render_mode = "human" if render else None
    env = DroneEnv(max_steps=500, render_mode=render_mode)

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
            action, _states = model.predict(obs, deterministic=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drohnen-RL Training mit Stable-Baselines3')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Modus: train oder eval')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'],
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
        print("Drohnen-RL Training")
        print("=" * 60)
        print(f"Algorithmus: {args.algorithm}")
        print(f"Timesteps: {args.timesteps}")
        print(f"Modell-Pfad: {args.model_path}")
        print("=" * 60)

        train_with_sb3(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            save_path=args.model_path
        )

    elif args.mode == 'eval':
        print("=" * 60)
        print("Drohnen-RL Evaluation")
        print("=" * 60)
        print(f"Modell-Pfad: {args.model_path}")
        print(f"Episoden: {args.episodes}")
        print(f"Rendering: {'OFF' if args.no_render else 'ON'}")
        print("=" * 60)

        evaluate_model(
            model_path=args.model_path,
            episodes=args.episodes,
            render=not args.no_render
        )

