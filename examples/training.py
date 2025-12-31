"""
Example training with Ray RLlib.

Installation:
    pip install ray[rllib] gymnasium torch

Usage:
    python training.py --algorithm PPO --timesteps 100000
"""
import logging
import sys
from pathlib import Path

from ray.rllib.env.single_agent_episode import SingleAgentEpisode

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import os
from src.drone_env import RLlibDroneEnv, ThrustChangeController
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.appo import APPO
import ray

logger = logging.getLogger(__name__)

# Environment configuration
env_config = {
    "max_steps": 600, # 30 seconds
    "render_mode": None,
    "enable_crash_detection": False,
    "enable_out_of_bounds_detection": True,
    "dt": 1.0/20, # 20 fps
    "use_wind": True,
    "wrappers": [
        ThrustChangeController,
    ]
}
env_config_eval = env_config.copy()
env_config_eval["render_mode"] = "human"


class CustomMetricsCallback(DefaultCallbacks):
    """Custom callback to log episode metrics to TensorBoard."""

    def on_episode_end(self, episode: SingleAgentEpisode, env_runner=None, metrics_logger=None, env=None, env_index=None, rl_module=None, **kwargs):
        """
        Logs custom metrics at the end of each episode.

        Extracts episode_time from the last info dict and logs it to TensorBoard.
        RLlib will automatically compute min/mean/max across episodes.
        """
        # Get the last info dict from the episode
        last_info = episode.infos[-1]

        if metrics_logger is None:
            logger.warning("metrics_logger is None. Custom metrics won't be visible")
            return

        for key, value in last_info.items():
            metrics_logger.log_value(f"custom_logs/{key}", value)


def train_with_rllib(algorithm='PPO', total_timesteps=100000, save_path='../models/drone_model'):
    """
    Trains an RL agent with Ray RLlib.

    Args:
        algorithm: RL algorithm ('PPO', 'SAC', 'APPO'). Default is 'PPO'.
        total_timesteps: Number of training steps. Default is 100000.
        save_path: Path to save the trained model. Default is '../models/drone_model'.
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_gpus=0)

    # Convert to absolute path (RLlib 2.5+ requires absolute path)
    save_path = os.path.abspath(save_path)

    # Create directories
    os.makedirs(save_path, exist_ok=True)

    print(f"Creating {algorithm} configuration...")

    # Algorithm configuration
    if algorithm == 'PPO':
        config = (
            PPOConfig()
            .environment(RLlibDroneEnv, env_config=env_config)
            .framework("torch")
            .callbacks(CustomMetricsCallback)
            .training(
                lr=3e-4,
                train_batch_size=2048,
                minibatch_size=64,
                num_epochs=10,
                gamma=0.96,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
                use_critic=True,
                use_gae=True,
            )
            .env_runners(num_env_runners=15, num_envs_per_env_runner=1)
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
            .callbacks(CustomMetricsCallback)
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
            .env_runners(num_env_runners=2)
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
            .callbacks(CustomMetricsCallback)
            .training(
                lr=3e-4,
                train_batch_size=2048,
                gamma=0.99,
            )
            .env_runners(num_env_runners=4)
            .resources(num_gpus=0)
        )
    else:
        print(f"ERROR: Unknown algorithm '{algorithm}'")
        print("Available algorithms: PPO, SAC, APPO")
        ray.shutdown()
        return None

    # Training
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("=" * 60)

    # Calculate number of iterations
    steps_per_iteration = config.train_batch_size
    num_iterations = max(1, total_timesteps // steps_per_iteration)

    algo = None
    try:
        # Create algorithm
        algo = config.build()

        # Training loop
        for i in range(num_iterations):
            result = algo.train()

            # Logging
            if i % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Iteration {i}/{num_iterations}")
                print(f"{'='*60}")

                # RLlib 2.5+ uses 'env_runners' instead of 'episode_*'
                env_runners = result.get('env_runners', {})
                print(f"Episode Reward Mean: {env_runners.get('episode_return_mean', 0.0):.2f}")
                print(f"Episode Reward Min:  {env_runners.get('episode_return_min', 0.0):.2f}")
                print(f"Episode Reward Max:  {env_runners.get('episode_return_max', 0.0):.2f}")
                print(f"Episode Length Mean: {env_runners.get('episode_len_mean', 0.0):.1f}")
                print(f"Episodes:            {env_runners.get('num_episodes', 0):.0f}")
                print(f"Env Steps Sampled:   {env_runners.get('num_env_steps_sampled', 0):.0f}")
                print(f"Env Steps Lifetime:  {result.get('num_env_steps_sampled_lifetime', 0):.0f}")

            # Save checkpoint
            if i % 50 == 0 and i > 0:
                checkpoint_dir = algo.save(save_path)
                print(f"\nCheckpoint saved: {checkpoint_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")

    # Save final model
    if algo is not None:
        final_checkpoint = algo.save(save_path)
        print(f"\nFinal model saved: {final_checkpoint}")
        algo.stop()

    ray.shutdown()

    return algo


def evaluate_model(model_path, episodes=5, render=True, algorithm='PPO'):
    """
    Evaluates a trained model.

    Args:
        model_path: Path to the saved checkpoint.
        episodes: Number of test episodes. Default is 5.
        render: Whether to visualize the environment. Default is True.
        algorithm: Which algorithm was used for training ('PPO', 'SAC', 'APPO').
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_gpus=0)

    # Load algorithm
    model_path = os.path.abspath(model_path)
    print(f"Loading model from {model_path}...")

    algo_class = {'PPO': PPO, 'SAC': SAC, 'APPO': APPO}.get(algorithm)
    if algo_class is None:
        print(f"ERROR: Unknown algorithm '{algorithm}'")
        ray.shutdown()
        return

    try:
        algo = algo_class.from_checkpoint(model_path)
        print(f"Model successfully loaded as {algorithm}")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        ray.shutdown()
        return

    # Environment
    env = RLlibDroneEnv(config=env_config_eval)

    # Get RLModule for inference (new API in RLlib 2.5+)
    rl_module = algo.get_module()

    episode_rewards = []
    episode_distances = []

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        distances = []

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{episodes}")
        print(f"Initial distance: {info['distance_to_target']:.2f}m")

        done = False
        step = 0

        while not done:
            # Use RLModule.forward_inference for the new API
            import torch

            # Convert observation to tensor and add batch dimension
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            # Forward pass through RLModule
            with torch.no_grad():
                output = rl_module.forward_inference({"obs": obs_tensor})

            # Extract action (deterministic)
            if "actions" in output:
                action = output["actions"].cpu().numpy()[0]
            elif "action_dist_inputs" in output:
                # For deterministic policy: use mean
                # action_dist_inputs contains [mean, log_std] concatenated
                # We only need the first half (mean)
                action_dist = output["action_dist_inputs"].cpu().numpy()[0]
                action_size = len(action_dist) // 2
                action = action_dist[:action_size]  # Only mean
            else:
                raise ValueError(f"Unexpected output format: {output.keys()}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            distances.append(info['distance_to_target'])
            step += 1

            if render:
                env.render()

            if step % 100 == 0:
                print(f"  Step {step}: Distance={info['distance_to_target']:.2f}m")

        min_dist = np.min(distances)
        avg_dist = np.mean(distances)
        final_dist = distances[-1]

        print(f"\nResults:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Avg. Distance: {avg_dist:.2f}m")
        print(f"  Min. Distance: {min_dist:.2f}m")
        print(f"  Final Distance: {final_dist:.2f}m")
        if info.get('crashed'):
            print(f"  ⚠️  Episode ended with crash!")

        episode_rewards.append(total_reward)
        episode_distances.append(avg_dist)

    env.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary over {episodes} episodes:")
    print(f"{'='*60}")
    print(f"Average Total Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Distance: {np.mean(episode_distances):.2f} ± {np.std(episode_distances):.2f}m")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")

    # Cleanup
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drone RL Training with Ray RLlib')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode: train or eval')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'APPO'],
                        help='RL algorithm')
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Number of training timesteps')
    parser.add_argument('--model-path', type=str, default='models/drone_model',
                        help='Path to model (for saving/loading)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering during evaluation')

    args = parser.parse_args()

    if args.mode == 'train':
        print("=" * 60)
        print("Drone RL Training with Ray RLlib")
        print("=" * 60)
        print(f"Algorithm: {args.algorithm}")
        print(f"Timesteps: {args.timesteps}")
        print(f"Model Path: {args.model_path}")
        print("=" * 60)

        train_with_rllib(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            save_path=args.model_path
        )

    elif args.mode == 'eval':
        print("=" * 60)
        print("Drone RL Evaluation with Ray RLlib")
        print("=" * 60)
        print(f"Algorithm: {args.algorithm}")
        print(f"Model Path: {args.model_path}")
        print(f"Episodes: {args.episodes}")
        print(f"Rendering: {'OFF' if args.no_render else 'ON'}")
        print("=" * 60)

        evaluate_model(
            model_path=args.model_path,
            episodes=args.episodes,
            render=not args.no_render,
            algorithm=args.algorithm
        )

