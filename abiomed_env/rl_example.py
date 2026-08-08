#!/usr/bin/env python3
"""
Example script demonstrating how to use the AbiomedRLEnv for reinforcement learning.

This script shows how to:
1. Create the RL environment
2. Train a simple random policy
3. Evaluate the policy
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import argparse
import json
import pickle
import gym
from datetime import datetime
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.logger import configure
from rl_env import AbiomedRLEnvFactory
import os

class RandomPolicy:
    """Simple random policy for baseline comparison."""
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
    
    def get_env(self):
        return self.env
    
    def get_action(self, observation):
        return self.action_space.sample()
    
    def learn(self, total_timesteps: int = 100):

        """Train a policy for a given number of episodes."""
        episode_rewards = []

        for episode in range(total_timesteps):
            obs, info = self.env.reset()
            total_reward = 0
            
            for step in range(self.env.max_steps):
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Update policy if it supports learning
                if hasattr(self, 'update'):
                    self.update(obs, action, reward, next_obs, terminated or truncated)
                
                total_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                if hasattr(self, 'epsilon'):
                    print(f"Episode {episode}: Average reward (last 10) = {avg_reward:.3f}, Epsilon = {self.epsilon:.3f}")
                else:
                    print(f"Episode {episode}: Average reward (last 10) = {avg_reward:.3f}")

        return episode_rewards


def evaluate_policy(policy, num_episodes: int = 50) -> Dict[str, float]:
    """Evaluate a trained policy."""
    episode_rewards = []
    episode_lengths = []
    env = policy.get_env()

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(env.max_steps):
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards
    }

def evaluate_policy_stable_baselines(policy, num_episodes: int = 50, max_steps: int = 24) -> Dict[str, float]:
    """Evaluate a trained policy using stable baselines."""

    env = policy.get_env()
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
        print(f"Wrapped environment in DummyVecEnv")

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = policy.predict(obs)
            obs, reward, dones, info = env.step(action)
            total_reward += reward
            steps += 1
    
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    episode_rewards = np.array(episode_rewards).flatten()

    return {
        "mean_reward": np.mean(episode_rewards).item(),
        "std_reward": np.std(episode_rewards).item(),
        "min_reward": np.min(episode_rewards).item(),
        "max_reward": np.max(episode_rewards).item(),
        "mean_length": np.mean(episode_lengths).item(),
        "episode_rewards": episode_rewards.tolist()
    }
    
    


def main():
    parser = argparse.ArgumentParser(description="RL Environment Example")
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--train_episodes", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--policy_type", type=str, default="random", 
                       choices=["random","sac", "dqn", "ppo", "a2c"])
    parser.add_argument("--normalize_rewards", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--results_path", type=str, default="results/")

    args = parser.parse_args()
    
    print("Creating RL environment...")
    
    if args.policy_type == "sac":
        action_space_type = "continuous"
    else:
        action_space_type = "discrete"

    env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path,
        max_steps=args.max_steps,
        action_space_type=action_space_type,
        reward_type="smooth",
        normalize_rewards=args.normalize_rewards,
        seed=42
    )
    
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    model_device = env.world_model.device
    print(f"World model device: {model_device}")
    
    policy_kwargs = {
        "batch_size": args.batch_size,
    }

    # Create policy
    if args.policy_type == "random":
        policy = RandomPolicy(env)
        print("Using random policy")
    elif args.policy_type == "sac":
        policy = SAC("MlpPolicy", env, verbose=1, device=model_device, **policy_kwargs)
        print("Using SAC policy")
    elif args.policy_type == "ppo":
        policy = PPO("MlpPolicy", env, verbose=1, device=model_device, **policy_kwargs)
        print("Using PPO policy")
    
    if args.policy_type != "random":
        print(f"Policy created on device: {policy.device}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    log_results_name = f"{args.policy_type}_{time_str}"
    if args.policy_type in ["ppo", "sac"]:
        tmp_path = f"{args.results_path}/log/{log_results_name}"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        policy.set_logger(new_logger)
    
    # Train policyπ
    print(f"\nTraining policy for {args.train_episodes} episodes...")
    policy.learn(total_timesteps=args.train_episodes*args.max_steps)
    
    # Save model if specified
    if hasattr(policy, 'save'):
        policy.save(f"{args.results_path}/{log_results_name}")
        print(f"Model saved to {args.results_path}/{log_results_name}")
    
    # Evaluate policy
    print(f"\nEvaluating policy for {args.eval_episodes} episodes...")
    if args.policy_type in ["ppo", "sac"]:
        eval_results = evaluate_policy_stable_baselines(policy, args.eval_episodes)
    else:
        eval_results = evaluate_policy(policy, args.eval_episodes)
    
    print(f"\nEvaluation Results:")
    print(f"Mean reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
    print(f"Min reward: {eval_results['min_reward']:.3f}")
    print(f"Max reward: {eval_results['max_reward']:.3f}")
    print(f"Mean episode length: {eval_results['mean_length']:.1f}")
    
    # Save results
    results = {
        "policy_type": args.policy_type,
        "model_name": args.model_name,
        "max_steps": args.max_steps,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "policy_results": eval_results,
    }
    

    with open(f"results/{log_results_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(f"results/{log_results_name}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {log_results_name}.json and {log_results_name}.pkl")
    
    env.close()


if __name__ == "__main__":
    main() 