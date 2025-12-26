import copy
import datetime
import errno
import fcntl
import itertools
import os
import pickle
import random
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from sb3.noise import MonitoredEntropyInjectionWrapper
from sb3.trpo import TRPO
from sb3.trpor import TRPOR


class RewardLoggerCallback(BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose)
    self.step_rewards = []
    self.episode_infos = []

  def _on_step(self) -> bool:
    rewards = self.locals["rewards"]
    dones = self.locals["dones"]
    infos = self.locals["infos"]
    self.step_rewards.extend([float(r) for r in rewards])
    for i in range(len(dones)):
      if dones[i] and "episode" in infos[i]:
        ep_reward = infos[i]["episode"]["r"]
        end_ts = self.num_timesteps
        self.episode_infos.append({"reward": float(ep_reward), "end_timestep": int(end_ts)})
    return True


def make_env(env_id, noise_configs=None):
  def _init():
    env = gym.make(env_id)
    if noise_configs:
      env = MonitoredEntropyInjectionWrapper(env, noise_configs=noise_configs)
    env = Monitor(env)
    return env

  return _init


def make_video_env(env_id, noise_configs=None):
  env = gym.make(env_id, render_mode="rgb_array")
  if noise_configs:
    env = MonitoredEntropyInjectionWrapper(env, noise_configs=noise_configs)
  return env


def record_best_model_video(variant_name, Variant, compare_dir, env_id, num_runs, noise_configs=None, name_suffix=""):
  max_reward = -float("inf")
  best_run = None
  for run in range(num_runs):
    key = f"{variant_name}_run{run}"
    pkl_path = os.path.join(compare_dir, f"{key}.pkl")
    if os.path.exists(pkl_path):
      with open(pkl_path, "rb") as f:
        run_data = pickle.load(f)
      inf_mean = run_data.get("inference_mean_reward", -float("inf"))
      if inf_mean > max_reward:
        max_reward = inf_mean
        best_run = run

  if best_run is not None:
    best_model_path = os.path.join(compare_dir, f"{variant_name}_run{best_run}")
    video_env = make_video_env(env_id, noise_configs)
    video_env = RecordVideo(video_env, compare_dir, name_prefix=f"{variant_name}_best{name_suffix}", episode_trigger=lambda x: True)
    model = Variant.load(best_model_path, env=video_env)
    obs, info = video_env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = video_env.step(action)
    video_env.close()


def load_hyperparams(model_class, env_id):
  model_name = model_class.__name__.lower()
  yaml_path = f"hyperparameters/{model_name}.yaml"
  if not os.path.exists(yaml_path):
    return None

  with open(yaml_path, "r") as f:
    hyperparams = yaml.safe_load(f)

  return hyperparams.get(env_id, {})


def run_experiment(configs, n_envs, total_timesteps, num_runs, n_eval_episodes):
  base_compare_dir = "assets"

  # Perform training with interleaved runs
  for run in range(num_runs):
    for idx, (env_id, Variant, noise_level) in enumerate(configs):
      compare_dir = os.path.join(base_compare_dir, f"{env_id}".lower())
      os.makedirs(compare_dir, exist_ok=True)

      variant_base_name = Variant.__name__
      variant_name = variant_base_name if noise_level is None else f"{variant_base_name}_noise{noise_level}"
      print(f"Training variant {variant_name} on {env_id} run {run+1}/{num_runs}")

      key = f"{variant_name}_run{run}"
      pkl_path = os.path.join(compare_dir, f"{key}.pkl")
      if os.path.exists(pkl_path):
        print(f"  Skipping existing run for {key}")
        continue

      params = load_hyperparams(Variant, env_id)
      if noise_level is not None:
        params["noise_configs"] = [
          {
            "noise_type": "uniform",
            "noise_level": noise_level,
            "component": "action",
          }
        ]
      try:
        env = SubprocVecEnv([make_env(env_id, params.get("noise_configs", None)) for _ in range(n_envs)])

        # print hyperpameters
        print(f"  Hyperparameters for {key}:")
        for k, v in params.items():
          print(f"    {k}: {v}")

        model = Variant("MlpPolicy", env, verbose=1, device="cpu", **params)
        callback = RewardLoggerCallback()
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10000)

        model.save(os.path.join(compare_dir, f"{key}.zip"))

        raw_step_rewards = callback.step_rewards
        if len(raw_step_rewards) > total_timesteps:
          raw_step_rewards = raw_step_rewards[:total_timesteps]

        episode_infos = callback.episode_infos

        # Evaluate inference stability on training env
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        print(f"    Inference evaluation over {n_eval_episodes} episodes: mean reward = {mean_reward}, std = {std_reward}")

        # Evaluate inference stability on no-noise env
        clean_env = SubprocVecEnv([make_env(env_id, None) for _ in range(n_envs)])
        clean_mean_reward, clean_std_reward = evaluate_policy(model, clean_env, n_eval_episodes=n_eval_episodes, deterministic=True)
        print(f"    Clean inference evaluation over {n_eval_episodes} episodes: mean reward = {clean_mean_reward}, std = {clean_std_reward}")

        run_data = {
          "timesteps": list(range(1, len(raw_step_rewards) + 1)),
          "step_rewards": raw_step_rewards,
          "episode_rewards": [ep["reward"] for ep in episode_infos],
          "episode_end_timesteps": [ep["end_timestep"] for ep in episode_infos],
          "inference_mean_reward": float(mean_reward),
          "inference_std_reward": float(std_reward),
          "clean_inference_mean_reward": float(clean_mean_reward),
          "clean_inference_std_reward": float(clean_std_reward),
          "rollout_metrics": model.rollout_metrics,
        }

        # Save to individual pkl
        with open(pkl_path, "wb") as write_f:
          pickle.dump(run_data, write_f)

        env.close()
        clean_env.close()
      except Exception as e:
        print(f"  Error during training {key}: {e}")
        raise e

  # After all training, record videos
  print("All training done, now recording videos")
  for idx, (env_id, Variant, noise_level) in enumerate(configs):
    compare_dir = os.path.join(base_compare_dir, f"{env_id}".lower())

    variant_base_name = Variant.__name__
    variant_name = variant_base_name if noise_level is None else f"{variant_base_name}_noise{noise_level}"

    params = load_hyperparams(Variant, env_id)
    if noise_level is not None:
      params["noise_configs"] = [
        {
          "noise_type": "uniform",
          "noise_level": noise_level,
          "component": "action",
        }
      ]

    # Record video for the best model
    record_best_model_video(variant_name, Variant, compare_dir, env_id, num_runs, noise_configs=params.get("noise_configs", None), name_suffix="")
    if noise_level is not None:
      record_best_model_video(variant_name, Variant, compare_dir, env_id, num_runs, noise_configs=None, name_suffix="_clean")


if __name__ == "__main__":
  # not relevant for research concerns
  n_envs = 8
  n_eval_episodes = 100

  # Experimentation schedule

  # 1M timesteps is sufficient to see the trends in performance for these environments, sampling longer runs 2M, 10M has not shown significant changes in trend
  timesteps = 1_000_000
  num_runs = 100  # 5  # then 10
  algos = [TRPOR, TRPO]

  # Explicit dict without hardcoded conditional logic, so you can easily replace configs per env

  env_to_noises = {
    "HalfCheetah-v5": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, None],
    "Hopper-v5": [0.05, 0.1, 0.15, 0.2, None],
    "Swimmer-v5": [0.05, 0.1, 0.15, 0.2, None],
    "Walker2d-v5": [0.05, 0.1, 0.15, 0.2, None],
    "Humanoid-v5": [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, None],
    "HumanoidStandup-v5": [0.1, 0.2, 0.3, 0.4, 0.5, None],
  }

  full_sweep_override = [None, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
  for env in env_to_noises.keys():
    env_to_noises[env] = full_sweep_override

  all_configs = []
  for env in env_to_noises.keys():
    noises = env_to_noises[env]
    all_configs.extend(
      list(
        itertools.product(
          [env],
          algos,
          noises,
        )
      )
    )
  run_experiment(all_configs, n_envs, timesteps, num_runs=num_runs, n_eval_episodes=n_eval_episodes)
