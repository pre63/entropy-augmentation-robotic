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


def record_best_model_video(variant_name, Variant, compare_dir, env_id, noise_configs=None):
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
    video_env = RecordVideo(video_env, compare_dir, name_prefix=f"{variant_name}_best", episode_trigger=lambda x: True)
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
  all_ts = [1_000_000, 2_000_000, 10_000_000]

  for idx, (env_id, Variant, noise_level) in enumerate(configs):
    compare_dir = os.path.join(base_compare_dir, f"{env_id}_ts{total_timesteps}_runs{num_runs}".lower())
    os.makedirs(compare_dir, exist_ok=True)

    variant_base_name = Variant.__name__
    variant_name = variant_base_name if noise_level is None else f"{variant_base_name}_noise{noise_level}"
    print(f"Training variant {variant_name} on {env_id}")
    for run in range(num_runs):
      key = f"{variant_name}_run{run}"
      pkl_path = os.path.join(compare_dir, f"{key}.pkl")

      # New logic: check for higher steps pkl and truncate if exists
      higher_ts_list = sorted([t for t in all_ts if t > total_timesteps], reverse=True)
      truncated = False
      for higher_ts in higher_ts_list:
        print(f"  Checking for higher ts {higher_ts} for possible truncation...")
        higher_dir = os.path.join(base_compare_dir, f"{env_id}_ts{higher_ts}_runs{num_runs}".lower())
        higher_pkl = os.path.join(higher_dir, f"{key}.pkl")
        if os.path.exists(higher_pkl):
          with open(higher_pkl, "rb") as f:
            data = pickle.load(f)
          trunc_ts = total_timesteps
          if len(data["timesteps"]) < trunc_ts:
            continue
          data["timesteps"] = data["timesteps"][:trunc_ts]
          data["step_rewards"] = data["step_rewards"][:trunc_ts]
          keep_idx = [i for i, e in enumerate(data["episode_end_timesteps"]) if e <= trunc_ts]
          data["episode_end_timesteps"] = [data["episode_end_timesteps"][i] for i in keep_idx]
          data["episode_rewards"] = [data["episode_rewards"][i] for i in keep_idx]
          if "inference_mean_reward" in data:
            del data["inference_mean_reward"]
          if "inference_std_reward" in data:
            del data["inference_std_reward"]
          with open(pkl_path, "wb") as wf:
            pickle.dump(data, wf)
          print(f"  Truncated {key} from {higher_ts} to {total_timesteps}")
          truncated = True
          break

      if truncated:
        continue

      # Existing check
      if os.path.exists(pkl_path):
        print(f"  Skipping {key}, already done")
        continue

      print(f"  Run {run+1}/{num_runs} for {variant_name}")
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

        model = Variant("MlpPolicy", env, verbose=1, device="cpu", **params)
        callback = RewardLoggerCallback()
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10000)

        model.save(os.path.join(compare_dir, f"{key}.zip"))

        raw_step_rewards = callback.step_rewards
        if len(raw_step_rewards) > total_timesteps:
          raw_step_rewards = raw_step_rewards[:total_timesteps]

        episode_infos = callback.episode_infos

        # Evaluate inference stability
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)

        run_data = {
          "timesteps": list(range(1, len(raw_step_rewards) + 1)),
          "step_rewards": raw_step_rewards,
          "episode_rewards": [ep["reward"] for ep in episode_infos],
          "episode_end_timesteps": [ep["end_timestep"] for ep in episode_infos],
          "inference_mean_reward": float(mean_reward),
          "inference_std_reward": float(std_reward),
        }

        # Save to individual pkl
        with open(pkl_path, "wb") as write_f:
          pickle.dump(run_data, write_f)

        env.close()
      except Exception as e:
        print(f"  Error during training {key}: {e}")
    params = load_hyperparams(Variant, env_id)

    # Record video for the best model after all runs
    record_best_model_video(variant_name, Variant, compare_dir, env_id, noise_configs=params.get("noise_configs", None))


if __name__ == "__main__":
  n_envs = 8
  num_runs = 100
  n_eval_episodes = 100

  configs = [
    # ===================================================================
    # Humanoid-v5
    # ===================================================================
    ("Humanoid-v5", TRPO, None),
    ("Humanoid-v5", TRPOR, None),
    # ===================================================================
    # HalfCheetah-v5
    # ===================================================================
    ("HalfCheetah-v5", TRPO, None),
    ("HalfCheetah-v5", TRPOR, None),
    # ===================================================================
    # Hopper-v5
    # ===================================================================
    ("Hopper-v5", TRPO, None),
    ("Hopper-v5", TRPOR, None),
    # ===================================================================
    # Swimmer-v5
    ("Swimmer-v5", TRPO, None),
    ("Swimmer-v5", TRPOR, None),
  ]
  for i in [0.1, 0.2]:
    configs.append(("HalfCheetah-v5", TRPOR, i))
    configs.append(("HalfCheetah-v5", TRPO, i))
    configs.append(("Hopper-v5", TRPOR, i))
    configs.append(("Hopper-v5", TRPO, i))
    configs.append(("Swimmer-v5", TRPOR, i))
    configs.append(("Swimmer-v5", TRPO, i))

  for i in [0.1, 0.2]:
    configs.append(("Humanoid-v5", TRPO, i))
    configs.append(("Humanoid-v5", TRPOR, i))

  for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    configs.append(("HalfCheetah-v5", TRPOR, i))
    configs.append(("HalfCheetah-v5", TRPO, i))
    configs.append(("Hopper-v5", TRPOR, i))
    configs.append(("Hopper-v5", TRPO, i))
    configs.append(("Swimmer-v5", TRPOR, i))
    configs.append(("Swimmer-v5", TRPO, i))
    configs.append(("Humanoid-v5", TRPO, i))
    configs.append(("Humanoid-v5", TRPOR, i))

  # ===================================================================
  # HumanoidStandup-v5
  # ===================================================================
  configs.extend(
    [
      ("HumanoidStandup-v5", TRPO, None),
      ("HumanoidStandup-v5", TRPOR, None),
    ]
  )

  for i in [0.1, 0.2]:
    configs.append(("HumanoidStandup-v5", TRPOR, i))
    configs.append(("HumanoidStandup-v5", TRPO, i))

  for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    configs.append(("HumanoidStandup-v5", TRPOR, i))
    configs.append(("HumanoidStandup-v5", TRPO, i))

  # experiments
  M1 = 1_000_000

  run_experiment(configs, n_envs, M1, num_runs, n_eval_episodes)
