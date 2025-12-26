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
from scripts.experiments import RewardLoggerCallback, load_hyperparams, make_env, make_video_env, record_best_model_video

if __name__ == "__main__":
  # not relevant for research concerns
  n_envs = 8
  n_eval_episodes = 100

  # Experimentation schedule

  # 1M timesteps is sufficient to see the trends in performance for these environments, sampling longer runs 2M, 10M has not shown significant changes in trend
  total_timesteps = 10_000_000
  num_runs = 10  # Start small for testing, can increase to 100

  # Explicitly define the two configs
  all_configs = [
    ("HalfCheetah-v5", TRPOR, "bonus", 0.0001, False),
    ("HalfCheetah-v5", TRPOR, "bonus", 0.0001, True),
    ("HalfCheetah-v5", TRPOR, "penalty", 0.05, False),
    ("HalfCheetah-v5", TRPOR, "penalty", 0.05, True),
    ("HalfCheetah-v5", TRPO, None, None, False),
  ]

  for run in range(num_runs):
    for env_id, Variant, entropy_mode, ent_coef, normalize_entropy in all_configs:

      key = f"{Variant.__name__}_{entropy_mode}_{ent_coef}_{'norm' if normalize_entropy else 'nonorm'}_run{run}"
      compare_dir = "bp_comparison"
      compare_dir = os.path.join(compare_dir, f"{env_id}")
      os.makedirs(compare_dir, exist_ok=True)

      # if key already exist skip
      pkl_path = os.path.join(compare_dir, f"{key}.pkl")
      if os.path.exists(pkl_path):
        print(f"Skipping existing run {key}")

        continue
      else:
        print(f"Starting run {key}")

      env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
      hyperparams = load_hyperparams(Variant, env_id)
      if hyperparams is None:
        hyperparams = {}

      hyperparams["entropy_mode"] = entropy_mode
      hyperparams["ent_coef"] = ent_coef
      hyperparams["normalize_entropy"] = normalize_entropy
      model = Variant("MlpPolicy", env, **hyperparams, verbose=0)

      reward_logger = RewardLoggerCallback()
      model.learn(total_timesteps=total_timesteps, callback=reward_logger)

      model.save(os.path.join(compare_dir, f"{key}.zip"))

      raw_step_rewards = reward_logger.step_rewards
      if len(raw_step_rewards) > total_timesteps:
        raw_step_rewards = raw_step_rewards[:total_timesteps]

      episode_infos = reward_logger.episode_infos

      mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
      env.close()

      # Evaluate inference stability on no-noise env
      clean_env = SubprocVecEnv([make_env(env_id, None) for _ in range(n_envs)])
      clean_mean_reward, clean_std_reward = evaluate_policy(model, clean_env, n_eval_episodes=n_eval_episodes, deterministic=True)

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

      with open(pkl_path, "wb") as f:
        pickle.dump(run_data, f)

  # After all runs, generate and print the hypothesis report
  print("\nGenerating hypothesis report...")

  # Collect data for bonus and penalty modes with norm/nonorm split
  data = {
    "bonus": {
      "nonorm": {"reg_ratios": [], "raw_improvs": [], "ls_success": [], "ls_coeffs": [], "final_rewards": []},
      "norm": {"reg_ratios": [], "raw_improvs": [], "ls_success": [], "ls_coeffs": [], "final_rewards": []},
    },
    "penalty": {
      "nonorm": {"reg_ratios": [], "raw_improvs": [], "ls_success": [], "ls_coeffs": [], "final_rewards": []},
      "norm": {"reg_ratios": [], "raw_improvs": [], "ls_success": [], "ls_coeffs": [], "final_rewards": []},
    },
  }

  pkl_files = [f for f in os.listdir(compare_dir) if f.endswith(".pkl")]

  for f in pkl_files:
    parts = f.split("_")
    if len(parts) != 5:
      continue  # Skip invalid files
    variant = parts[0]
    entropy_mode = parts[1]
    ent_coef_str = parts[2]
    norm_str = parts[3]
    run_str = parts[4].replace(".pkl", "")

    try:
      ent_coef = float(ent_coef_str)
      run = int(run_str.replace("run", ""))
    except ValueError:
      continue  # Skip invalid parses

    pkl_path = os.path.join(compare_dir, f)
    with open(pkl_path, "rb") as ff:
      run_data = pickle.load(ff)

    # Extraction: rollout_metrics is dict of lists (per-update means)
    rollout_metrics = run_data["rollout_metrics"]

    # Get mean over updates for the run
    reg_ratios = rollout_metrics.get("reg_ratio_mean", [])
    raw_improvs = rollout_metrics.get("raw_improv_mean", [])
    ls_success = rollout_metrics.get("line_search_success_mean", [])
    ls_coeffs = rollout_metrics.get("ls_coeff_mean", [])

    mode_data = data[entropy_mode][norm_str]
    mode_data["reg_ratios"].append(np.mean(reg_ratios) if reg_ratios else 0)
    mode_data["raw_improvs"].append(np.mean(raw_improvs) if raw_improvs else 0)
    mode_data["ls_success"].append(np.mean(ls_success) if ls_success else 0)
    mode_data["ls_coeffs"].append(np.mean(ls_coeffs) if ls_coeffs else 0)

    mode_data["final_rewards"].append(run_data["inference_mean_reward"])

  # Compute aggregates (mean ± std across runs)
  def compute_agg(data_dict, key):
    values = data_dict[key]
    if values:
      return f"{np.mean(values):.4f} ± {np.std(values):.4f}"
    return "N/A"

  # Print report
  print("\nHypothesis Report: Bonus vs Penalty Mode (Split by Norm/Nonorm)")
  print("==================================================================")
  print("Note: Bonus uses ent_coef=0.0001, Penalty uses ent_coef=0.05")
  print("Hypothesis 1: Bonus mode leads to entropy hijacking (high reg_ratio, low raw_improvement, high ls_coeff ~1)")
  print("Hypothesis 2: Penalty mode avoids hijacking (low reg_ratio, high raw_improvement, lower ls_coeff, better performance)")
  print("\nMetric                  | Bonus-Nonorm        | Bonus-Norm          | Penalty-Nonorm      | Penalty-Norm")
  print("------------------------|---------------------|---------------------|---------------------|-------------")
  print(
    f"Reg Ratio (mean±std)    | {compute_agg(data['bonus']['nonorm'], 'reg_ratios')} | {compute_agg(data['bonus']['norm'], 'reg_ratios')} | {compute_agg(data['penalty']['nonorm'], 'reg_ratios')} | {compute_agg(data['penalty']['norm'], 'reg_ratios')}"
  )
  print(
    f"Raw Improvement         | {compute_agg(data['bonus']['nonorm'], 'raw_improvs')} | {compute_agg(data['bonus']['norm'], 'raw_improvs')} | {compute_agg(data['penalty']['nonorm'], 'raw_improvs')} | {compute_agg(data['penalty']['norm'], 'raw_improvs')}"
  )
  print(
    f"LS Success Rate         | {compute_agg(data['bonus']['nonorm'], 'ls_success')} | {compute_agg(data['bonus']['norm'], 'ls_success')} | {compute_agg(data['penalty']['nonorm'], 'ls_success')} | {compute_agg(data['penalty']['norm'], 'ls_success')}"
  )
  print(
    f"Avg LS Coeff            | {compute_agg(data['bonus']['nonorm'], 'ls_coeffs')} | {compute_agg(data['bonus']['norm'], 'ls_coeffs')} | {compute_agg(data['penalty']['nonorm'], 'ls_coeffs')} | {compute_agg(data['penalty']['norm'], 'ls_coeffs')}"
  )
  print(
    f"Final Mean Reward       | {compute_agg(data['bonus']['nonorm'], 'final_rewards')} | {compute_agg(data['bonus']['norm'], 'final_rewards')} | {compute_agg(data['penalty']['nonorm'], 'final_rewards')} | {compute_agg(data['penalty']['norm'], 'final_rewards')}"
  )
  print("==================================================================")
