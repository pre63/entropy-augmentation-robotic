import os

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3.trpor import TRPOR
from scripts.experiments import make_env


def load_hyperparams(model_class, env_id):
  model_name = model_class.__name__.lower()
  yaml_path = f"hyperparameters/{model_name}.yaml"
  if not os.path.exists(yaml_path):
    print(f"Warning: Hyperparams file {yaml_path} not found. Using defaults.")
    return {}
  with open(yaml_path, "r") as f:
    hyperparams = yaml.safe_load(f)
  return hyperparams.get(env_id, {})


def train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level=None):
  if noise_level is None:
    env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
  else:
    noise_configs = [
      {
        "noise_type": "uniform",
        "noise_level": noise_level,
        "component": "action",
      }
    ]
    env = SubprocVecEnv([make_env(env_id, noise_configs) for _ in range(n_envs)])
  hyperparams = load_hyperparams(TRPOR, env_id)
  hyperparams["ent_coef"] = ent_coef
  model = TRPOR("MlpPolicy", env, **hyperparams, verbose=0)
  model.learn(total_timesteps=total_timesteps)
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
  env.close()
  return mean_reward, std_reward


# Define print_report to group by env, compute mean/std of means, sort by mean desc
def print_report(results):
  for env_id in results:
    print(f"\nResults for {env_id}:")
    config_stats = []
    for key, data in results[env_id].items():
      if len(data["means"]) > 0:
        mean_of_means = np.mean(data["means"])
        std_of_means = np.std(data["means"]) if len(data["means"]) > 1 else 0.0
        num_runs = len(data["means"])
        config_stats.append((key, mean_of_means, std_of_means, num_runs))
    sorted_stats = sorted(config_stats, key=lambda x: x[1], reverse=True)
    print("Noise | Ent_Coef | Mean | Std | Num_Runs")
    for (noise, coef), m, s, n in sorted_stats:
      noise_str = "None" if noise is None else noise
      print(f"{noise_str} | {coef} | {m:.2f} | {s:.2f} | {n}")


envs = ["HalfCheetah-v5", "Hopper-v5", "Swimmer-v5", "Walker2d-v5", "Humanoid-v5", "HumanoidStandup-v5"]
ent_coefs = sorted(
  set(
    [
      0.00001,
      0.0001,
      0.0002,
      0.0003,
      0.0004,
      0.0005,
      0.0006,
      0.0007,
      0.0008,
      0.0009,
      0.001,
      0.002,
      0.003,
      0.004,
      0.005,
      0.006,
      0.007,
      0.008,
      0.009,
      0.01,
      0.02,
      0.03,
      0.04,
      0.05,
      0.06,
      0.07,
      0.08,
      0.09,
    ]
  )
)

if __name__ == "__main__":
  dry_run = False  # Set to True for quick testing

  n_envs = 12
  total_timesteps = 1_000_000
  n_eval_episodes = 100
  noise_levels = [0.1]
  n_eval_runs = 3

  results = {}  # env -> {(noise, ent_coef): {"means": [], "stds": []}}

  if dry_run:
    ent_coefs = [0.0002]
    noise_levels = [0.1]
    total_timesteps = 2
    n_eval_episodes = 1

  for env_id in envs:
    results[env_id] = {}
    for noise_level in noise_levels:
      for ent_coef in ent_coefs:
        for run in range(n_eval_runs):
          key = (noise_level, ent_coef)
          results[env_id][key] = {"means": [], "stds": []}
          print(f"Evaluating {env_id} with ent_coef={ent_coef} and noise={noise_level} (initial run)")
          mean_reward, std_reward = train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level)
          results[env_id][key]["means"].append(mean_reward)
          results[env_id][key]["stds"].append(std_reward)
          print(f"  Result: mean: {mean_reward:.2f}, std: {std_reward:.2f}")
          print_report(results)

  print_report(results)

  for env_id in envs:
    config_means = []
    for key in results[env_id]:
      initial_mean = results[env_id][key]["means"][0]
      config_means.append((key, initial_mean))
    sorted_configs = sorted(config_means, key=lambda x: x[1], reverse=True)
    top_4 = sorted_configs[:4]
    for key, _ in top_4:
      noise_level, ent_coef = key
      num_additional_runs = 1 if dry_run else 7
      for run in range(num_additional_runs):
        print(f"Evaluating {env_id} with ent_coef={ent_coef} and noise={noise_level} (additional run {run+1}/{num_additional_runs})")
        mean_reward, std_reward = train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level)
        results[env_id][key]["means"].append(mean_reward)
        results[env_id][key]["stds"].append(std_reward)
        print(f"  Result for additional run {run+1}: mean: {mean_reward:.2f}, std: {std_reward:.2f}")
        running_means = results[env_id][key]["means"]
        running_avg = np.mean(running_means)
        running_std_of_means = np.std(running_means) if len(running_means) > 1 else 0.0
        print(f"  Running average after {len(running_means)} runs: mean: {running_avg:.2f}, std of means: {running_std_of_means:.2f}")
        print_report(results)

  print_report(results)
