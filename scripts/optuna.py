import multiprocessing
import os

import gymnasium as gym
import numpy as np
import optuna
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from sb3.trpor import TRPOR


class ActionNoiseWrapper(gym.ActionWrapper):
  def __init__(self, env, noise_type, noise_level):
    super().__init__(env)
    self.noise_type = noise_type
    self.noise_level = noise_level

  def action(self, action):
    if self.noise_type == "uniform":
      noise = np.random.uniform(-self.noise_level, self.noise_level, action.shape)
      action += noise
    action = np.clip(action, self.action_space.low, self.action_space.high)
    return action


def make_env(env_id, noise_type=None, noise_level=None):
  def _init():
    env = gym.make(env_id)
    if noise_type and noise_level:
      env = ActionNoiseWrapper(env, noise_type, noise_level)
    env = Monitor(env)
    return env

  return _init


ent_coef_options = [
  # 0.0,
  0.0001,
  0.0003,
  0.0005,
  0.0007,
  0.0009,
  0.001,
  0.003,
  0.005,
  0.007,
  0.009,
  0.01,
  0.03,
  0.05,
  0.07,
  0.09,
  0.1,
  0.3,
  0.5,
  0.7,
  0.9,
]
learning_rate_options = [0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.03, 0.07, 0.1, 0.3, 0.5]


def suggest_hyperparams(trial):
  params = {}
  params["ent_coef"] = trial.suggest_categorical("ent_coef", ent_coef_options)
  params["target_kl"] = trial.suggest_categorical("target_kl", [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5])
  params["learning_rate"] = trial.suggest_categorical("learning_rate", learning_rate_options)
  params["n_critic_updates"] = trial.suggest_categorical("n_critic_updates", [5, 10])
  params["sub_sampling_factor"] = trial.suggest_categorical("sub_sampling_factor", [0.1, 0.5, 1.0])
  params["net_arch_str"] = trial.suggest_categorical("net_arch", ["small", "medium"])
  params["gamma"] = 0.99
  params["gae_lambda"] = 0.95
  params["cg_max_steps"] = 5
  params["cg_damping"] = 0.1
  params["batch_size"] = 512
  params["n_steps"] = 2048
  return params


def auc(learning_curve):
  partial_auc = 0.0
  for i in range(1, len(learning_curve)):
    x1, y1 = learning_curve[i - 1]
    x2, y2 = learning_curve[i]
    partial_auc += (y1 + y2) / 2 * (x2 - x1)
  return partial_auc


def recent_mean_reward(vec_env, prev_episode_counts, n_envs):
  current_rewards_lists = vec_env.venv.get_attr("episode_returns")
  recent_rewards = []
  for i in range(n_envs):
    recent = current_rewards_lists[i][prev_episode_counts[i] :]
    recent_rewards.extend(recent)
    prev_episode_counts[i] += len(recent)
  if len(recent_rewards) > 0:
    recent_mean = np.mean(recent_rewards)
  else:
    recent_mean = 0.0
  return recent_mean


def objective(trial, env_id):
  params = suggest_hyperparams(trial)
  if params["net_arch_str"] == "small":
    net_arch = dict(pi=[64, 64], vf=[64, 64])
  else:
    net_arch = dict(pi=[128, 128], vf=[128, 128])
  policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.Tanh)
  n_envs = 14
  noise_type = "uniform"
  noise_level = 0.1
  vec_env = SubprocVecEnv([make_env(env_id, noise_type, noise_level) for _ in range(n_envs)])
  vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
  model = TRPOR(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=params["learning_rate"],
    n_steps=params["n_steps"],
    batch_size=params["batch_size"],
    gamma=params["gamma"],
    gae_lambda=params["gae_lambda"],
    ent_coef=params["ent_coef"],
    vf_coef=0.5,
    max_kl=params["target_kl"],
    cg_damping=params["cg_damping"],
    cg_max_steps=params["cg_max_steps"],
    vf_iters=params["n_critic_updates"],
    sub_sampling_factor=params["sub_sampling_factor"],
    policy_kwargs=policy_kwargs,
    verbose=0,
    seed=42,
  )
  total_timesteps = 1_000_000
  eval_interval = 50_000
  warmup_timesteps = 200_000
  learning_curve = [(0, 0.0)]
  prev_episode_counts = [0] * n_envs
  for step in range(eval_interval, total_timesteps + 1, eval_interval):
    if step <= warmup_timesteps:
      model.learn(total_timesteps=eval_interval, reset_num_timesteps=False, progress_bar=False)
    else:
      model.learn(total_timesteps=eval_interval, reset_num_timesteps=False, progress_bar=False)
    recent_mean = recent_mean_reward(vec_env, prev_episode_counts, n_envs)
    learning_curve.append((step, recent_mean))
    partial_auc = auc(learning_curve)
    trial.report(partial_auc, step)
    if trial.should_prune():
      vec_env.close()
      raise optuna.TrialPruned()
  print(f"Trial {trial.number} - AUC: {partial_auc:.2f}")
  vec_env.close()
  return partial_auc


if __name__ == "__main__":
  multiprocessing.set_start_method("spawn")
  envs = ["HalfCheetah-v5", "Hopper-v5", "Swimmer-v5", "Walker2d-v5", "Humanoid-v5", "HumanoidStandup-v5"]
  batch = 1000
  for env_id in envs:
    study_name = f"trpor-{env_id}-tuning"
    dir_path = os.path.expanduser("~/.optuna")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{study_name}_log")
    backend = JournalFileBackend(file_path)
    storage = JournalStorage(backend)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage, pruner=pruner, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, env_id), n_trials=batch)
