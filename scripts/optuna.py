import multiprocessing
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import optuna
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.storages.journal import JournalFileBackend, JournalStorage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from sb3.trpor import TRPOR


class TrialEvalCallback(EvalCallback):
  def __init__(
    self,
    eval_env,
    trial: optuna.Trial,
    n_eval_episodes: int = 10,
    eval_freq: int = 20000,
    deterministic: bool = True,
    verbose: int = 0,
  ):
    super().__init__(
      eval_env=eval_env,
      n_eval_episodes=n_eval_episodes,
      eval_freq=eval_freq,
      deterministic=deterministic,
      verbose=verbose,
    )
    self.trial = trial
    self.eval_idx = 0

  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      super()._on_step()  # Updates self.last_mean_reward

      self.eval_idx += 1
      self.trial.report(self.last_mean_reward, self.eval_idx)

      if self.trial.should_prune():
        raise optuna.TrialPruned()

    return True


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
    return env

  return _init


def suggest_hyperparams(trial):
  params = {}
  params["ent_coef"] = trial.suggest_float("ent_coef", 0.0001, 1, log=True)
  params["target_kl"] = trial.suggest_float("target_kl", 0.01, 0.5, log=True)
  params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
  params["sub_sampling_factor"] = trial.suggest_categorical("sub_sampling_factor", [0.1, 0.5, 1.0])
  params["n_critic_updates"] = trial.suggest_categorical("n_critic_updates", [5, 10])
  params["net_arch_str"] = "small"
  params["gamma"] = 0.99
  params["gae_lambda"] = 0.95
  params["cg_max_steps"] = 5
  params["cg_damping"] = 0.1
  params["batch_size"] = 256
  params["n_steps"] = 2048
  return params


def objective(trial, env_id):
  params = suggest_hyperparams(trial)

  if params["net_arch_str"] == "small":
    net_arch = dict(pi=[64, 64], vf=[64, 64])
  else:
    net_arch = dict(pi=[128, 128], vf=[128, 128])
  policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.Tanh)

  # YOUR setup: 14 envs with noise for BOTH training and evaluation
  n_envs = 14
  noise_type = "uniform"
  noise_level = 0.1

  # Training env
  train_env = SubprocVecEnv([make_env(env_id, noise_type, noise_level) for _ in range(n_envs)])
  train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

  # Evaluation env: SAME as training — noisy, 14 parallel envs, norm_reward=True
  eval_env = SubprocVecEnv([make_env(env_id, noise_type, noise_level) for _ in range(n_envs)])
  eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

  model = TRPOR(
    policy="MlpPolicy",
    env=train_env,
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

  n_timesteps = 1_000_000
  eval_freq = 20_000
  n_eval_episodes = 10

  eval_callback = TrialEvalCallback(
    eval_env=eval_env,
    trial=trial,
    n_eval_episodes=n_eval_episodes,
    eval_freq=eval_freq,
    deterministic=True,
  )

  model.learn(total_timesteps=n_timesteps, callback=eval_callback)

  return eval_callback.last_mean_reward  # Optuna maximizes this (normalized reward under noise)


if __name__ == "__main__":
  multiprocessing.set_start_method("spawn", force=True)

  envs = ["HalfCheetah-v5", "Hopper-v5", "Swimmer-v5", "Walker2d-v5", "Humanoid-v5", "HumanoidStandup-v5"]
  batch = 1000

  optuna_dir = os.path.expanduser("~/optuna")
  os.makedirs(optuna_dir, exist_ok=True)

  for env_id in envs:
    study_name = f"trpor-{env_id}-tuning"
    file_path = os.path.join(optuna_dir, f"{study_name}_log")

    storage = JournalStorage(JournalFileBackend(file_path))
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    sampler = optuna.samplers.TPESampler(n_startup_trials=5)

    study = optuna.create_study(
      study_name=study_name,
      direction="maximize",
      storage=storage,
      pruner=pruner,
      sampler=sampler,
      load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, env_id), n_trials=batch)
