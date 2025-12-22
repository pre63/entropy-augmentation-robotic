import os

import gymnasium as gym
import numpy as np
import optuna
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.storages.journal import JournalFileBackend, JournalStorage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from sb3.trpor import TRPOR


def load_default_hyperparams(env_id):
  # read file from hyperparameters/trpor.yaml
  import yaml

  yaml_path = "hyperparameters/trpor.yaml"
  if not os.path.exists(yaml_path):
    print(f"Warning: Hyperparams file {yaml_path} not found. Using defaults.")
    return {}
  with open(yaml_path, "r") as f:
    hyperparams = yaml.safe_load(f)
  return hyperparams.get(env_id, {})


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


def suggest_hyperparams(trial, env_id):

  params = load_default_hyperparams(env_id)
  params["ent_coef"] = trial.suggest_categorical("ent_coef", ent_coefs)
  params["target_kl"] = trial.suggest_categorical("target_kl", [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
  params["learning_rate"] = trial.suggest_categorical("learning_rate", [0.0009, 0.00095, 0.001, 0.0015, 0.002, 0.0025, 0.003])
  params["sub_sampling_factor"] = trial.suggest_categorical("sub_sampling_factor", [0.1, 0.5, 1.0])
  params["n_critic_updates"] = trial.suggest_categorical("n_critic_updates", [5, 10, 20])
  params["cg_max_steps"] = trial.suggest_categorical("cg_max_steps", [5, 25])
  return params


def get_noise_level(env_id):
  noise_levels = {
    "HalfCheetah-v5": 0.05,
    "Walker2d-v5": 0.2,
    "Hopper-v5": 0.05,
    "Swimmer-v5": 0.15,
    "Humanoid-v5": 0.3,
    "HumanoidStandup-v5": 0.2,
  }
  return noise_levels.get(env_id, 0.1)


def objective(trial, env_id):
  n_timesteps = 1_000_000
  eval_freq = 100_000
  n_eval_episodes = 10

  n_envs = 8
  noise_type = "uniform"
  noise_level = get_noise_level(env_id)

  params = suggest_hyperparams(trial, env_id)

  env = SubprocVecEnv([make_env(env_id, noise_type, noise_level) for _ in range(n_envs)])
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
  eval_env = VecNormalize(SubprocVecEnv([make_env(env_id)]), norm_obs=True, norm_reward=True, clip_obs=10.0)

  try:
    model = TRPOR(
      policy="MlpPolicy",
      env=env,
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
      net_arch=params["net_arch"],
      verbose=0,
      seed=42,
      device="cpu",
    )

    eval_callback = TrialEvalCallback(
      eval_env=eval_env,
      trial=trial,
      n_eval_episodes=n_eval_episodes,
      eval_freq=eval_freq,
      deterministic=True,
    )

    model.learn(total_timesteps=n_timesteps, callback=eval_callback)

    return eval_callback.last_mean_reward
  finally:
    env.close()
    eval_env.close()


def objective_ent(trial, params, env_id):
  n_timesteps = 1_000_000
  eval_freq = 100_000
  n_eval_episodes = 10

  n_envs = 8
  noise_type = "uniform"
  noise_level = get_noise_level(env_id)

  ent_coef = trial.suggest_categorical("ent_coef", ent_coefs)

  env = SubprocVecEnv([make_env(env_id, noise_type, noise_level) for _ in range(n_envs)])
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
  eval_env = VecNormalize(SubprocVecEnv([make_env(env_id)]), norm_obs=True, norm_reward=True, clip_obs=10.0)

  try:
    model = TRPOR(
      policy="MlpPolicy",
      env=env,
      learning_rate=params["learning_rate"],
      n_steps=params["n_steps"],
      batch_size=params["batch_size"],
      gamma=params["gamma"],
      gae_lambda=params["gae_lambda"],
      ent_coef=ent_coef,
      vf_coef=0.5,
      max_kl=params["target_kl"],
      cg_damping=params["cg_damping"],
      cg_max_steps=params["cg_max_steps"],
      vf_iters=params["n_critic_updates"],
      sub_sampling_factor=params["sub_sampling_factor"],
      net_arch=params["net_arch"],
      verbose=0,
      seed=42,
      device="cpu",
    )

    eval_callback = TrialEvalCallback(
      eval_env=eval_env,
      trial=trial,
      n_eval_episodes=n_eval_episodes,
      eval_freq=eval_freq,
      deterministic=True,
    )

    model.learn(total_timesteps=n_timesteps, callback=eval_callback)

    return eval_callback.last_mean_reward
  finally:
    env.close()
    eval_env.close()


def main():
  env_id = "HalfCheetah-v5"
  batch = 1000

  optuna_dir = "~optuna"
  os.makedirs(optuna_dir, exist_ok=True)

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

  # Take best parameters and for the other envs run only ent_coef finetuning
  print(f"Best trial for {env_id}:")
  trial = study.best_trial

  print(f"  Value: {trial.value}")

  print("  Params: ")
  for key, value in trial.params.items():
    print(f"    {key}: {value}")

  best_params = trial.params

  batch = 100
  other_envs = ["Walker2d-v5", "Hopper-v5", "Swimmer-v5", "Humanoid-v5", "HumanoidStandup-v5"]
  for env_id in other_envs:
    study_name = f"trpor-{env_id}-entcoef-tuning"
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
    params = load_default_hyperparams(env_id)
    params.update(best_params)
    study.optimize(lambda trial: objective_ent(trial, params, env_id), n_trials=batch)


def generate_trpor_report(optuna_dir="~optuna"):
  optuna_dir = os.path.expanduser(optuna_dir)
  envs = ["HalfCheetah-v5", "Walker2d-v5", "Hopper-v5", "Swimmer-v5", "Humanoid-v5", "HumanoidStandup-v5"]

  # Load HalfCheetah study to get base params
  hc_env_id = "HalfCheetah-v5"
  hc_study_name = f"trpor-{hc_env_id}-tuning"
  hc_file_path = os.path.join(optuna_dir, f"{hc_study_name}_log")
  hc_storage = JournalStorage(JournalFileBackend(hc_file_path))
  hc_study = optuna.load_study(study_name=hc_study_name, storage=hc_storage)
  base_params = hc_study.best_trial.params  # ent_coef, target_kl, learning_rate, sub_sampling_factor, n_critic_updates
  hc_best_value = hc_study.best_value

  # Fixed params common to all
  fixed_params = load_default_hyperparams(hc_env_id)

  print("TRPOR Hyperparameter Tuning Report")
  print("==================================")

  for env_id in envs:
    if env_id == hc_env_id:
      study_name = f"trpor-{env_id}-tuning"
      best_value = hc_best_value
      env_params = base_params.copy()
    else:
      study_name = f"trpor-{env_id}-entcoef-tuning"
      file_path = os.path.join(optuna_dir, f"{study_name}_log")
      storage = JournalStorage(JournalFileBackend(file_path))
      study = optuna.load_study(study_name=study_name, storage=storage)
      ent_coef = study.best_trial.params["ent_coef"]
      best_value = study.best_value
      env_params = base_params.copy()
      env_params["ent_coef"] = ent_coef

    env_params["net_arch"] = "medium" if "Humanoid" in env_id else "small"

    # Combine with fixed
    full_params = {**fixed_params, **env_params}

    print(f"\nEnvironment: {env_id}")
    print(f"Study Name: {study_name}")
    print(f"Best Mean Reward: {best_value}")
    print("Best Parameters:")
    for key, value in sorted(full_params.items()):
      print(f"  {key}: {value}")
    print("-" * 40)


if __name__ == "__main__":
  main()
  generate_trpor_report()
