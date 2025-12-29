import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from sb3.noise import MonitoredEntropyInjectionWrapper
from sb3.trpo import TRPO
from sb3.trpor import TRPOR


def make_video_env(env_id):
  env = gym.make(env_id, render_mode="rgb_array")
  return env


def record_model_video(model, config):
  """
    Records videos of the model running in the environment for multiple episodes and keeps the best one based on total reward.

    Args:
        model: The trained model instance (e.g., from stable_baselines3).
        config: A dictionary containing:
            - 'env_id': str, the Gym environment ID.
            - 'output_dir': str, directory to save the video.
            - 'name_prefix': str, prefix for the video file name.
            - 'name_suffix': optional str, suffix for the video file name (default: "").
            - 'num_episodes': optional int, number of episodes to run and select the best from (default: 10).
    """
  env_id = config["env_id"]
  output_dir = config["output_dir"]
  name_prefix = config["name_prefix"]
  name_suffix = config.get("name_suffix", "")
  num_episodes = config.get("num_episodes", 10)

  # if video exist skip
  final_video_path = os.path.join(output_dir, f"{name_prefix}{name_suffix}.mp4")
  if os.path.exists(final_video_path):
    print(f"Video {final_video_path} already exists, skipping recording.")
    return

  video_env = make_video_env(env_id)
  video_env = RecordVideo(video_env, output_dir, name_prefix=name_prefix, episode_trigger=lambda x: True)

  episode_rewards = []
  for ep in range(num_episodes):
    obs, _ = video_env.reset()
    ep_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, _ = video_env.step(action)
      ep_reward += reward
    episode_rewards.append(ep_reward)

  video_env.close()

  # Find the best episode
  best_ep = max(enumerate(episode_rewards), key=lambda x: x[1])[0]

  # Assume .mp4 extension
  best_video_path = os.path.join(output_dir, f"{name_prefix}-episode-{best_ep}.mp4")
  final_video_path = os.path.join(output_dir, f"{name_prefix}{name_suffix}.mp4")

  # Rename the best video
  if os.path.exists(best_video_path):
    os.rename(best_video_path, final_video_path)

  # Delete the other videos
  for ep in range(num_episodes):
    if ep != best_ep:
      video_path = os.path.join(output_dir, f"{name_prefix}-episode-{ep}.mp4")
      if os.path.exists(video_path):
        os.remove(video_path)


def record_all_models(base_compare_dir):
  for env_id in os.listdir(base_compare_dir):
    print(f"Processing directory: {env_id}")
    subdir_path = os.path.join(base_compare_dir, env_id)
    if os.path.isdir(subdir_path):
      compare_dir = subdir_path
      for file in os.listdir(compare_dir):
        if file.endswith(".zip"):
          key = file[:-4]
          try:
            variant_name, run_str = key.rsplit("_run", 1)
            run = int(run_str)
          except ValueError:
            print(f"Filename {file} does not match expected pattern, skipping.")
            continue
          if variant_name.startswith("TRPOR"):
            Variant = TRPOR
          elif variant_name.startswith("TRPO"):
            Variant = TRPO
          else:
            print(f"Unknown model type in filename {file}, skipping.")
            continue
          model_path = os.path.join(compare_dir, file)
          try:
            model = Variant.load(model_path)
          except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue
          config = {"env_id": env_id, "output_dir": compare_dir, "name_prefix": key, "name_suffix": ""}
          record_model_video(model, config)


if __name__ == "__main__":
  # take param --folder fist and oly argument
  folder_arg = "bp_comparison"
  if len(sys.argv) > 1:
    folder_arg = sys.argv[1]
  print(f"Using folder argument: {folder_arg}")

  record_all_models(folder_arg)
