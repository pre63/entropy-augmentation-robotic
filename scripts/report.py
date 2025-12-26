import argparse
import bisect
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

NUM_RUNS_PLOT = 1000
FIG_SIZE = (24, 12)  # Wider and taller to accommodate side legend


def load_file(args):
  compare_dir, filename = args
  key = filename[:-4]  # remove .pkl
  pkl_path = os.path.join(compare_dir, filename)
  with open(pkl_path, "rb") as pf:
    return key, pickle.load(pf)


def load_data(compare_dir):
  all_files = [f for f in os.listdir(compare_dir) if f.endswith(".pkl") and "_run" in f]

  # Group files by config
  config_files = {}
  for filename in all_files:
    key = filename[:-4]
    if "_run" in key:
      parts = key.rsplit("_run", 1)
      if len(parts) == 2:
        config, run_str = parts
        try:
          run_num = int(run_str)
        except ValueError:
          continue
        if config not in config_files:
          config_files[config] = []
        config_files[config].append((run_num, filename))

  # For each config, sort by run_num and select first NUM_RUNS_PLOT files
  selected_files = []
  for config in config_files:
    sorted_files = sorted(config_files[config])
    for run_num, filename in sorted_files[:NUM_RUNS_PLOT]:
      selected_files.append(filename)

  all_data = {}
  if selected_files:
    results = []
    # read file synch
    for filename in selected_files:
      result = load_file((compare_dir, filename))
      results.append(result)

    for key, data in results:
      all_data[key] = data

  config_dict = {}
  for key in all_data:
    if "_run" in key:
      parts = key.rsplit("_run", 1)
      if len(parts) == 2:
        config, run_str = parts
        try:
          run_num = int(run_str)
        except ValueError:
          continue
        if config not in config_dict:
          config_dict[config] = []
        config_dict[config].append((run_num, all_data[key]))

  return config_dict


def compute_iqm(values):
  if len(values) == 0:
    return 0.0
  sorted_values = np.sort(values)
  n = len(sorted_values)
  lower = int(np.ceil(0.25 * n))
  upper = int(np.floor(0.75 * n)) + 1
  if lower >= upper:
    return np.mean(sorted_values)
  iqm_values = sorted_values[lower:upper]
  return np.mean(iqm_values)


def generate_detailed_report(
  config_dict,
  config_names,
  episode_lists,
  per_run_total_steps,
  inference_means_lists,
  inference_stds_lists,
  clean_inference_means_lists,
  clean_inference_stds_lists,
  env_id,
  original_config_names,
):
  print(f"## Detailed Performance for {env_id}\n")
  print(
    "| Config | Run | AUC | Final Return | Max Return | Inference Mean | Inference Std | Stability | Clean Inference Mean | Clean Inference Std | Clean Stability |"
  )
  print(
    "|--------|-----|-----|--------------|------------|----------------|---------------|-----------|----------------------|---------------------|-----------------|"
  )
  for config_idx, config in enumerate(config_names):
    original_config = original_config_names[config_idx]
    if original_config not in config_dict:
      continue
    runs_data = sorted(config_dict[original_config], key=lambda x: x[0])
    for r, (run_num, run_data) in enumerate(runs_data):
      run_eps = episode_lists[config_idx][r]
      run_total_ts = per_run_total_steps[config_idx][r]
      auc = compute_episode_auc(run_eps, run_total_ts)
      run_rets = [ep["return"] for ep in run_eps]
      max_return = max(run_rets) if run_rets else 0.0
      threshold_ts = 0.8 * run_total_ts
      final_rets = [ep["return"] for ep in run_eps if ep["end_timestep"] > threshold_ts]
      final_return = np.mean(final_rets) if final_rets else 0.0
      inf_mean = inference_means_lists[config_idx][r]
      inf_std = inference_stds_lists[config_idx][r]
      stability = inf_mean / inf_std if inf_std > 0 else 0.0
      clean_inf_mean = clean_inference_means_lists[config_idx][r]
      clean_inf_std = clean_inference_stds_lists[config_idx][r]
      clean_stability = clean_inf_mean / clean_inf_std if clean_inf_std > 0 else 0.0
      print(
        f"| {config} | {run_num} | {auc:.2e} | {final_return:.2e} | {max_return:.2e} | {inf_mean:.2e} | {inf_std:.2e} | {stability:.2e} | {clean_inf_mean:.2e} | {clean_inf_std:.2e} | {clean_stability:.2e} |"
      )


def prepare_lists(config_dict):
  config_names = sorted(config_dict.keys())
  step_rewards_lists = []
  episode_lists = []
  run_numbers_lists = []
  episode_entropies_lists = []  # Kept for compatibility, but optional
  kl_lists = []
  surrogate_lists = []
  inference_means_lists = []
  inference_stds_lists = []
  clean_inference_means_lists = []
  clean_inference_stds_lists = []

  for config in config_names:
    runs_data = sorted(config_dict[config])
    step_rewards_list = [run_data["step_rewards"] for _, run_data in runs_data]
    episode_rewards = [run_data["episode_rewards"] for _, run_data in runs_data]
    episode_end_timesteps = [run_data["episode_end_timesteps"] for _, run_data in runs_data]
    run_numbers = [run_num for run_num, _ in runs_data]
    episode_list = []
    episode_entropies_list = []
    kl_list = []
    surrogate_list = []
    for i, (_, run_data) in enumerate(runs_data):
      entropies = run_data.get("rollout_metrics", {}).get("entropy_mean", [])  # Assuming key
      episode_entropies_list.append(entropies)
      kl_vals = run_data.get("rollout_metrics", {}).get("kl_div", [])  # New
      kl_list.append(kl_vals)
      surrogate_vals = run_data.get("rollout_metrics", {}).get("policy_objective", [])  # New
      surrogate_list.append(surrogate_vals)

      eps = [{"return": r, "end_timestep": t} for r, t in zip(episode_rewards[i], episode_end_timesteps[i])]
      episode_list.append(eps)

    inference_means_list = [run_data.get("inference_mean_reward", 0.0) for _, run_data in runs_data]
    inference_stds_list = [run_data.get("inference_std_reward", 0.0) for _, run_data in runs_data]
    clean_inference_means_list = [run_data.get("clean_inference_mean_reward", 0.0) for _, run_data in runs_data]
    clean_inference_stds_list = [run_data.get("clean_inference_std_reward", 0.0) for _, run_data in runs_data]

    step_rewards_lists.append(step_rewards_list)
    episode_lists.append(episode_list)
    run_numbers_lists.append(run_numbers)
    episode_entropies_lists.append(episode_entropies_list)
    kl_lists.append(kl_list)
    surrogate_lists.append(surrogate_list)
    inference_means_lists.append(inference_means_list)
    inference_stds_lists.append(inference_stds_list)
    clean_inference_means_lists.append(clean_inference_means_list)
    clean_inference_stds_lists.append(clean_inference_stds_list)

  return (
    config_names,
    step_rewards_lists,
    episode_lists,
    run_numbers_lists,
    episode_entropies_lists,
    kl_lists,
    surrogate_lists,
    inference_means_lists,
    inference_stds_lists,
    clean_inference_means_lists,
    clean_inference_stds_lists,
  )


def compute_timesteps_and_downsample(step_rewards_lists, disable_downsampling=False):
  total_timesteps = max(max(len(rews) for rews in step_list) if step_list else 1 for step_list in step_rewards_lists) if step_rewards_lists else 1
  if disable_downsampling:
    downsample_factor = 1
  else:
    downsample_factor = max(1, total_timesteps // 1000)
  timesteps_np = np.arange(downsample_factor // 2 + 1, total_timesteps + 1, downsample_factor)  # Approximate mid-window points
  return total_timesteps, downsample_factor, timesteps_np


def compute_episode_auc(run_eps, run_total_ts):
  if not run_eps:
    return 0.0
  auc = 0.0
  prev_ret = 0.0
  prev_ts = 0
  for ep in run_eps:
    end_ts = ep["end_timestep"]
    length = end_ts - prev_ts
    auc += prev_ret * length
    prev_ret = ep["return"]
    prev_ts = end_ts
  if prev_ts < run_total_ts:
    length = run_total_ts - prev_ts
    auc += prev_ret * length
  return auc


def compute_averaged_curves(config_names, episode_lists, total_timesteps, bin_size=1000):
  bins = np.arange(0, total_timesteps + 1, bin_size)
  averaged = {}
  num_bins = len(bins) - 1
  for v_idx, config in enumerate(config_names):
    num_runs = len(episode_lists[v_idx])
    if num_runs == 0:
      averaged[config] = (np.full(num_bins, np.nan), np.full(num_bins, np.nan))
      continue
    ys = np.full((num_runs, total_timesteps), np.nan)
    for run in range(num_runs):
      ep_infos = episode_lists[v_idx][run]
      prev_ts = 0
      last_ret = 0.0
      for ep in ep_infos:
        end_idx = min(ep["end_timestep"], total_timesteps)
        ys[run, prev_ts:end_idx] = last_ret
        last_ret = ep["return"]
        prev_ts = end_idx
      if prev_ts < total_timesteps:
        ys[run, prev_ts:] = last_ret
    mean_y = np.nanmean(ys, axis=0)
    std_y = np.nanstd(ys, axis=0)
    # Bin average
    binned_mean = np.full(num_bins, np.nan)
    binned_std = np.full(num_bins, np.nan)
    for k in range(num_bins):
      start = bins[k]
      end = bins[k + 1]
      slice_mean = mean_y[start:end]
      slice_std = std_y[start:end]
      if len(slice_mean) > 0 and not np.all(np.isnan(slice_mean)):
        binned_mean[k] = np.nanmean(slice_mean)
        binned_std[k] = np.nanmean(slice_std)
    averaged[config] = (binned_mean, binned_std)
  return averaged, bins[:-1]  # x for plot


def compute_averaged_metric_over_timesteps(config_names, metric_lists, total_timesteps, per_run_total_steps, bin_size=1000):
  bins = np.arange(0, total_timesteps + 1, bin_size)
  averaged = {}
  num_bins = len(bins) - 1
  for v_idx, config in enumerate(config_names):
    num_runs = len(metric_lists[v_idx])
    if num_runs == 0:
      averaged[config] = (np.full(num_bins, np.nan), np.full(num_bins, np.nan))
      continue
    ys = np.full((num_runs, total_timesteps), np.nan)
    for r in range(num_runs):
      metrics = metric_lists[v_idx][r]
      if not metrics:
        continue
      num_updates = len(metrics)
      run_total_ts = per_run_total_steps[v_idx][r]
      step_size = run_total_ts / num_updates if num_updates > 0 else 0
      prev_ts = 0
      for k in range(num_updates):
        end_idx = min(int((k + 1) * step_size), total_timesteps)
        ys[r, prev_ts:end_idx] = metrics[k]
        prev_ts = end_idx
      if prev_ts < total_timesteps and metrics:
        ys[r, prev_ts:] = metrics[-1]
    mean_y = np.nanmean(ys, axis=0)
    std_y = np.nanstd(ys, axis=0)
    binned_mean = np.full(num_bins, np.nan)
    binned_std = np.full(num_bins, np.nan)
    for k in range(num_bins):
      start = bins[k]
      end = bins[k + 1]
      slice_mean = mean_y[start:end]
      slice_std = std_y[start:end]
      if len(slice_mean) > 0 and not np.all(np.isnan(slice_mean)):
        binned_mean[k] = np.nanmean(slice_mean)
        binned_std[k] = np.nanmean(slice_std)
    averaged[config] = (binned_mean, binned_std)
  return averaged


def smooth_data(data, window_size=5):
  if window_size < 2:
    return data
  return pd.Series(data).rolling(window_size, min_periods=1, center=True).mean().to_numpy()


def plot_learning_curve(config_names, averaged_curves, bins, color_map, compare_dir, filename, title, disable_smoothing=False, ylabel="Average Return"):
  plt.figure(figsize=FIG_SIZE)
  ax = plt.gca()
  ax.set_facecolor("gainsboro")
  ax.grid(True, color="darkgrey", linestyle=":")
  for spine in ax.spines.values():
    spine.set_edgecolor("darkgrey")
  all_mins = []
  all_maxs = []
  linewidth = 2.5
  num_configs = len(config_names)
  has_data = False
  for i, config in enumerate(config_names):
    if config not in averaged_curves:
      continue
    mean_y, std_y = averaged_curves[config]
    if np.all(np.isnan(mean_y)):
      continue
    has_data = True
    if disable_smoothing:
      smoothed_mean = mean_y
    else:
      smoothed_mean = smooth_data(mean_y)
    color = color_map[config]
    plt.plot(bins, smoothed_mean, label=config, color=color, linewidth=linewidth)
    sparse_step = max(1, len(bins) // 20)
    x_sparse = bins[::sparse_step]
    mean_sparse = mean_y[::sparse_step]
    std_sparse_list = []
    for k in range(len(x_sparse)):
      start = k * sparse_step
      end = min((k + 1) * sparse_step, len(std_y))
      avg_std = np.nanmean(std_y[start:end]) if end > start else std_y[start]
      std_sparse_list.append(avg_std)
    std_sparse = np.array(std_sparse_list)
    plt.errorbar(
      x_sparse,
      mean_sparse,
      yerr=std_sparse,
      fmt="none",
      ecolor=color,
      elinewidth=linewidth,
      capsize=5,
      uplims=False,
      lolims=False,
      errorevery=(i, num_configs),
    )
    valid_mask = ~np.isnan(smoothed_mean) & ~np.isnan(std_y)
    if np.any(valid_mask):
      valid_smoothed = smoothed_mean[valid_mask]
      valid_std = std_y[valid_mask]
      all_mins.append(np.min(valid_smoothed - valid_std))
      all_maxs.append(np.max(valid_smoothed + valid_std))
  if has_data and all_mins and all_maxs:
    global_min = np.min(all_mins)
    global_max = np.max(all_maxs)
    padding = (global_max - global_min) * 0.05 if global_max > global_min else 1.0
    ax.set_ylim(global_min - padding, global_max + padding)
  elif not has_data:
    plt.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=20)
  handles, labels = plt.gca().get_legend_handles_labels()
  sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i].lower())
  handles = [handles[i] for i in sorted_indices]
  labels = [labels[i] for i in sorted_indices]
  plt.xlabel("timesteps")
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend(handles, labels, loc="upper left", facecolor="gainsboro")
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_raw_learning_curves(config_names, episode_lists, run_numbers_lists, x_bins, total_timesteps, color_map, compare_dir, filename, title):
  plt.figure(figsize=FIG_SIZE)
  ax = plt.gca()
  ax.set_facecolor("gainsboro")
  ax.grid(True, color="darkgrey", linestyle=":")
  for spine in ax.spines.values():
    spine.set_edgecolor("darkgrey")

  has_data = False
  legend_elements = []

  total_runs = sum(len(episode_lists[i]) for i in range(len(config_names)))

  # Adaptive transparency and linewidth
  if total_runs <= 10:
    alpha = 1.0
    linewidth_raw = 1.8
  elif total_runs <= 20:
    alpha = 0.9
    linewidth_raw = 1.4
  elif total_runs <= 50:
    alpha = 0.8
    linewidth_raw = 1.0
  else:
    alpha = 0.7
    linewidth_raw = 0.8

  for cfg_idx, config in enumerate(config_names):
    color = color_map[config]
    run_nums = run_numbers_lists[cfg_idx]
    for run_idx, run_eps in enumerate(episode_lists[cfg_idx]):
      if not run_eps:
        continue
      run_num = run_nums[run_idx]

      # Ensure episodes sorted
      run_eps = sorted(run_eps, key=lambda e: e["end_timestep"])

      interval_starts = [0] + [ep["end_timestep"] for ep in run_eps]
      interval_returns = [0.0] + [ep["return"] for ep in run_eps]
      interval_starts.append(total_timesteps)
      interval_returns.append(interval_returns[-1])

      y_binned = []
      for bin_x in x_bins:
        k = bisect.bisect_right(interval_starts, bin_x) - 1
        y_binned.append(interval_returns[max(k, 0)])

      plt.plot(x_bins, y_binned, color=color, alpha=alpha, linewidth=linewidth_raw)
      has_data = True

      # Proxy for legend: thick solid line, opaque
      legend_elements.append(Line2D([0], [0], color=color, lw=3, alpha=1.0, label=f"{config} Run {run_num}"))

  if has_data:
    # Multi-column if many entries
    ncol = 1 if total_runs <= 20 else 2 if total_runs <= 50 else 3
    ax.legend(
      handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=ncol, fontsize=12, title=f"Runs ({total_runs} total)", title_fontsize=14
    )
  else:
    plt.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=20)

  plt.subplots_adjust(right=0.72)  # Make room for legend on the right
  plt.xlabel("timesteps")
  plt.ylabel("Episode Return")
  plt.title(title)
  plt.tight_layout()
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_combined_metrics(config, averaged_episode, averaged_kl, averaged_surrogate, bins, color_map, compare_dir, filename, title, disable_smoothing=False):
  fig, axs = plt.subplots(3, 1, figsize=(20, 20), sharex=True)
  color = color_map[config]
  linewidth = 2.5

  for idx, (data, y_label, sub_title) in enumerate(
    [
      (averaged_episode, "Average Return", "Episode Returns"),
      (averaged_kl, "KL Divergence", "KL Divergence"),
      (averaged_surrogate, "Surrogate Objective", "Surrogate Objective"),
    ]
  ):
    ax = axs[idx]
    ax.set_facecolor("gainsboro")
    ax.grid(True, color="darkgrey", linestyle=":")
    for spine in ax.spines.values():
      spine.set_edgecolor("darkgrey")
    mean_y, std_y = data
    if np.all(np.isnan(mean_y)):
      ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=20)
      continue
    if disable_smoothing:
      smoothed_mean = mean_y
    else:
      smoothed_mean = smooth_data(mean_y)
    ax.plot(bins, smoothed_mean, color=color, linewidth=linewidth)
    sparse_step = max(1, len(bins) // 20)
    x_sparse = bins[::sparse_step]
    mean_sparse = mean_y[::sparse_step]
    std_sparse_list = []
    for k in range(len(x_sparse)):
      start = k * sparse_step
      end = min((k + 1) * sparse_step, len(std_y))
      avg_std = np.nanmean(std_y[start:end]) if end > start else std_y[start]
      std_sparse_list.append(avg_std)
    std_sparse = np.array(std_sparse_list)
    ax.errorbar(
      x_sparse,
      mean_sparse,
      yerr=std_sparse,
      fmt="none",
      ecolor=color,
      elinewidth=linewidth,
      capsize=5,
      uplims=False,
      lolims=False,
    )
    ax.set_ylabel(y_label)
    ax.set_title(sub_title)
    if idx == 2:
      ax.set_xlabel("timesteps")

  fig.suptitle(title)
  plt.tight_layout()
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_scatter_variations(perfs, compare_dir, filename, title):
  if len(perfs) < 2:
    return
  plt.figure(figsize=FIG_SIZE)
  ax = plt.gca()
  ax.set_facecolor("gainsboro")
  ax.grid(True, color="darkgrey", linestyle=":")
  for spine in ax.spines.values():
    spine.set_edgecolor("darkgrey")
  for p in perfs:
    plt.scatter(p["avg_auc"], p["avg_final_return"], label=p["config"], s=50)
  plt.xlabel("area under the curve")
  plt.ylabel("average final return")
  plt.title(title)
  plt.legend(loc="upper left", facecolor="gainsboro")
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_noise_vs_performance(perfs, compare_dir, filename, title):
  noises = []
  finals = []
  for p in perfs:
    config_low = p["config"].lower()
    if "(" in config_low:
      try:
        noise_str = p["config"].split("(")[1].split(")")[0]
        noise = float(noise_str)
      except:
        continue
    else:
      noise = 0.0
    noises.append(noise)
    finals.append(p["avg_final_return"])
  if noises:
    sorted_indices = np.argsort(noises)
    noises = np.array(noises)[sorted_indices]
    finals = np.array(finals)[sorted_indices]
    plt.figure(figsize=FIG_SIZE)
    plt.bar(noises, finals, width=0.05)
    plt.xlabel("Noise Level")
    plt.ylabel("Average Final Return")
    plt.title(title)
    plt.savefig(os.path.join(compare_dir, filename))
    plt.close()


def plot_inference_bar(config_names, inference_means_lists, inference_stds_lists, color_map, compare_dir, filename, title):
  plt.figure(figsize=FIG_SIZE)
  ax = plt.gca()
  ax.set_facecolor("gainsboro")
  ax.grid(True, color="darkgrey", linestyle=":")
  for spine in ax.spines.values():
    spine.set_edgecolor("darkgrey")
  inference_means = [np.mean(l) if l else 0.0 for l in inference_means_lists]
  inference_stds = [np.std(l) if l else 0.0 for l in inference_stds_lists]
  colors_list = [color_map[v] for v in config_names]
  x_pos = np.arange(len(config_names))
  plt.bar(x_pos, inference_means, yerr=inference_stds, color=colors_list, capsize=5)
  plt.xticks(x_pos, config_names, rotation=45, ha="right")
  plt.xlabel("Configs")
  plt.ylabel("Inference Mean Reward (± std)")
  plt.title(title)
  plt.tight_layout()
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_inference_scatter(perfs, compare_dir, filename, title):
  if len(perfs) < 1:
    return
  plt.figure(figsize=FIG_SIZE)
  ax = plt.gca()
  ax.set_facecolor("gainsboro")
  ax.grid(True, color="darkgrey", linestyle=":")
  for spine in ax.spines.values():
    spine.set_edgecolor("darkgrey")
  for p in perfs:
    plt.scatter(p["avg_inference_mean"], p["avg_inference_std"], label=p["config"], s=50)
  plt.xlabel("Average Inference Mean")
  plt.ylabel("Average Inference Std")
  plt.title(title)
  plt.legend(loc="upper left", facecolor="gainsboro")
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_clean_inference_bar(config_names, clean_inference_means_lists, clean_inference_stds_lists, color_map, compare_dir, filename, title):
  plt.figure(figsize=FIG_SIZE)
  ax = plt.gca()
  ax.set_facecolor("gainsboro")
  ax.grid(True, color="darkgrey", linestyle=":")
  for spine in ax.spines.values():
    spine.set_edgecolor("darkgrey")
  clean_inference_means = [np.mean(l) if l else 0.0 for l in clean_inference_means_lists]
  clean_inference_stds = [np.std(l) if l else 0.0 for l in clean_inference_stds_lists]
  colors_list = [color_map[v] for v in config_names]
  x_pos = np.arange(len(config_names))
  plt.bar(x_pos, clean_inference_means, yerr=clean_inference_stds, color=colors_list, capsize=5)
  plt.xticks(x_pos, config_names, rotation=45, ha="right")
  plt.xlabel("Configs")
  plt.ylabel("Clean Inference Mean Reward (± std)")
  plt.title(title)
  plt.tight_layout()
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_clean_inference_scatter(perfs, compare_dir, filename, title):
  if len(perfs) < 1:
    return
  plt.figure(figsize=FIG_SIZE)
  ax = plt.gca()
  ax.set_facecolor("gainsboro")
  ax.grid(True, color="darkgrey", linestyle=":")
  for spine in ax.spines.values():
    spine.set_edgecolor("darkgrey")
  for p in perfs:
    plt.scatter(p["avg_clean_inference_mean"], p["avg_clean_inference_std"], label=p["config"], s=50)
  plt.xlabel("Average Clean Inference Mean")
  plt.ylabel("Average Clean Inference Std")
  plt.title(title)
  plt.legend(loc="upper left", facecolor="gainsboro")
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def compute_config_metrics(
  config, episode_list, per_run_total_steps, inference_means_list, inference_stds_list, clean_inference_means_list, clean_inference_stds_list
):
  num_runs = len(episode_list)
  if num_runs == 0:
    return {
      "config": config,
      "avg_final_return": 0.0,
      "std_final_return": 0.0,
      "avg_max_return": 0.0,
      "avg_auc": 0.0,
      "std_auc": 0.0,
      "avg_inference_mean": 0.0,
      "avg_inference_std": 0.0,
      "inference_stability": 0.0,
      "avg_clean_inference_mean": 0.0,
      "avg_clean_inference_std": 0.0,
      "clean_inference_stability": 0.0,
      "iqm_auc": 0.0,
      "iqm_final_return": 0.0,
      "iqm_max_return": 0.0,
      "iqm_inference_mean": 0.0,
      "iqm_stability": 0.0,
      "iqm_clean_inference_mean": 0.0,
      "iqm_clean_stability": 0.0,
    }
  per_run_all_ep_returns = [[ep["return"] for ep in run_eps] for run_eps in episode_list]
  per_run_max = [max(run_rets, default=0.0) for run_rets in per_run_all_ep_returns]
  avg_max_return = np.mean(per_run_max)
  per_run_final = []
  for r in range(num_runs):
    run_eps = episode_list[r]
    run_total_ts = per_run_total_steps[r]
    threshold_ts = 0.8 * run_total_ts
    final_rets = [ep["return"] for ep in run_eps if ep["end_timestep"] > threshold_ts]
    per_run_final.append(np.mean(final_rets) if final_rets else 0.0)
  avg_final_return = np.mean(per_run_final)
  std_final_return = np.std(per_run_final)
  per_run_auc = [compute_episode_auc(episode_list[r], per_run_total_steps[r]) for r in range(num_runs)]
  avg_auc = np.mean(per_run_auc)
  std_auc = np.std(per_run_auc)
  avg_inference_mean = np.mean(inference_means_list)
  avg_inference_std = np.mean(inference_stds_list)
  inference_stability = avg_inference_mean / avg_inference_std if avg_inference_std > 0 else 0.0
  avg_clean_inference_mean = np.mean(clean_inference_means_list)
  avg_clean_inference_std = np.mean(clean_inference_stds_list)
  clean_inference_stability = avg_clean_inference_mean / avg_clean_inference_std if avg_clean_inference_std > 0 else 0.0

  iqm_auc = compute_iqm(per_run_auc)
  iqm_final_return = compute_iqm(per_run_final)
  iqm_max_return = compute_iqm(per_run_max)
  iqm_inference_mean = compute_iqm(inference_means_list)
  per_run_stability = [inference_means_list[r] / inference_stds_list[r] if inference_stds_list[r] > 0 else 0.0 for r in range(num_runs)]
  iqm_stability = compute_iqm(per_run_stability)
  iqm_clean_inference_mean = compute_iqm(clean_inference_means_list)
  per_run_clean_stability = [clean_inference_means_list[r] / clean_inference_stds_list[r] if clean_inference_stds_list[r] > 0 else 0.0 for r in range(num_runs)]
  iqm_clean_stability = compute_iqm(per_run_clean_stability)

  return {
    "config": config,
    "avg_final_return": avg_final_return,
    "std_final_return": std_final_return,
    "avg_max_return": avg_max_return,
    "avg_auc": avg_auc,
    "std_auc": std_auc,
    "avg_inference_mean": avg_inference_mean,
    "avg_inference_std": avg_inference_std,
    "inference_stability": inference_stability,
    "avg_clean_inference_mean": avg_clean_inference_mean,
    "avg_clean_inference_std": avg_clean_inference_std,
    "clean_inference_stability": clean_inference_stability,
    "iqm_auc": iqm_auc,
    "iqm_final_return": iqm_final_return,
    "iqm_max_return": iqm_max_return,
    "iqm_inference_mean": iqm_inference_mean,
    "iqm_stability": iqm_stability,
    "iqm_clean_inference_mean": iqm_clean_inference_mean,
    "iqm_clean_stability": iqm_clean_stability,
  }


def generate_report(model_performances, env_id):
  print(f"# Performance Comparison for {env_id}\n")
  print(
    "| Config | AUC (Mean ± Std) | Final Return (Mean ± Std) | Max Return (Avg) | Inference Mean (± Std) | Stability | Clean Inference Mean (± Std) | Clean Stability | IQM AUC | IQM Final Return | IQM Max Return | IQM Inference Mean | IQM Stability | IQM Clean Inference Mean | IQM Clean Stability |"
  )
  print(
    "|--------|------------------|---------------------------|------------------|------------------------|-----------|------------------------------|-----------------|---------|------------------|----------------|--------------------|---------------|--------------------------|---------------------|"
  )

  for perf in sorted(model_performances, key=lambda x: x["avg_auc"], reverse=True):
    auc_str = f"{perf['avg_auc']:.2e} ± {perf['std_auc']:.2e}"
    final_str = f"{perf['avg_final_return']:.2e} ± {perf['std_final_return']:.2e}"
    max_str = f"{perf['avg_max_return']:.2e}"
    inf_str = f"{perf['avg_inference_mean']:.2e} ± {perf['avg_inference_std']:.2e}"
    stab_str = f"{perf['inference_stability']:.2e}"
    clean_inf_str = f"{perf['avg_clean_inference_mean']:.2e} ± {perf['avg_clean_inference_std']:.2e}"
    clean_stab_str = f"{perf['clean_inference_stability']:.2e}"
    iqm_auc_str = f"{perf['iqm_auc']:.2e}"
    iqm_final_str = f"{perf['iqm_final_return']:.2e}"
    iqm_max_str = f"{perf['iqm_max_return']:.2e}"
    iqm_inf_str = f"{perf['iqm_inference_mean']:.2e}"
    iqm_stab_str = f"{perf['iqm_stability']:.2e}"
    iqm_clean_inf_str = f"{perf['iqm_clean_inference_mean']:.2e}"
    iqm_clean_stab_str = f"{perf['iqm_clean_stability']:.2e}"

    print(
      f"| {perf['config']} | {auc_str} | {final_str} | {max_str} | {inf_str} | {stab_str} | {clean_inf_str} | {clean_stab_str} | {iqm_auc_str} | {iqm_final_str} | {iqm_max_str} | {iqm_inf_str} | {iqm_stab_str} | {iqm_clean_inf_str} | {iqm_clean_stab_str} |"
    )


def report(compare_dir, env_id, disable_downsampling=False, disable_smoothing=False):
  print(f"Generating report for experiments in {compare_dir} on environment {env_id}...\n")
  all_files = [f for f in os.listdir(compare_dir) if f.endswith(".pkl") and "_run" in f]
  config_dict = load_data(compare_dir)
  print(f"Found configs: {len(config_dict)}\n")

  (
    config_names,
    step_rewards_lists,
    episode_lists,
    run_numbers_lists,
    episode_entropies_lists,
    kl_lists,
    surrogate_lists,
    inference_means_lists,
    inference_stds_lists,
    clean_inference_means_lists,
    clean_inference_stds_lists,
  ) = prepare_lists(config_dict)

  # Save original config names
  original_config_names = config_names[:]

  # Modify config names
  modified_config_names = []
  for orig in config_names:
    if "noise" in orig.lower():
      match = re.search(r"noise([\d\.]+)", orig, re.IGNORECASE)
      if match:
        noise = match.group(1)
        base = orig.split("_noise")[0].upper()
        new_name = f"{base} ({noise})"
      else:
        new_name = orig.upper()
    else:
      new_name = orig.upper()
    modified_config_names.append(new_name)
  config_names = modified_config_names

  colors = plt.get_cmap("tab20b")(np.linspace(0, 1, len(config_names)))
  color_map = dict(zip(config_names, [tuple(c) for c in colors]))

  total_timesteps, downsample_factor, timesteps_np = compute_timesteps_and_downsample(step_rewards_lists, disable_downsampling)

  # Collect per_run_total_steps
  per_run_total_steps = [[len(run_data["step_rewards"]) for _, run_data in sorted(config_dict[config])] for config in original_config_names]

  averaged_episodes, bins = compute_averaged_curves(config_names, episode_lists, total_timesteps)

  averaged_kl = compute_averaged_metric_over_timesteps(config_names, kl_lists, total_timesteps, per_run_total_steps)

  averaged_surrogate = compute_averaged_metric_over_timesteps(config_names, surrogate_lists, total_timesteps, per_run_total_steps)

  averaged_entropies = compute_averaged_metric_over_timesteps(config_names, episode_entropies_lists, total_timesteps, per_run_total_steps)

  model_performances = [
    compute_config_metrics(
      config_names[j],
      episode_lists[j],
      per_run_total_steps[j],
      inference_means_lists[j],
      inference_stds_lists[j],
      clean_inference_means_lists[j],
      clean_inference_stds_lists[j],
    )
    for j in range(len(config_names))
  ]

  # Plot scatter for all configs
  plot_scatter_variations(model_performances, compare_dir, "scatter_all_configs.png", f"All Configs on {env_id}")

  # Find best TRPOR
  trpor_perfs = [p for p in model_performances if "trpor" in p["config"].lower()]
  if trpor_perfs:
    best_trpor = max(trpor_perfs, key=lambda x: x["avg_auc"])
    best_config = best_trpor["config"]
    best_color = color_map[best_config]

    # Plot episode for best TRPOR
    plot_learning_curve(
      [best_config],
      {best_config: averaged_episodes[best_config]},
      bins,
      color_map,
      compare_dir,
      "best_trpor_episode.png",
      f"Best TRPOR Episode Returns on {env_id}",
    )

    # Plot KL for best TRPOR
    plot_learning_curve(
      [best_config],
      {best_config: averaged_kl[best_config]},
      bins,
      color_map,
      compare_dir,
      "best_trpor_kl.png",
      f"Best TRPOR KL Divergence on {env_id}",
      ylabel="KL Divergence",
    )

    # Plot surrogate for best TRPOR
    plot_learning_curve(
      [best_config],
      {best_config: averaged_surrogate[best_config]},
      bins,
      color_map,
      compare_dir,
      "best_trpor_surrogate.png",
      f"Best TRPOR Surrogate Objective on {env_id}",
      ylabel="Surrogate Objective",
    )

    # Plot entropy for best TRPOR
    plot_learning_curve(
      [best_config],
      {best_config: averaged_entropies[best_config]},
      bins,
      color_map,
      compare_dir,
      "best_trpor_entropy.png",
      f"Best TRPOR Entropy on {env_id}",
      ylabel="Entropy",
    )

    # Combined plot for best TRPOR
    plot_combined_metrics(
      best_config,
      averaged_episodes[best_config],
      averaged_kl[best_config],
      averaged_surrogate[best_config],
      bins,
      color_map,
      compare_dir,
      "best_trpor_combined.png",
      f"Combined Metrics for Best TRPOR on {env_id}",
      disable_smoothing,
    )

    # Scatter for TRPOR
    plot_scatter_variations(trpor_perfs, compare_dir, "scatter_trpor_configs.png", f"TRPOR on {env_id}")

  # Scatter for TRPO
  trpo_perfs = [p for p in model_performances if "trpo" in p["config"].lower() and "trpor" not in p["config"].lower()]
  if trpo_perfs:
    plot_scatter_variations(trpo_perfs, compare_dir, "scatter_trpo_configs.png", f"TRPO on {env_id}")

  # Main learning curve for all (means + error bars)
  plot_learning_curve(
    config_names, averaged_episodes, bins, color_map, compare_dir, "learning_curve.png", f"Learning Curves on {env_id}", disable_smoothing=disable_smoothing
  )

  # Raw individual runs - ALL runs on ONE plot, each numbered in the side legend
  total_runs = sum(len(episode_lists[i]) for i in range(len(config_names)))
  plot_raw_learning_curves(
    config_names,
    episode_lists,
    run_numbers_lists,
    bins,
    total_timesteps,
    color_map,
    compare_dir,
    "raw_learning_curves.png",
    f"Raw Individual Runs ({total_runs} total) on {env_id}",
  )

  # Inference plots
  plot_inference_bar(config_names, inference_means_lists, inference_stds_lists, color_map, compare_dir, "inference_bar.png", f"Inference Rewards on {env_id}")

  # Clean inference plots
  plot_clean_inference_bar(
    config_names,
    clean_inference_means_lists,
    clean_inference_stds_lists,
    color_map,
    compare_dir,
    "clean_inference_bar.png",
    f"Clean Inference Rewards on {env_id}",
  )

  # For specific envs, add noise curves and scatters
  if env_id in ["HalfCheetah", "Humanoid"]:
    trpo_perfs = [p for p in model_performances if "trpo" in p["config"].lower() and "trpor" not in p["config"].lower()]
    trpor_perfs = [p for p in model_performances if "trpor" in p["config"].lower()]

    if trpo_perfs:
      trpo_noise_names = [p["config"] for p in trpo_perfs]
      plot_learning_curve(
        trpo_noise_names,
        {v: averaged_episodes[v] for v in trpo_noise_names if v in averaged_episodes},
        bins,
        color_map,
        compare_dir,
        "trpo_noise_curves.png",
        f"TRPO Noise Configurations on {env_id}",
      )
      plot_noise_vs_performance(trpo_perfs, compare_dir, "trpo_noise_vs_final.png", f"TRPO Noise vs Final Return on {env_id}")
      plot_inference_scatter(trpo_perfs, compare_dir, "trpo_inference_scatter.png", f"TRPO Inference Stability on {env_id}")
      plot_clean_inference_scatter(trpo_perfs, compare_dir, "trpo_clean_inference_scatter.png", f"TRPO Clean Inference Stability on {env_id}")

    if trpor_perfs:
      trpor_noise_names = [p["config"] for p in trpor_perfs]
      plot_learning_curve(
        trpor_noise_names,
        {v: averaged_episodes[v] for v in trpor_noise_names if v in averaged_episodes},
        bins,
        color_map,
        compare_dir,
        "trpor_noise_curves.png",
        f"TRPOR Noise Configurations on {env_id}",
      )
      plot_noise_vs_performance(trpor_perfs, compare_dir, "trpor_noise_vs_final.png", f"TRPOR Noise vs Final Return on {env_id}")
      plot_inference_scatter(trpor_perfs, compare_dir, "trpor_inference_scatter.png", f"TRPOR Inference Stability on {env_id}")
      plot_clean_inference_scatter(trpor_perfs, compare_dir, "trpor_clean_inference_scatter.png", f"TRPOR Clean Inference Stability on {env_id}")

  generate_report(model_performances, env_id)
  generate_detailed_report(
    config_dict,
    config_names,
    episode_lists,
    per_run_total_steps,
    inference_means_lists,
    inference_stds_lists,
    clean_inference_means_lists,
    clean_inference_stds_lists,
    env_id,
    original_config_names,
  )


if __name__ == "__main__":
  import matplotlib as mpl

  mpl.rcParams.update(
    {
      "font.family": "monospace",
      "font.size": 24,
      "axes.labelsize": 19,
      "axes.titlesize": 24,
      "legend.fontsize": 19,
      "xtick.labelsize": 19,
      "ytick.labelsize": 19,
      "axes.grid": True,
      "grid.color": "gainsboro",
      "grid.alpha": 0.5,
    }
  )

  disable_downsampling = False
  disable_smoothing = False

  base_compare_dirs = ["bp_comparison"]
  for base_compare_dir in base_compare_dirs:
    subdirs = sorted([d for d in os.listdir(base_compare_dir) if os.path.isdir(os.path.join(base_compare_dir, d))])
    print("# Table of Contents\n")
    for subdir in subdirs:
      env_id = subdir.split("_")[0].lower()
      print(f"- [{env_id}](#performance-comparison-for-{env_id})\n")
    print("\n")
    for subdir in subdirs:
      env_id = subdir.split("_")[0]
      compare_dir = os.path.join(base_compare_dir, subdir)
      report(compare_dir, env_id, disable_downsampling, disable_smoothing)
