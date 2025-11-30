import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np


def load_file(args):
  compare_dir, filename = args
  key = filename[:-4]  # remove .pkl
  pkl_path = os.path.join(compare_dir, filename)
  with open(pkl_path, "rb") as pf:
    return key, pickle.load(pf)


def load_data(compare_dir):
  all_data = {}
  files = [f for f in os.listdir(compare_dir) if f.endswith(".pkl") and "_run" in f]
  if files:
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
      results = executor.map(load_file, [(compare_dir, f) for f in files])
      for key, data in results:
        all_data[key] = data

  variant_dict = {}
  for key in all_data:
    if "_run" in key:
      parts = key.rsplit("_run", 1)
      if len(parts) == 2:
        variant, run_str = parts
        try:
          run_num = int(run_str)
        except ValueError:
          continue
        if variant not in variant_dict:
          variant_dict[variant] = []
        variant_dict[variant].append((run_num, all_data[key]))

  return variant_dict


def prepare_lists(variant_dict):
  variant_names = sorted(variant_dict.keys())
  step_rewards_lists = []
  episode_lists = []
  episode_entropies_lists = []  # Added for entropy
  inference_means_lists = []
  inference_stds_lists = []

  for variant in variant_names:
    runs_data = sorted(variant_dict[variant])
    step_rewards_list = [run_data["step_rewards"] for _, run_data in runs_data]
    episode_rewards = [run_data["episode_rewards"] for _, run_data in runs_data]
    episode_end_timesteps = [run_data["episode_end_timesteps"] for _, run_data in runs_data]
    episode_list = []
    episode_entropies_list = []
    for i, (_, run_data) in enumerate(runs_data):
      entropies = run_data.get("rollout_metrics", {}).get("entropy_mean", [])
      episode_entropies_list.append(entropies)

      eps = [{"reward": r, "end_timestep": t} for r, t in zip(episode_rewards[i], episode_end_timesteps[i])]
      episode_list.append(eps)

    inference_means_list = [run_data.get("inference_mean_reward", 0.0) for _, run_data in runs_data]
    inference_stds_list = [run_data.get("inference_std_reward", 0.0) for _, run_data in runs_data]

    step_rewards_lists.append(step_rewards_list)
    episode_lists.append(episode_list)
    episode_entropies_lists.append(episode_entropies_list)
    inference_means_lists.append(inference_means_list)
    inference_stds_lists.append(inference_stds_list)

  return variant_names, step_rewards_lists, episode_lists, episode_entropies_lists, inference_means_lists, inference_stds_lists


def compute_timesteps_and_downsample(step_rewards_lists):
  total_timesteps = max(max(len(rews) for rews in step_list) if step_list else 1 for step_list in step_rewards_lists) if step_rewards_lists else 1
  downsample_factor = max(1, total_timesteps // 1000)
  timesteps_np = np.arange(1, total_timesteps + 1)[::downsample_factor]
  return total_timesteps, downsample_factor, timesteps_np


def compute_episode_auc(run_eps, run_total_ts):
  if not run_eps:
    return 0.0
  auc = 0.0
  prev_rew = 0.0
  prev_ts = 0
  for ep in run_eps:
    end_ts = ep["end_timestep"]
    length = end_ts - prev_ts
    auc += prev_rew * length
    prev_rew = ep["reward"]
    prev_ts = end_ts
  if prev_ts < run_total_ts:
    length = run_total_ts - prev_ts
    auc += prev_rew * length
  return auc


def compute_variant_metrics(variant, step_rewards_list, episode_list, episode_entropies_list, inference_means_list, inference_stds_list):
  num_variant_runs = len(step_rewards_list)
  if num_variant_runs == 0:
    return {
      "variant": variant,
      "avg_reward": 0.0,
      "std_reward": 0.0,
      "avg_max_reward": 0.0,
      "std_max_reward": 0.0,
      "avg_max_timestep_percent": 0.0,
      "std_max_timestep_percent": 0.0,
      "avg_episode_reward": 0.0,
      "std_episode_reward": 0.0,
      "avg_max_episode_reward": 0.0,
      "std_max_episode_reward": 0.0,
      "median_max_episode_reward": 0.0,
      "avg_max_episode_timestep_percent": 0.0,
      "std_max_episode_timestep_percent": 0.0,
      "avg_auc_episode": 0.0,
      "std_auc_episode": 0.0,
      "avg_final_episode_reward": 0.0,
      "std_final_episode_reward": 0.0,
      "avg_episode_entropy": "No Data",
      "std_episode_entropy": "No Data",
      "avg_inference_mean": 0.0,
      "std_inference_mean": 0.0,
      "avg_inference_std": 0.0,
      "inference_stability": 0.0,
    }

  # Step metrics
  per_run_avg_step = [np.mean(rews) if len(rews) > 0 else 0.0 for rews in step_rewards_list]
  avg_step_reward = np.mean(per_run_avg_step)
  std_step_reward = np.std(per_run_avg_step)
  per_run_max_step = [np.max(rews) if len(rews) > 0 else 0.0 for rews in step_rewards_list]
  avg_max_step_reward = np.mean(per_run_max_step)
  std_max_step_reward = np.std(per_run_max_step)
  per_run_argmax_step = [np.argmax(rews) if len(rews) > 0 else 0 for rews in step_rewards_list]
  per_run_total_steps = [len(rews) for rews in step_rewards_list]
  per_run_percent_step = [((argmax + 1) / total * 100) if total > 0 else 0.0 for argmax, total in zip(per_run_argmax_step, per_run_total_steps)]
  avg_max_step_timestep_percent = np.mean(per_run_percent_step)
  std_max_step_timestep_percent = np.std(per_run_percent_step)

  # Episode metrics
  per_run_all_ep_rewards = [[ep["reward"] for ep in run_eps] for run_eps in episode_list]
  per_run_avg_ep = [np.mean(run_rews) if run_rews else 0.0 for run_rews in per_run_all_ep_rewards]
  avg_episode_reward = np.mean(per_run_avg_ep)
  std_episode_reward = np.std(per_run_avg_ep)
  per_run_max_ep = [max(run_rews, default=0.0) for run_rews in per_run_all_ep_rewards]
  avg_max_episode_reward = np.mean(per_run_max_ep)
  std_max_episode_reward = np.std(per_run_max_ep)
  median_max_episode_reward = np.median(per_run_max_ep)
  per_run_argmax_ep = [np.argmax(run_rews) if run_rews else 0 for run_rews in per_run_all_ep_rewards]
  per_run_timestep_ep_percent = []
  for r in range(num_variant_runs):
    run_eps = episode_list[r]
    if run_eps:
      max_idx = per_run_argmax_ep[r]
      end_ts = run_eps[max_idx]["end_timestep"]
      run_total_ts = per_run_total_steps[r]
      percent = (end_ts / run_total_ts) * 100 if run_total_ts > 0 else 0.0
    else:
      percent = 0.0
    per_run_timestep_ep_percent.append(percent)
  avg_max_episode_timestep_percent = np.mean(per_run_timestep_ep_percent)
  std_max_episode_timestep_percent = np.std(per_run_timestep_ep_percent)

  # AUC for episode reward curve
  per_run_auc = [compute_episode_auc(episode_list[r], per_run_total_steps[r]) for r in range(num_variant_runs)]
  avg_auc_episode = np.mean(per_run_auc)
  std_auc_episode = np.std(per_run_auc)

  # Final average episode reward (last 20% of timesteps)
  per_run_final_avg = []
  for r in range(num_variant_runs):
    run_eps = episode_list[r]
    run_total_ts = per_run_total_steps[r]
    threshold_ts = 0.8 * run_total_ts
    final_rews = [ep["reward"] for ep in run_eps if ep["end_timestep"] > threshold_ts]
    if final_rews:
      per_run_final_avg.append(np.mean(final_rews))
    else:
      per_run_final_avg.append(avg_episode_reward)  # fallback
  avg_final_episode_reward = np.mean(per_run_final_avg)
  std_final_episode_reward = np.std(per_run_final_avg)

  # Entropy metrics
  per_run_all_ep_entropies = episode_entropies_list
  per_run_avg_ent = []
  for run_ents in per_run_all_ep_entropies:
    if run_ents:
      per_run_avg_ent.append(np.mean(run_ents))
    else:
      per_run_avg_ent.append(np.nan)
  if all(np.isnan(e) for e in per_run_avg_ent):
    avg_episode_entropy = "No Data"
    std_episode_entropy = "No Data"
  else:
    avg_episode_entropy = np.nanmean(per_run_avg_ent)
    std_episode_entropy = np.nanstd(per_run_avg_ent)

  # Inference metrics
  avg_inference_mean = np.mean(inference_means_list)
  std_inference_mean = np.std(inference_means_list)
  avg_inference_std = np.mean(inference_stds_list)
  inference_stability = avg_inference_mean / avg_inference_std if avg_inference_std > 0 else 0.0

  return {
    "variant": variant,
    "avg_reward": avg_step_reward,
    "std_reward": std_step_reward,
    "avg_max_reward": avg_max_step_reward,
    "std_max_reward": std_max_step_reward,
    "avg_max_timestep_percent": avg_max_step_timestep_percent,
    "std_max_timestep_percent": std_max_step_timestep_percent,
    "avg_episode_reward": avg_episode_reward,
    "std_episode_reward": std_episode_reward,
    "avg_max_episode_reward": avg_max_episode_reward,
    "std_max_episode_reward": std_max_episode_reward,
    "median_max_episode_reward": median_max_episode_reward,
    "avg_max_episode_timestep_percent": avg_max_episode_timestep_percent,
    "std_max_episode_timestep_percent": std_max_episode_timestep_percent,
    "avg_auc_episode": avg_auc_episode,
    "std_auc_episode": std_auc_episode,
    "avg_final_episode_reward": avg_final_episode_reward,
    "std_final_episode_reward": std_final_episode_reward,
    "avg_episode_entropy": avg_episode_entropy,
    "std_episode_entropy": std_episode_entropy,
    "avg_inference_mean": avg_inference_mean,
    "std_inference_mean": std_inference_mean,
    "avg_inference_std": avg_inference_std,
    "inference_stability": inference_stability,
  }


def compute_model_performances(variant_names, step_rewards_lists, episode_lists, episode_entropies_lists, inference_means_lists, inference_stds_lists):
  model_performances = []
  for j in range(len(variant_names)):
    metrics = compute_variant_metrics(
      variant_names[j], step_rewards_lists[j], episode_lists[j], episode_entropies_lists[j], inference_means_lists[j], inference_stds_lists[j]
    )
    model_performances.append(metrics)
  return model_performances


def find_base_names(model_performances):
  trpo_name = next((perf["variant"] for perf in model_performances if perf["variant"].lower() == "trpo"), None)
  trpor_name = next((perf["variant"] for perf in model_performances if perf["variant"].lower() == "trpor"), None)
  return trpo_name, trpor_name


def select_top_noise_by_delta(model_performances, base_name, base_auc, noise_filter):
  noise_perfs = [p for p in model_performances if noise_filter(p["variant"].lower()) and p["variant"].lower() != base_name.lower()]
  if noise_perfs:
    noise_perfs.sort(key=lambda p: p["avg_auc_episode"] - base_auc, reverse=True)
    return noise_perfs[:2]
  return []


def get_noise_variants(model_performances, base_name, noise_keyword, extra_condition=""):
  base_perf = next((p for p in model_performances if p["variant"].lower() == base_name.lower()), None)
  if not base_perf:
    return []
  base_auc = base_perf["avg_auc_episode"]
  noise_filter = lambda v: noise_keyword in v and base_name.lower() in v and extra_condition in v
  top_2_noise = select_top_noise_by_delta(model_performances, base_name, base_auc, noise_filter)
  variants = [base_perf["variant"]] + [p["variant"] for p in top_2_noise]
  return variants


def get_best_vs_trpo(model_performances, trpo_name):
  if not model_performances or not trpo_name:
    return []
  best_variant = model_performances[0]["variant"]
  if best_variant != trpo_name:
    return [best_variant, trpo_name]
  return [trpo_name]


def get_best_trpor_vs_equiv_trpo(model_performances, variant_names):
  trpor_variants_perfs = [p for p in model_performances if "trpor" in p["variant"].lower()]
  if not trpor_variants_perfs:
    return []
  best_trpor_perf = max(trpor_variants_perfs, key=lambda x: x["avg_auc_episode"])
  best_trpor_variant = best_trpor_perf["variant"]
  equiv_trpo = best_trpor_variant.lower().replace("trpor", "trpo")
  equiv_trpo_name = next((v for v in variant_names if v.lower() == equiv_trpo), None)
  if equiv_trpo_name:
    return [best_trpor_variant, equiv_trpo_name]
  return [best_trpor_variant]


def select_top_variants(model_performances, trpo_name, trpor_name):
  top_2_noise_names = []
  for perf in model_performances:
    v = perf["variant"]
    v_low = v.lower()
    if "noise" in v_low and "trpor" in v_low and v_low not in {"trpo", "trpor"}:
      top_2_noise_names.append(v)
      if len(top_2_noise_names) == 2:
        break

  forced_names = [n for n in [trpo_name, trpor_name] if n is not None] + top_2_noise_names
  forced_names_set = set(forced_names)

  selected_names = list(forced_names_set)
  for perf in model_performances:
    v = perf["variant"]
    if v not in forced_names_set and len(selected_names) < 4:
      selected_names.append(v)

  name_to_rank = {perf["variant"]: i for i, perf in enumerate(model_performances)}
  selected_names.sort(key=lambda name: name_to_rank.get(name, float("inf")))

  return selected_names


def plot_episode_rewards(
  indices, variant_names, episode_lists, total_timesteps, downsample_factor, timesteps_np, color_map, perf_dict, compare_dir, filename, title
):
  plt.figure(figsize=(20, 10))
  for idx in indices:
    num_variant_runs = len(episode_lists[idx])
    if num_variant_runs == 0:
      continue
    ys = []
    for run in range(num_variant_runs):
      ep_infos = episode_lists[idx][run]
      if not ep_infos:
        continue
      end_tss = [ep["end_timestep"] for ep in ep_infos]
      rews = [ep["reward"] for ep in ep_infos]
      y = np.full(total_timesteps, np.nan)
      prev_ts = 0
      last_rew = 0.0
      for k in range(len(rews)):
        end_idx = min(end_tss[k], total_timesteps)
        y[prev_ts:end_idx] = last_rew
        last_rew = rews[k]
        prev_ts = end_idx
      y[prev_ts : min(prev_ts + (total_timesteps - prev_ts), total_timesteps)] = last_rew
      ys.append(y)
    if ys:
      ys_np = np.array(ys)
      full_mean_y = np.nanmean(ys_np, axis=0)
      full_std_y = np.nanstd(ys_np, axis=0)
      mean_y = full_mean_y[::downsample_factor]  # Use unsmoothed mean
      std_y = full_std_y[::downsample_factor]
      label = variant_names[idx]
      color = color_map[label]
      plt.plot(timesteps_np, mean_y, label=label, color=color)
      plt.fill_between(timesteps_np, mean_y - std_y, mean_y + std_y, color=color, alpha=0.2)
      del ys_np

  handles, labels = plt.gca().get_legend_handles_labels()
  sorted_indices = sorted(range(len(labels)), key=lambda i: perf_dict[labels[i]], reverse=True)
  handles = [handles[i] for i in sorted_indices]
  labels = [labels[i] for i in sorted_indices]

  plt.xlabel("Timesteps")
  plt.ylabel("Episode Reward (Mean ± Std)")
  plt.title(title)
  plt.legend(handles, labels)
  plt.grid(True)
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_episode_entropies(
  indices,
  variant_names,
  episode_entropies_lists,
  per_run_total_steps,
  total_timesteps,
  downsample_factor,
  timesteps_np,
  color_map,
  perf_dict,
  compare_dir,
  filename,
  title,
):
  plt.figure(figsize=(20, 10))
  has_data = False
  for idx in indices:
    num_variant_runs = len(episode_entropies_lists[idx])
    if num_variant_runs == 0:
      continue
    ys = []
    for r in range(num_variant_runs):
      entropies = episode_entropies_lists[idx][r]
      run_total_ts = per_run_total_steps[idx][r]
      if not entropies:
        continue
      num_updates = len(entropies)
      step_size = run_total_ts / num_updates if num_updates > 0 else 0
      end_tss = [int((k + 1) * step_size) for k in range(num_updates)]
      y = np.full(total_timesteps, np.nan)
      prev_ts = 0
      for k in range(num_updates):
        end_idx = min(end_tss[k], total_timesteps)
        y[prev_ts:end_idx] = entropies[k]
        prev_ts = end_idx
      if prev_ts < total_timesteps and entropies:
        y[prev_ts:] = entropies[-1]
      ys.append(y)
    if ys:
      ys_np = np.array(ys)
      full_mean_y = np.nanmean(ys_np, axis=0)
      full_std_y = np.nanstd(ys_np, axis=0)
      mean_y = full_mean_y[::downsample_factor]
      std_y = full_std_y[::downsample_factor]
      if np.all(np.isnan(mean_y)):
        continue
      has_data = True
      label = variant_names[idx]
      color = color_map[label]
      plt.plot(timesteps_np, mean_y, label=label, color=color)
      plt.fill_between(timesteps_np, mean_y - std_y, mean_y + std_y, color=color, alpha=0.2)
      del ys_np

  if not has_data:
    plt.text(0.5, 0.5, "No Entropy Data", ha="center", va="center", fontsize=20)
  else:
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_indices = sorted(range(len(labels)), key=lambda i: perf_dict[labels[i]], reverse=True)
    handles = [handles[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    plt.legend(handles, labels)

  plt.xlabel("Timesteps")
  plt.ylabel("Episode Entropy (Mean ± Std)")
  plt.title(title)
  plt.grid(True)
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_step_rewards(
  indices, variant_names, step_rewards_lists, total_timesteps, downsample_factor, timesteps_np, color_map, perf_dict, compare_dir, filename, title
):
  plt.figure(figsize=(20, 10))
  for idx in indices:
    num_runs = len(step_rewards_lists[idx])
    if num_runs == 0 or not step_rewards_lists[idx]:
      continue
    variant_step_rewards = np.full((num_runs, total_timesteps), np.nan, dtype=np.float64)
    for i in range(num_runs):
      rews = np.array(step_rewards_lists[idx][i])
      variant_step_rewards[i, : len(rews)] = rews
    mean_step_rewards = np.nanmean(variant_step_rewards, axis=0)
    std_step_rewards = np.nanstd(variant_step_rewards, axis=0)
    mean_step_rewards = mean_step_rewards[::downsample_factor]
    std_step_rewards = std_step_rewards[::downsample_factor]
    label = variant_names[idx]
    color = color_map[label]
    plt.plot(timesteps_np, mean_step_rewards, label=label, color=color)
    plt.fill_between(timesteps_np, mean_step_rewards - std_step_rewards, mean_step_rewards + std_step_rewards, color=color, alpha=0.2)
    del variant_step_rewards

  handles, labels = plt.gca().get_legend_handles_labels()
  sorted_indices = sorted(range(len(labels)), key=lambda i: perf_dict[labels[i]], reverse=True)
  handles = [handles[i] for i in sorted_indices]
  labels = [labels[i] for i in sorted_indices]

  plt.xlabel("Timesteps")
  plt.ylabel("Step Reward (Mean ± Std)")
  plt.title(title)
  plt.legend(handles, labels)
  plt.grid(True)
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_inference_bar(variant_names, variant_names_list, inference_means_lists, inference_stds_lists, color_map, compare_dir, filename, title):
  plt.figure(figsize=(10, 6))
  inference_means = []
  inference_stds = []
  for name in variant_names_list:
    idx = variant_names.index(name)
    if inference_means_lists[idx]:
      mean = np.mean(inference_means_lists[idx])
      std = np.std(inference_means_lists[idx])
      inference_means.append(mean)
      inference_stds.append(std)
    else:
      inference_means.append(0.0)
      inference_stds.append(0.0)

  colors_list = [color_map[v] for v in variant_names_list]
  plt.bar(variant_names_list, inference_means, yerr=inference_stds, capsize=5, color=colors_list)
  plt.xlabel("Variants")
  plt.ylabel("Inference Mean Reward (Mean ± Std across runs)")
  plt.title(title)
  plt.grid(True)
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def plot_scatter_variations(perfs, compare_dir, filename, title):
  if len(perfs) < 2:
    return
  plt.figure(figsize=(10, 8))
  for p in perfs:
    plt.scatter(p["avg_auc_episode"], p["avg_final_episode_reward"], label=p["variant"])
  plt.xlabel("Average AUC Episode Reward")
  plt.ylabel("Average Final Episode Reward")
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.savefig(os.path.join(compare_dir, filename))
  plt.close()


def generate_report(model_performances, env_id):
  print(f"# Model Performance Report for {env_id}")
  print("## Ranked by average AUC of episode reward curve")
  for rank, perf in enumerate(model_performances, 1):
    print(f"### Rank {rank}: {perf['variant']}")
    print(f"- Average Step Reward: {perf['avg_reward']:.4f} (± {perf['std_reward']:.4f})")
    print(f"- Average Max Step Reward: {perf['avg_max_reward']:.4f} (± {perf['std_max_reward']:.4f})")
    print(
      f"- Sample Efficiency - Average Timestep at Max Step Reward (%): {perf['avg_max_timestep_percent']:.2f}% (± {perf['std_max_timestep_percent']:.2f}%)\n"
    )
    print(f"- Average Episode Reward: {perf['avg_episode_reward']:.4f} (± {perf['std_episode_reward']:.4f})")
    print(f"- Average Max Episode Reward: {perf['avg_max_episode_reward']:.4f} (± {perf['std_max_episode_reward']:.4f})")
    print(f"- Median Max Episode Reward: {perf['median_max_episode_reward']:.4f}")
    print(
      f"- Sample Efficiency - Average Timestep at Max Episode Reward (%): {perf['avg_max_episode_timestep_percent']:.2f}% (± {perf['std_max_episode_timestep_percent']:.2f}%)\n"
    )
    print(f"- Average AUC Episode Reward: {perf['avg_auc_episode']:.4f} (± {perf['std_auc_episode']:.4f})")
    print(f"- Average Final Episode Reward (last 20% timesteps): {perf['avg_final_episode_reward']:.4f} (± {perf['std_final_episode_reward']:.4f})")
    entropy_str = (
      perf["avg_episode_entropy"]
      if isinstance(perf["avg_episode_entropy"], str)
      else f"{perf['avg_episode_entropy']:.4f} (± {perf['std_episode_entropy']:.4f})"
    )
    print(f"- Average Episode Entropy: {entropy_str}")
    print(f"- Average Inference Mean Reward: {perf['avg_inference_mean']:.4f} (± {perf['std_inference_mean']:.4f})")
    print(f"- Average Inference Std Reward: {perf['avg_inference_std']:.4f}")
    print(f"- Inference Stability (mean/std): {perf['inference_stability']:.4f}")

  print("\n## Alternative Rankings")

  auc_sorted = sorted(model_performances, key=lambda x: x["avg_auc_episode"], reverse=True)
  print("### Ranked by average AUC of episode reward curve")
  for rank, perf in enumerate(auc_sorted, 1):
    print(f"- Rank {rank}: {perf['variant']} (AUC: {perf['avg_auc_episode']:.4f})")

  final_sorted = sorted(model_performances, key=lambda x: x["avg_final_episode_reward"], reverse=True)
  print("### Ranked by average final episode reward (last 20% timesteps)")
  for rank, perf in enumerate(final_sorted, 1):
    print(f"- Rank {rank}: {perf['variant']} (Final Avg: {perf['avg_final_episode_reward']:.4f})")

  median_sorted = sorted(model_performances, key=lambda x: x["median_max_episode_reward"], reverse=True)
  print("### Ranked by median max episode reward")
  for rank, perf in enumerate(median_sorted, 1):
    print(f"- Rank {rank}: {perf['variant']} (Median Max: {perf['median_max_episode_reward']:.4f})")

  print("\n## Performance Comparison Table (Mean ± Std over runs)")
  print("| Variant | Final Reward | AUC | Max Reward | Median Max Reward | Entropy |")
  print("|---------|--------------|-----|------------|-------------------|---------|")
  for perf in sorted(model_performances, key=lambda x: x["avg_final_episode_reward"], reverse=True):
    final_str = f"{perf['avg_final_episode_reward']:.4f} ± {perf['std_final_episode_reward']:.4f}"
    auc_str = f"{perf['avg_auc_episode']:.4f} ± {perf['std_auc_episode']:.4f}"
    max_str = f"{perf['avg_max_episode_reward']:.4f} ± {perf['std_max_episode_reward']:.4f}"
    median_str = f"{perf['median_max_episode_reward']:.4f}"
    entropy_str = (
      perf["avg_episode_entropy"] if isinstance(perf["avg_episode_entropy"], str) else f"{perf['avg_episode_entropy']:.4f} ± {perf['std_episode_entropy']:.4f}"
    )
    print(f"| {perf['variant']} | {final_str} | {auc_str} | {max_str} | {median_str} | {entropy_str} |")


def report(compare_dir, env_id):
  print(f"Generating report for experiments in {compare_dir} on environment {env_id}...")
  variant_dict = load_data(compare_dir)
  print(f"Found variants: {len(variant_dict)}")

  variant_names, step_rewards_lists, episode_lists, episode_entropies_lists, inference_means_lists, inference_stds_lists = prepare_lists(variant_dict)

  colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(variant_names)))
  color_map = dict(zip(variant_names, colors))

  total_timesteps, downsample_factor, timesteps_np = compute_timesteps_and_downsample(step_rewards_lists)

  # Collect per_run_total_steps for entropy plotting
  per_run_total_steps = [[len(run_data["step_rewards"]) for _, run_data in sorted(variant_dict[variant])] for variant in variant_names]

  model_performances = compute_model_performances(
    variant_names, step_rewards_lists, episode_lists, episode_entropies_lists, inference_means_lists, inference_stds_lists
  )

  model_performances.sort(key=lambda x: x["avg_auc_episode"], reverse=True)

  trpo_name, trpor_name = find_base_names(model_performances)

  trpor_noise_variants = get_noise_variants(model_performances, "trpor", "noise", "")

  trpo_noise_variants = get_noise_variants(model_performances, "trpo", "noise", " and 'trpor' not in v")

  best_vs_trpo_variants = get_best_vs_trpo(model_performances, trpo_name)

  best_trpor_vs_equiv_trpo = get_best_trpor_vs_equiv_trpo(model_performances, variant_names)

  top_variant_names = select_top_variants(model_performances, trpo_name, trpor_name)
  top_indices = [variant_names.index(v) for v in top_variant_names if v in variant_names]

  perf_dict = {perf["variant"]: perf["avg_auc_episode"] for perf in model_performances}

  plot_step_rewards(
    top_indices,
    variant_names,
    step_rewards_lists,
    total_timesteps,
    downsample_factor,
    timesteps_np,
    color_map,
    perf_dict,
    compare_dir,
    "step_plot.png",
    f"Step Reward Training Performance Comparison on {env_id} (Top Variants)",
  )

  plot_episode_rewards(
    top_indices,
    variant_names,
    episode_lists,
    total_timesteps,
    downsample_factor,
    timesteps_np,
    color_map,
    perf_dict,
    compare_dir,
    "episode_plot.png",
    f"Episode Reward Training Performance Comparison on {env_id} (Top Variants)",
  )

  plot_episode_entropies(
    top_indices,
    variant_names,
    episode_entropies_lists,
    per_run_total_steps,
    total_timesteps,
    downsample_factor,
    timesteps_np,
    color_map,
    perf_dict,
    compare_dir,
    "entropy_plot.png",
    f"Episode Entropy Training Performance Comparison on {env_id} (Top Variants)",
  )

  plot_inference_bar(
    variant_names,
    top_variant_names,
    inference_means_lists,
    inference_stds_lists,
    color_map,
    compare_dir,
    "inference_plot.png",
    f"Inference Performance Comparison on {env_id} (Top Variants)",
  )

  if trpor_noise_variants:
    trpor_noise_indices = [variant_names.index(v) for v in trpor_noise_variants if v in variant_names]
    plot_episode_rewards(
      trpor_noise_indices,
      variant_names,
      episode_lists,
      total_timesteps,
      downsample_factor,
      timesteps_np,
      color_map,
      perf_dict,
      compare_dir,
      "episode_trpor_noise_delta.png",
      f"Episode Reward: TRPOR and Top 2 Noise Variants by AUC Delta on {env_id}",
    )

  if trpo_noise_variants:
    trpo_noise_indices = [variant_names.index(v) for v in trpo_noise_variants if v in variant_names]
    plot_episode_rewards(
      trpo_noise_indices,
      variant_names,
      episode_lists,
      total_timesteps,
      downsample_factor,
      timesteps_np,
      color_map,
      perf_dict,
      compare_dir,
      "episode_trpo_noise_delta.png",
      f"Episode Reward: TRPO and Top 2 Noise Variants by AUC Delta on {env_id}",
    )

  if best_vs_trpo_variants:
    best_vs_trpo_indices = [variant_names.index(v) for v in best_vs_trpo_variants if v in variant_names]
    plot_episode_rewards(
      best_vs_trpo_indices,
      variant_names,
      episode_lists,
      total_timesteps,
      downsample_factor,
      timesteps_np,
      color_map,
      perf_dict,
      compare_dir,
      "episode_best_vs_trpo.png",
      f"Episode Reward: Best Model vs TRPO on {env_id}",
    )

  if best_trpor_vs_equiv_trpo:
    best_trpor_vs_equiv_indices = [variant_names.index(v) for v in best_trpor_vs_equiv_trpo if v in variant_names]
    plot_episode_rewards(
      best_trpor_vs_equiv_indices,
      variant_names,
      episode_lists,
      total_timesteps,
      downsample_factor,
      timesteps_np,
      color_map,
      perf_dict,
      compare_dir,
      "episode_best_trpor_vs_equiv_trpo.png",
      f"Episode Reward: Best TRPOR Variant vs Equivalent TRPO on {env_id}",
    )

  # Best vs worst noise for TRPOR
  trpor_noise_perfs = [p for p in model_performances if "noise" in p["variant"].lower() and "trpor" in p["variant"].lower() and p["variant"].lower() != "trpor"]
  if trpor_noise_perfs:
    trpor_noise_perfs.sort(key=lambda p: p["avg_auc_episode"], reverse=True)
    best_trpor_noise = trpor_noise_perfs[0]["variant"]
    worst_trpor_noise = trpor_noise_perfs[-1]["variant"]
    trpor_best_worst_noise = [best_trpor_noise, worst_trpor_noise]
    trpor_best_worst_indices = [variant_names.index(v) for v in trpor_best_worst_noise if v in variant_names]
    plot_episode_rewards(
      trpor_best_worst_indices,
      variant_names,
      episode_lists,
      total_timesteps,
      downsample_factor,
      timesteps_np,
      color_map,
      perf_dict,
      compare_dir,
      "episode_trpor_best_worst_noise.png",
      f"Episode Reward: Best vs Worst TRPOR Noise on {env_id}",
    )

  # Best vs worst noise for TRPO
  trpo_noise_perfs = [
    p
    for p in model_performances
    if "noise" in p["variant"].lower() and "trpo" in p["variant"].lower() and "trpor" not in p["variant"].lower() and p["variant"].lower() != "trpo"
  ]
  if trpo_noise_perfs:
    trpo_noise_perfs.sort(key=lambda p: p["avg_auc_episode"], reverse=True)
    best_trpo_noise = trpo_noise_perfs[0]["variant"]
    worst_trpo_noise = trpo_noise_perfs[-1]["variant"]
    trpo_best_worst_noise = [best_trpo_noise, worst_trpo_noise]
    trpo_best_worst_indices = [variant_names.index(v) for v in trpo_best_worst_noise if v in variant_names]
    plot_episode_rewards(
      trpo_best_worst_indices,
      variant_names,
      episode_lists,
      total_timesteps,
      downsample_factor,
      timesteps_np,
      color_map,
      perf_dict,
      compare_dir,
      "episode_trpo_best_worst_noise.png",
      f"Episode Reward: Best vs Worst TRPO Noise on {env_id}",
    )

  # Scatter plot for TRPOR variations
  trpor_perfs = [p for p in model_performances if "trpor" in p["variant"].lower()]
  plot_scatter_variations(trpor_perfs, compare_dir, "scatter_trpor_variations.png", f"Scatter: TRPOR Variations on {env_id} (AUC vs Final Reward)")

  # Scatter plot for TRPO variations
  trpo_perfs = [p for p in model_performances if "trpo" in p["variant"].lower() and "trpor" not in p["variant"].lower()]
  plot_scatter_variations(trpo_perfs, compare_dir, "scatter_trpo_variations.png", f"Scatter: TRPO Variations on {env_id} (AUC vs Final Reward)")

  generate_report(model_performances, env_id)


if __name__ == "__main__":
  base_compare_dir = "assets"
  subdirs = sorted(os.listdir(base_compare_dir))
  print("# Table of Contents")
  for subdir in subdirs:
    env_id = subdir.split("_")[0].lower()
    print(f"- [{env_id}](#model-performance-report-for-{env_id})")
  print("")
  for subdir in subdirs:
    env_id = subdir.split("_")[0]
    compare_dir = os.path.join(base_compare_dir, subdir)
    print(f"Generating report for {env_id} in {compare_dir}...")
    report(compare_dir, env_id)
