import copy
from functools import partial

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from sb3.noise import MonitoredEntropyInjectionWrapper
from sb3.trpo import TRPO


def debug_summary(policy_objective: th.Tensor, raw_regularization_term: th.Tensor, regularization_term: th.Tensor):
  """
    Prints a short summary of the entropy regularization impact, including sign (p/n) and percentage magnitude relative to the surrogate.
    Handles scalars or means of batches.
    """
  # Get scalar values (mean if batched)
  surrogate = policy_objective.mean().item() if policy_objective.numel() > 1 else policy_objective.item()
  original_reg = raw_regularization_term.mean().item() if raw_regularization_term.numel() > 1 else raw_regularization_term.item()
  applied_reg = regularization_term.mean().item() if regularization_term.numel() > 1 else regularization_term.item()

  # Percentage magnitude (avoid div by zero)
  if abs(surrogate) > 1e-8:
    percent_change = (applied_reg / abs(surrogate)) * 100
  else:
    percent_change = 0.0  # Or inf, but cap at 0 for safety

  # Short message
  print(f"Reg: -{percent_change:.2f}%")


def compute_surrogate_objective_entropy(policy_objective: th.Tensor, entropy: th.Tensor, ent_coef: th.Tensor, debug: bool = False) -> th.Tensor:
  """
    We handicap the surrogate objective with entropy to promote exploration.
    We need to ensure non negative subtractions.
    if the policy is negative, double negative gives us positive. which is really bad for our goal of exploratory improvement and monotonic improvement guarantee.
    Optionally print a summary of the impact of the entropy regularization (disabled by default for performance).

    Mathematically:

    Let S denote the surrogate policy objective (a scalar value).

    Let E denote the entropy term (a scalar, which may be positive or negative depending on context).

    Let c > 0 denote the entropy coefficient.

    The raw regularization term is r = c * E.

    To always apply a penalty that helps choose better gradients, we use the absolute magnitude:

    J = S - |r|

    This ensures a consistent downward handicap, preventing boosts and stabilizing the system, especially in cases where signs could lead to unintended effects.

    Supports batched inputs for vectorized computation.
    """
  # Compute raw regularization term (handles batches via broadcasting)
  raw_regularization_term = ent_coef * entropy

  # Always apply absolute penalty
  regularization_term = th.abs(raw_regularization_term)

  new_policy_objective = policy_objective - regularization_term

  if debug:
    debug_summary(policy_objective, raw_regularization_term, regularization_term)

  return new_policy_objective


class TRPOR(TRPO):
  """
    TRPOR: Entropy-Regularized Trust Region Policy Optimization with Reinforcement Learning

    This is an extension of the standard Trust Region Policy Optimization (TRPO) algorithm
    that incorporates entropy regularization into the policy objective. The entropy bonus
    encourages exploration and prevents premature convergence to deterministic policies.

    Key Features:
    - Adds an entropy bonus term to the policy objective to promote exploration.
    - Retains the KL-divergence constraint from TRPO for stable policy updates.
    - The entropy coefficient (`ent_coef`) is tunable for balancing exploration and exploitation.
    - Suitable for environments with sparse rewards where additional exploration is needed.

    Mathematical Formulation:
    -------------------------
    Standard TRPO objective:
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) ]

    TRPOR modified objective:
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) + α * H(π_θ) ]

    where:
    - π_θ is the current policy.
    - π_θ_old is the old policy.
    - Â is the advantage function.
    - α (`ent_coef`) is the entropy coefficient.
    - H(π_θ) is the entropy of the policy.

    Parameters:
    -----------
    policy : Union[str, type[ActorCriticPolicy]]
        The policy model to be used (e.g., "MlpPolicy").
    env : Union[GymEnv, str]
        The environment to learn from.
    ent_coef : float, optional
        Entropy coefficient controlling the strength of the entropy bonus (default: 0.01).
    learning_rate : Union[float, Schedule], optional
        Learning rate for the optimizer (default: 1e-3).
    n_steps : int, optional
        Number of steps to run per update (default: 2048).
    batch_size : int, optional
        Minibatch size for the value function updates (default: 128).
    gamma : float, optional
        Discount factor for the reward (default: 0.99).
    cg_max_steps : int, optional
        Maximum steps for the conjugate gradient solver (default: 10).
    target_kl : float, optional
        Target KL divergence for policy updates (default: 0.01).

    Differences from Standard TRPO:
    -------------------------------
    - **Entropy Bonus:** Adds entropy to the policy objective for better exploration.
    - **Policy Objective:** Modified to include the entropy coefficient (`ent_coef`).
    - **Line Search:** Considers the entropy term while checking policy improvement.
    - **Logging:** Logs entropy-regularized objectives and KL divergence values.

    """

  def __init__(self, *args, ent_coef=0.01, batch_size=32, **kwargs):
    super().__init__(*args, **kwargs)

    self.ent_coef = ent_coef
    self.batch_size = batch_size

  def train(self) -> None:
    """
        Update policy using the currently gathered rollout buffer.
        """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    line_search_results = []
    value_losses = []

    # This will only loop once (get all data in one go)
    for rollout_data in self.rollout_buffer.get(batch_size=None):

      # Optional: sub-sample data for faster computation
      if self.sub_sampling_factor > 1:
        indices = slice(None, None, self.sub_sampling_factor)
        rollout_data = RolloutBufferSamples(
          rollout_data.observations[indices],
          rollout_data.actions[indices],
          None,  # type: ignore[arg-type]  # old values, not used here
          rollout_data.old_log_prob[indices],
          rollout_data.advantages[indices],
          rollout_data.returns[indices],  # Include returns for critic
        )

      observations = rollout_data.observations
      actions = rollout_data.actions
      returns = rollout_data.returns
      advantages = rollout_data.advantages
      old_log_prob = rollout_data.old_log_prob

      if isinstance(self.action_space, spaces.Discrete):
        # Convert discrete action from float to long
        actions = actions.long().flatten()

      with th.no_grad():
        # Use deepcopy for safety with complex distributions
        old_distribution = copy.deepcopy(self.policy.get_distribution(observations))

      distribution = self.policy.get_distribution(observations)
      log_prob = distribution.log_prob(actions)

      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      # ratio between old and new policy, should be one at the first iteration
      ratio = th.exp(log_prob - old_log_prob)

      # surrogate policy objective with entropy regularization (matches TRPOR)
      policy_objective = (advantages * ratio).mean()
      policy_objective = compute_surrogate_objective_entropy(
        policy_objective,
        distribution.entropy().mean(),
        self.ent_coef,
      )

      # KL divergence
      kl_div = kl_divergence(distribution, old_distribution).mean()

      # Surrogate & KL gradient
      self.policy.optimizer.zero_grad()

      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

      # Hessian-vector dot product function used in the conjugate gradient step
      hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

      # Computing search direction
      search_direction = conjugate_gradient_solver(
        hessian_vector_product_fn,
        policy_objective_gradients,
        max_iter=self.cg_max_steps,
      )

      # Maximal step length
      line_search_max_step_size = 2 * self.target_kl
      line_search_max_step_size /= th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
      line_search_max_step_size = th.sqrt(line_search_max_step_size)  # type: ignore[assignment, arg-type]

      line_search_backtrack_coeff = 1.0
      original_actor_params = [param.detach().clone() for param in actor_params]

      is_line_search_success = False
      with th.no_grad():
        # Line-search (backtracking)
        for _ in range(self.line_search_max_iter):
          start_idx = 0
          # Applying the scaled step direction
          for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
            n_params = param.numel()
            param.data = original_param.data + line_search_backtrack_coeff * line_search_max_step_size * search_direction[
              start_idx : (start_idx + n_params)
            ].view(shape)
            start_idx += n_params

          # Recomputing the policy log-probabilities
          distribution = self.policy.get_distribution(observations)
          log_prob = distribution.log_prob(actions)

          # New policy objective
          ratio = th.exp(log_prob - old_log_prob)
          line_search_entropy = self.ent_coef * distribution.entropy().mean()
          new_policy_objective = (advantages * ratio).mean()
          new_policy_objective = compute_surrogate_objective_entropy(
            new_policy_objective,
            distribution.entropy().mean(),
            self.ent_coef,
          )

          # New KL-divergence
          kl_div = kl_divergence(distribution, old_distribution).mean()

          # Constraint criteria:
          # we need to improve the surrogate policy objective
          # while being close enough (in term of kl div) to the old policy
          if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
            is_line_search_success = True
            break

          # Reducing step size if line-search wasn't successful
          line_search_backtrack_coeff *= self.line_search_shrinking_factor

        line_search_results.append(is_line_search_success)

        if not is_line_search_success:
          # If the line-search wasn't successful we revert to the original parameters
          for param, original_param in zip(actor_params, original_actor_params):
            param.data = original_param.data.clone()

          policy_objective_values.append(policy_objective.item())
          kl_divergences.append(0.0)
        else:
          policy_objective_values.append(new_policy_objective.item())
          kl_divergences.append(kl_div.item())

    # Critic update
    value_grad_norms = []
    for _ in range(self.n_critic_updates):
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        values_pred = self.policy.predict_values(rollout_data.observations)
        value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
        value_losses.append(value_loss.item())

        self.policy.optimizer.zero_grad()
        value_loss.backward()
        grad_norm_value = 0.0
        for p in self.policy.value_net.parameters():
          if p.grad is not None:
            grad_norm_value += p.grad.norm(2) ** 2
        grad_norm_value = grad_norm_value**0.5
        value_grad_norms.append(grad_norm_value)
        # Removing gradients of parameters shared with the actor
        # otherwise it defeats the purposes of the KL constraint
        for param in actor_params:
          param.grad = None
        self.policy.optimizer.step()

    self._n_updates += 1
    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Additional metrics
    if hasattr(self.policy, "log_std"):
      policy_stds = th.exp(self.policy.log_std).detach().cpu().numpy()
    else:
      policy_stds = np.array([0.0])

    # Entropy
    distribution = self.policy.get_distribution(rollout_data.observations)
    entropies = distribution.entropy().detach().cpu().numpy()

    # Noise stats
    action_deltas = []
    reward_deltas = []
    if hasattr(self.env, "envs"):
      for e in self.env.envs:
        if isinstance(e, MonitoredEntropyInjectionWrapper):
          a_deltas, r_deltas = e.get_noise_deltas()
          action_deltas.extend(a_deltas)
          reward_deltas.extend(r_deltas)
    elif isinstance(self.env, MonitoredEntropyInjectionWrapper):
      action_deltas, reward_deltas = self.env.get_noise_deltas()

    advantages_numpy = advantages.detach().cpu().numpy()
    po = policy_objective.detach().cpu().numpy()
    rewards = self.rollout_buffer.rewards.flatten()

    self._save_rollout_metrics(
      kl_divergences,
      explained_var,
      value_losses,
      policy_stds,
      line_search_results,
      None,  #
      value_grad_norms,
      advantages_numpy,
      entropies,
      action_deltas,
      reward_deltas,
      rewards,
      po,
      kl_div,
    )

    # Logs
    # add serogate objecive and antropy
    self.logger.record("train/entropy", np.mean(entropies))
    self.logger.record("train/advantage_mean", np.mean(advantages_numpy))
    self.logger.record("train/advantage_std", np.std(advantages_numpy))
    self.logger.record("train/policy_objective", np.mean(policy_objective_values))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", np.mean(line_search_results))
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


class TRPOR2(TRPO):
  """
    TRPOR: Entropy-Regularized Trust Region Policy Optimization with Reinforcement Learning

    This is an extension of the standard Trust Region Policy Optimization (TRPO) algorithm
    that incorporates entropy regularization into the policy objective. The entropy bonus
    encourages exploration and prevents premature convergence to deterministic policies.

    Key Features:
    - Adds an entropy bonus term to the policy objective to promote exploration.
    - Retains the KL-divergence constraint from TRPO for stable policy updates.
    - The entropy coefficient (`ent_coef`) is tunable for balancing exploration and exploitation.
    - Adaptive target KL: dynamically adjusts the trust region based on entropy to prevent
      premature over-exploitation and performance collapse in sparse/deceptive environments.
    - Corrected line-search: now properly accounts for the entropy term in the improvement check.

    Parameters:
    -----------
    policy : Union[str, type[ActorCriticPolicy]]
        The policy model to be used (e.g., "MlpPolicy").
    env : Union[GymEnv, str]
        The environment to learn from.
    ent_coef : float, optional
        Entropy coefficient controlling the strength of the entropy bonus (default: 0.01).
    learning_rate : Union[float, Schedule], optional
        Learning rate for the optimizer (default: 1e-3).
    n_steps : int, optional
        Number of steps to run per update (default: 2048).
    batch_size : int, optional
        Minibatch size for the value function updates (default: 128).
    gamma : float, optional
        Discount factor for the reward (default: 0.99).
    cg_max_steps : int, optional
        Maximum steps for the conjugate gradient solver (default: 10).
    target_kl : float, optional
        Base target KL divergence for policy updates (default: 0.01). Adapted dynamically if adaptive_kl_power > 0.
    adaptive_kl_power : float, optional
        Exponent for entropy ratio in adaptive KL scaling (default: 0.5). Set to 0.0 to disable adaptation.
    adaptive_kl_min_mult / adaptive_kl_max_mult : float, optional
        Bounds for the adaptive multiplier (default: 0.2 and 5.0).
    """

  def __init__(
    self,
    *args,
    ent_coef=0.01,
    batch_size=32,
    target_kl=0.01,
    adaptive_kl_power=0.5,
    adaptive_kl_min_mult=0.2,
    adaptive_kl_max_mult=5.0,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)

    self.ent_coef = ent_coef
    self.batch_size = batch_size

    # Adaptive KL parameters
    self.base_target_kl = target_kl
    self.target_kl = target_kl
    self.adaptive_kl_power = adaptive_kl_power
    self.adaptive_kl_min_mult = adaptive_kl_min_mult
    self.adaptive_kl_max_mult = adaptive_kl_max_mult

    self.initial_entropy = None
    self.ent_ema = None
    self.ent_ema_alpha = 0.1  # EMA smoothing factor

  def train(self) -> None:
    """
        Update policy using the currently gathered rollout buffer.
        """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    line_search_results = []
    value_losses = []
    value_grad_norms = []

    # This will only loop once (get all data in one go)
    for rollout_data in self.rollout_buffer.get(batch_size=None):

      # Optional: sub-sample data for faster computation
      if self.sub_sampling_factor > 1:
        indices = slice(None, None, self.sub_sampling_factor)
        rollout_data = RolloutBufferSamples(
          rollout_data.observations[indices],
          rollout_data.actions[indices],
          None,  # old values, not used here
          rollout_data.old_log_prob[indices],
          rollout_data.advantages[indices],
          rollout_data.returns[indices],
        )

      observations = rollout_data.observations
      actions = rollout_data.actions
      returns = rollout_data.returns
      advantages = rollout_data.advantages
      old_log_prob = rollout_data.old_log_prob

      if isinstance(self.action_space, spaces.Discrete):
        actions = actions.long().flatten()

      with th.no_grad():
        old_distribution = copy.deepcopy(self.policy.get_distribution(observations))

      distribution = self.policy.get_distribution(observations)
      log_prob = distribution.log_prob(actions)

      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      ratio = th.exp(log_prob - old_log_prob)

      # Correct entropy-regularized surrogate objective
      old_entropy = old_distribution.entropy().mean()
      old_surrogate = (advantages * ratio).mean()  # ratio ≈ 1 initially
      old_objective = old_surrogate + self.ent_coef * old_entropy

      # Track entropy for adaptation (use pre-update entropy)
      current_entropy_val = old_entropy.item()

      if self.initial_entropy is None:
        self.initial_entropy = current_entropy_val

      if self.ent_ema is None:
        self.ent_ema = current_entropy_val
      else:
        self.ent_ema = self.ent_ema_alpha * current_entropy_val + (1 - self.ent_ema_alpha) * self.ent_ema

      # KL divergence (initial)
      kl_div = kl_divergence(distribution, old_distribution).mean()

      # Compute gradients on the entropy-regularized objective
      self.policy.optimizer.zero_grad()
      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, old_objective)

      hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

      search_direction = conjugate_gradient_solver(
        hessian_vector_product_fn,
        policy_objective_gradients,
        max_iter=self.cg_max_steps,
      )

      # Maximal step length based on current target_kl
      line_search_max_step_size = 2 * self.target_kl
      line_search_max_step_size /= th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
      line_search_max_step_size = th.sqrt(line_search_max_step_size)

      line_search_backtrack_coeff = 1.0
      original_actor_params = [param.detach().clone() for param in actor_params]

      is_line_search_success = False
      with th.no_grad():
        for _ in range(self.line_search_max_iter):
          start_idx = 0
          for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
            n_params = param.numel()
            param.data = original_param.data + line_search_backtrack_coeff * line_search_max_step_size * search_direction[
              start_idx : (start_idx + n_params)
            ].view(shape)
            start_idx += n_params

          distribution = self.policy.get_distribution(observations)
          log_prob = distribution.log_prob(actions)
          ratio = th.exp(log_prob - old_log_prob)

          new_surrogate = (advantages * ratio).mean()
          new_entropy = distribution.entropy().mean()
          new_objective = new_surrogate + self.ent_coef * new_entropy

          kl_div = kl_divergence(distribution, old_distribution).mean()

          if (kl_div < self.target_kl) and (new_objective > old_objective + 1e-6):
            is_line_search_success = True
            break

          line_search_backtrack_coeff *= self.line_search_shrinking_factor

        line_search_results.append(is_line_search_success)

        if not is_line_search_success:
          for param, original_param in zip(actor_params, original_actor_params):
            param.data = original_param.data.clone()

          policy_objective_values.append(old_objective.item())
          kl_divergences.append(0.0)
        else:
          policy_objective_values.append(new_objective.item())
          kl_divergences.append(kl_div.item())

          # Update adaptive target_kl only on successful updates
          if self.adaptive_kl_power > 0.0:
            entropy_ratio = self.ent_ema / (self.initial_entropy + 1e-8)
            mult = entropy_ratio**self.adaptive_kl_power
            mult = mult.clamp(self.adaptive_kl_min_mult, self.adaptive_kl_max_mult)
            self.target_kl = self.base_target_kl * mult

    # Critic update (unchanged from original)
    for _ in range(self.n_critic_updates):
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        values_pred = self.policy.predict_values(rollout_data.observations)
        value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
        value_losses.append(value_loss.item())

        self.policy.optimizer.zero_grad()
        value_loss.backward()
        grad_norm_value = 0.0
        for p in self.policy.value_net.parameters():
          if p.grad is not None:
            grad_norm_value += p.grad.norm(2) ** 2
        grad_norm_value = grad_norm_value**0.5
        value_grad_norms.append(grad_norm_value)

        for param in actor_params:
          param.grad = None
        self.policy.optimizer.step()

    self._n_updates += 1
    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Metrics (using last rollout_data for entropy/std)
    if hasattr(self.policy, "log_std"):
      policy_stds = th.exp(self.policy.log_std).detach().cpu().numpy()
    else:
      policy_stds = np.array([0.0])

    distribution = self.policy.get_distribution(rollout_data.observations)
    entropies = distribution.entropy().detach().cpu().numpy()

    # Noise stats (if applicable)
    action_deltas = []
    reward_deltas = []
    if hasattr(self.env, "envs"):
      for e in self.env.envs:
        if isinstance(e, MonitoredEntropyInjectionWrapper):
          a_deltas, r_deltas = e.get_noise_deltas()
          action_deltas.extend(a_deltas)
          reward_deltas.extend(r_deltas)
    elif isinstance(self.env, MonitoredEntropyInjectionWrapper):
      action_deltas, reward_deltas = self.env.get_noise_deltas()

    advantages_numpy = advantages.detach().cpu().numpy()
    rewards = self.rollout_buffer.rewards.flatten()

    # Use means for saved metrics (fixes potential scope issues in original)
    mean_policy_objective = np.mean(policy_objective_values) if policy_objective_values else 0.0
    mean_kl = np.mean(kl_divergences) if kl_divergences else 0.0

    self._save_rollout_metrics(
      kl_divergences,
      explained_var,
      value_losses,
      policy_stds,
      line_search_results,
      None,
      value_grad_norms,
      advantages_numpy,
      entropies,
      action_deltas,
      reward_deltas,
      rewards,
      mean_policy_objective,
      mean_kl,
    )

    # Logs
    self.logger.record("train/entropy", np.mean(entropies))
    self.logger.record("train/advantage_mean", np.mean(advantages_numpy))
    self.logger.record("train/advantage_std", np.std(advantages_numpy))
    self.logger.record("train/policy_objective", np.mean(policy_objective_values))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", np.mean(line_search_results))
    self.logger.record("train/target_kl_current", self.target_kl)
    self.logger.record("train/entropy_ema", self.ent_ema)
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
