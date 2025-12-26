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


class TRPOR(TRPO):
  """
    TRPOR: Entropy-Regularized Trust Region Policy Optimization with Reinforcement Learning

    This is an extension of the standard Trust Region Policy Optimization (TRPO) algorithm
    that incorporates entropy regularization into the policy objective. The entropy term
    can be applied as either a bonus (to encourage exploration) or a penalty (to handicap uncertainty
    and favor confident paths), controlled by the `entropy_mode` hyperparameter.

    Additional metrics are logged to measure the hypothesis that the bonus mode may hijack the line search
    when the entropy term dominates the raw surrogate, leading to suboptimal updates, while penalty mode
    avoids this by handicapping high-entropy proposals.

    Key Added Metrics:
    - raw_policy_objective: The surrogate objective without entropy regularization.
    - entropy_regularization_term: The entropy term applied (positive for bonus, negative for penalty in effect).
    - regularization_ratio: |entropy_term| / |raw_objective| to measure dominance.
    - raw_improvement: Improvement in raw surrogate after update.
    - avg_line_search_coeff: Average backtracking coefficient (closer to 1 indicates less backtracking, potentially tiny steps in bonus mode).

    Mathematical Formulation:
    -------------------------
    Standard TRPO objective:
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) ]

    TRPOR modified objective (bonus mode):
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) + α * H(π_θ) ]

    TRPOR modified objective (penalty mode):
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) - α * |H(π_θ)| ]

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
    entropy_mode : str, optional
        Mode for entropy regularization: 'bonus' to add entropy or 'penalty' to subtract (default: 'penalty').
    ent_coef : float, optional
        Entropy coefficient controlling the strength of the entropy term (default: 0.01).
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
    - **Entropy Term:** Adds entropy to the policy objective as bonus or penalty based on mode.
    - **Policy Objective:** Modified to include the entropy coefficient (`ent_coef`).
    - **Line Search:** Considers the entropy term while checking policy improvement.
    - **Logging:** Logs entropy-regularized objectives, KL divergence values, and hypothesis-measuring metrics.

    """

  def __init__(self, *args, entropy_mode: str = "penalty", ent_coef=0.01, batch_size=32, normalize_entropy=False, **kwargs):
    if entropy_mode not in ["bonus", "penalty"]:
      raise ValueError("entropy_mode must be 'bonus' or 'penalty'")
    super().__init__(*args, **kwargs)

    self.entropy_mode = entropy_mode
    self.ent_coef = ent_coef
    self.batch_size = batch_size
    self.normalize_entropy = normalize_entropy

  def _compute_policy_objective(self, advantages, ratio, distribution):
    """Overridable method for computing policy objective."""
    policy_objective = (advantages * ratio).mean()
    entropy = distribution.entropy()

    entropy_term = entropy.mean()  # Fallback to original

    # Normalize entropy to match surrogate magnitude if enabled
    if self.normalize_entropy:
      surrogate_mag = th.abs(policy_objective) + 1e-8
      entropy_mag = th.abs(entropy_term) + 1e-8
      scale_factor = surrogate_mag / entropy_mag
      entropy_term = entropy_term * scale_factor

    # Compute raw regularization term
    raw_regularization_term = self.ent_coef * entropy_term

    # Always apply absolute for consistency
    regularization_term = th.abs(raw_regularization_term)

    # Apply based on mode
    if self.entropy_mode == "bonus":
      new_policy_objective = policy_objective + regularization_term
      applied_term = regularization_term  # Positive for bonus
    elif self.entropy_mode == "penalty":
      new_policy_objective = policy_objective - regularization_term
      applied_term = -regularization_term  # Negative for penalty

    self.applied_entropy_term = applied_term  # Store for logging

    return new_policy_objective

  def train(self) -> None:
    """
        Update policy using the currently gathered rollout buffer.
        """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    raw_policy_objective_values = []
    entropy_terms = []
    regularization_ratios = []
    raw_improvements = []
    kl_divergences = []
    line_search_results = []
    line_search_coeffs = []
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

      # Raw surrogate policy objective (initial, should be ~0)
      initial_raw_policy_objective = (advantages * ratio).mean()

      # surrogate policy objective with entropy regularization (matches TRPOR)
      policy_objective = self._compute_policy_objective(advantages, ratio, distribution)
      applied_entropy_term = self.applied_entropy_term
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

          # Update ratio with new log_prob
          ratio = th.exp(log_prob - old_log_prob)

          # New raw policy objective
          raw_new_policy_objective = (advantages * ratio).mean()
          new_policy_objective = self._compute_policy_objective(advantages, ratio, distribution)
          new_applied_entropy_term = self.applied_entropy_term

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
          raw_policy_objective_values.append(initial_raw_policy_objective.item())
          entropy_terms.append(applied_entropy_term.item())
          regularization_ratios.append(abs(applied_entropy_term.item()) / (abs(initial_raw_policy_objective.item()) + 1e-8))
          raw_improvements.append(0.0)  # No improvement
          kl_divergences.append(0.0)
          line_search_coeffs.append(0.0)  # No step
        else:
          policy_objective_values.append(new_policy_objective.item())
          raw_policy_objective_values.append(raw_new_policy_objective.item())
          entropy_terms.append(new_applied_entropy_term.item())
          reg_ratio = abs(new_applied_entropy_term.item()) / (abs(raw_new_policy_objective.item()) + 1e-8)
          regularization_ratios.append(reg_ratio)
          raw_improvements.append(raw_new_policy_objective.item() - initial_raw_policy_objective.item())
          kl_divergences.append(kl_div.item())
          line_search_coeffs.append(line_search_backtrack_coeff)

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
      # Add new for save
      raw_policy_objective_values,
      entropy_terms,
      regularization_ratios,
      raw_improvements,
      line_search_coeffs,
    )

    # Logs
    # add surrogate objective and entropy
    self.logger.record("train/entropy", np.mean(entropies))
    self.logger.record("train/advantage_mean", np.mean(advantages_numpy))
    self.logger.record("train/advantage_std", np.std(advantages_numpy))
    self.logger.record("train/policy_objective", np.mean(policy_objective_values))
    self.logger.record("train/raw_policy_objective", np.mean(raw_policy_objective_values))
    self.logger.record("train/entropy_regularization_term", np.mean(entropy_terms))
    self.logger.record("train/regularization_ratio", np.mean(regularization_ratios))
    self.logger.record("train/raw_improvement", np.mean(raw_improvements))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", np.mean(line_search_results))
    self.logger.record("train/avg_line_search_coeff", np.mean(line_search_coeffs))
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
