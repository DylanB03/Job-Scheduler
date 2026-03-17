from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from config.base_config import hyperparams_config
from envs.queue_env import QueueEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size: int = 128):
        super().__init__()

        obs_dim = int(np.array(obs_space.shape).prod())
        action_dim = action_space.n

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.shared(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(x)
        return value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy, value


class PPOAgent:
    def __init__(self, env: Optional[QueueEnv] = None):
        self.config = hyperparams_config()
        self.device = torch.device(
            self.config["device"]
            if torch.cuda.is_available() and self.config["cuda"]
            else "cpu"
        )

        self.env = env or QueueEnv(
            max_jobs=self.config["max_jobs"],
            max_steps=self.config["max_steps"],
        )
        self.config["max_jobs"] = self.env.max_jobs
        self.config["max_steps"] = self.env.max_steps

        self._build_agent()

        self.training_history: list[dict] = []
        self.timesteps_done = 0
        self.episodes_done = 0

    def _build_agent(self) -> None:
        self.agent = ActorCritic(
            self.env.observation_space,
            self.env.action_space,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=self.config["learning_rate"],
        )

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: float,
        terminated: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards, device=self.device)
        last_advantage = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        next_value_tensor = torch.tensor(
            next_value, dtype=torch.float32, device=self.device
        )

        for step in reversed(range(rewards.shape[0])):
            if step == rewards.shape[0] - 1:
                next_non_terminal = 0.0 if terminated else 1.0
                next_values = next_value_tensor
            else:
                next_non_terminal = 1.0
                next_values = values[step + 1]

            delta = (
                rewards[step]
                + self.config["gamma"] * next_values * next_non_terminal
                - values[step]
            )
            last_advantage = (
                delta
                + self.config["gamma"]
                * self.config["gae_lambda"]
                * next_non_terminal
                * last_advantage
            )
            advantages[step] = last_advantage

        returns = advantages + values
        return advantages, returns

    def _collect_episode(self, seed: int) -> dict:
        obs, _ = self.env.reset(seed=seed)

        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []

        total_reward = 0.0
        total_wait_time = 0.0
        total_sojourn_time = 0.0
        total_tardiness = 0.0
        total_deadline_misses = 0
        total_invalid_actions = 0
        total_dropped_jobs = 0
        steps = 0
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            obs_tensor = self._obs_tensor(obs)
            with torch.no_grad():
                action, log_prob, _, value = self.agent.get_action_and_value(obs_tensor)

            next_obs, reward, terminated, truncated, info = self.env.step(
                int(action.item())
            )

            observations.append(obs.copy())
            actions.append(int(action.item()))
            log_probs.append(float(log_prob.item()))
            rewards.append(float(reward))
            values.append(float(value.item()))

            total_reward += float(reward)
            total_wait_time += float(info["wait_time"])
            total_sojourn_time += float(info["sojourn_time"])
            total_tardiness += float(info["tardiness"])
            total_deadline_misses += int(info["deadline_miss"])
            total_invalid_actions += int(info["invalid_action"])
            total_dropped_jobs += int(info["dropped_this_step"])
            steps += 1
            obs = next_obs

        with torch.no_grad():
            if terminated:
                next_value = 0.0
            else:
                next_value = float(self.agent.get_value(self._obs_tensor(obs)).item())

        observations_tensor = torch.as_tensor(
            np.asarray(observations),
            dtype=torch.float32,
            device=self.device,
        )
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        log_probs_tensor = torch.as_tensor(
            log_probs,
            dtype=torch.float32,
            device=self.device,
        )
        rewards_tensor = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )
        values_tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

        advantages, returns = self._compute_advantages(
            rewards_tensor,
            values_tensor,
            next_value=next_value,
            terminated=terminated,
        )

        completed_jobs = int(info["completed_jobs"])
        served_jobs = max(completed_jobs, 1)

        return {
            "observations": observations_tensor,
            "actions": actions_tensor,
            "old_log_probs": log_probs_tensor,
            "advantages": advantages,
            "returns": returns,
            "values": values_tensor,
            "reward": total_reward,
            "total_wait_time": total_wait_time,
            "total_sojourn_time": total_sojourn_time,
            "total_tardiness": total_tardiness,
            "wait_time": total_wait_time / served_jobs,
            "sojourn_time": total_sojourn_time / served_jobs,
            "tardiness": total_tardiness / served_jobs,
            "deadline_misses": total_deadline_misses,
            "deadline_miss_rate": total_deadline_misses / served_jobs,
            "invalid_actions": total_invalid_actions,
            "dropped_jobs": total_dropped_jobs,
            "steps": steps,
            "completed_jobs": completed_jobs,
            "jobs_left": int(info["jobs_left"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def _update(self, rollout: dict) -> dict:
        observations = rollout["observations"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_log_probs"]
        returns = rollout["returns"]
        advantages = rollout["advantages"]

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        batch_size = observations.shape[0]
        num_minibatches = max(1, min(self.config["num_minibatches"], batch_size))
        minibatch_size = max(1, batch_size // num_minibatches)

        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        approx_kl_value = 0.0

        for _ in range(self.config["update_epochs"]):
            batch_indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_indices = batch_indices[start:end]

                _, new_log_probs, entropy, new_values = self.agent.get_action_and_value(
                    observations[minibatch_indices],
                    actions[minibatch_indices],
                )

                log_ratio = new_log_probs - old_log_probs[minibatch_indices]
                ratio = log_ratio.exp()

                unclipped_loss = -advantages[minibatch_indices] * ratio
                clipped_loss = -advantages[minibatch_indices] * torch.clamp(
                    ratio,
                    1 - self.config["clip_coef"],
                    1 + self.config["clip_coef"],
                )
                policy_loss = torch.max(unclipped_loss, clipped_loss).mean()
                value_loss = 0.5 * (
                    returns[minibatch_indices] - new_values
                ).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config["vf_coef"] * value_loss
                    - self.config["ent_coef"] * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.config["max_grad_norm"]
                )
                self.optimizer.step()

                approx_kl = ((ratio - 1.0) - log_ratio).mean()

                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())
                entropy_value = float(entropy_bonus.item())
                approx_kl_value = float(approx_kl.item())

                if approx_kl_value > self.config["kl_target"]:
                    break

            if approx_kl_value > self.config["kl_target"]:
                break

        return {
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
            "approx_kl": approx_kl_value,
        }

    def train(
        self,
        total_timesteps: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> list[dict]:
        target_timesteps = (
            self.config["total_timesteps"]
            if total_timesteps is None
            else int(total_timesteps)
        )
        base_seed = self.config["seed"] if seed is None else seed
        self._set_seed(base_seed)

        self.training_history = []
        self.timesteps_done = 0
        self.episodes_done = 0
        self.agent.train()

        progress_bar = tqdm(
            total=target_timesteps,
            disable=not show_progress,
            desc="Custom PPO",
        )

        while self.timesteps_done < target_timesteps:
            episode_seed = base_seed + self.episodes_done
            rollout = self._collect_episode(seed=episode_seed)
            update_stats = self._update(rollout)

            self.timesteps_done += rollout["steps"]
            self.episodes_done += 1
            progress_bar.update(rollout["steps"])

            history_entry = {
                "episode": self.episodes_done,
                "timesteps": self.timesteps_done,
                "reward": rollout["reward"],
                "wait_time": rollout["wait_time"],
                "sojourn_time": rollout["sojourn_time"],
                "tardiness": rollout["tardiness"],
                "deadline_miss_rate": rollout["deadline_miss_rate"],
                "invalid_action_rate": rollout["invalid_actions"]
                / max(rollout["steps"], 1),
                "dropped_jobs": rollout["dropped_jobs"],
                "steps": rollout["steps"],
                "completed_jobs": rollout["completed_jobs"],
                "terminated": rollout["terminated"],
                "truncated": rollout["truncated"],
                **update_stats,
            }
            self.training_history.append(history_entry)

        progress_bar.close()
        return self.training_history

    def evaluate(
        self,
        n_episodes: Optional[int] = None,
        *,
        base_seed: Optional[int] = None,
        deterministic: bool = True,
    ) -> dict:
        eval_episodes = self.config["episodes"] if n_episodes is None else n_episodes
        seed = self.config["seed"] if base_seed is None else base_seed
        self.agent.eval()

        episode_results = []
        for episode_idx in range(eval_episodes):
            obs, info = self.env.reset(seed=seed + episode_idx)

            total_reward = 0.0
            total_wait_time = 0.0
            total_sojourn_time = 0.0
            total_tardiness = 0.0
            deadline_misses = 0
            invalid_actions = 0
            dropped_jobs = 0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                obs_tensor = self._obs_tensor(obs)
                with torch.no_grad():
                    logits, _ = self.agent(obs_tensor)
                    distribution = Categorical(logits=logits)
                    if deterministic:
                        action = int(torch.argmax(logits, dim=-1).item())
                    else:
                        action = int(distribution.sample().item())

                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                total_wait_time += float(info["wait_time"])
                total_sojourn_time += float(info["sojourn_time"])
                total_tardiness += float(info["tardiness"])
                deadline_misses += int(info["deadline_miss"])
                invalid_actions += int(info["invalid_action"])
                dropped_jobs += int(info["dropped_this_step"])
                steps += 1

            completed_jobs = int(info["completed_jobs"])
            served_jobs = max(completed_jobs, 1)

            episode_results.append(
                {
                    "reward": total_reward,
                    "total_wait_time": total_wait_time,
                    "total_sojourn_time": total_sojourn_time,
                    "total_tardiness": total_tardiness,
                    "wait_time": total_wait_time / served_jobs,
                    "sojourn_time": total_sojourn_time / served_jobs,
                    "tardiness": total_tardiness / served_jobs,
                    "deadline_misses": deadline_misses,
                    "deadline_miss_rate": deadline_misses / served_jobs,
                    "invalid_actions": invalid_actions,
                    "invalid_action_rate": invalid_actions / max(steps, 1),
                    "dropped_jobs": dropped_jobs,
                    "steps": steps,
                    "completed_jobs": completed_jobs,
                    "jobs_left": int(info["jobs_left"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

        return {
            "name": "custom_ppo",
            "episodes": eval_episodes,
            "mean_reward": float(np.mean([ep["reward"] for ep in episode_results])),
            "mean_wait_time": float(
                np.mean([ep["wait_time"] for ep in episode_results])
            ),
            "mean_sojourn_time": float(
                np.mean([ep["sojourn_time"] for ep in episode_results])
            ),
            "mean_tardiness": float(
                np.mean([ep["tardiness"] for ep in episode_results])
            ),
            "mean_deadline_miss_rate": float(
                np.mean([ep["deadline_miss_rate"] for ep in episode_results])
            ),
            "mean_invalid_action_rate": float(
                np.mean([ep["invalid_action_rate"] for ep in episode_results])
            ),
            "mean_dropped_jobs": float(
                np.mean([ep["dropped_jobs"] for ep in episode_results])
            ),
            "mean_completed_jobs": float(
                np.mean([ep["completed_jobs"] for ep in episode_results])
            ),
            "success_rate": float(
                np.mean(
                    [
                        ep["terminated"] and not ep["truncated"]
                        for ep in episode_results
                    ]
                )
            ),
            "episode_results": episode_results,
        }

    def save(self, model_name: str = "custom_ppo_queue.pt") -> Path:
        output_dir = self.config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / model_name
        if output_path.suffix != ".pt":
            output_path = output_path.with_suffix(".pt")

        torch.save(
            {
                "model_state_dict": self.agent.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in self.config.items()
                },
                "timesteps_done": self.timesteps_done,
                "episodes_done": self.episodes_done,
                "training_history": self.training_history,
            },
            output_path,
        )
        return output_path

    def load(self, model_path: Path | str) -> None:
        checkpoint_path = Path(model_path)
        if checkpoint_path.suffix != ".pt":
            checkpoint_path = checkpoint_path.with_suffix(".pt")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        saved_config = checkpoint.get("config", {})
        if saved_config:
            self.config.update(saved_config)

        saved_max_steps = self.config.get("max_steps")
        if isinstance(saved_max_steps, str):
            normalized_max_steps = saved_max_steps.strip().lower()
            saved_max_steps = (
                None
                if normalized_max_steps in {"none", "null", ""}
                else int(normalized_max_steps)
            )
            self.config["max_steps"] = saved_max_steps

        saved_output_dir = self.config.get("output_dir")
        if isinstance(saved_output_dir, str):
            self.config["output_dir"] = Path(saved_output_dir)

        self.env = QueueEnv(
            max_jobs=int(self.config["max_jobs"]),
            max_steps=saved_max_steps,
        )
        self._build_agent()
        self.agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.timesteps_done = int(checkpoint.get("timesteps_done", 0))
        self.episodes_done = int(checkpoint.get("episodes_done", 0))
        self.training_history = list(checkpoint.get("training_history", []))


PPO = PPOAgent


if __name__ == "__main__":
    trainer = PPOAgent()
    trainer.train()
    saved_path = trainer.save()
    print(f"Saved custom PPO model to {saved_path}")
