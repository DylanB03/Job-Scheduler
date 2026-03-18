from pathlib import Path
import sys

from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.base_config import hyperparams_config
from envs.queue_env import QueueEnv


class PPOTrainer:
    def __init__(self, n_envs: int = 4):
        self.config = hyperparams_config()

        self.vec_env = make_vec_env(
            lambda: QueueEnv(
                max_jobs=self.config["max_jobs"],
                max_steps=self.config["max_steps"],
            ),
            n_envs=n_envs,
            seed=self.config["seed"],
        )

        self.model = MaskablePPO(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=self.config["learning_rate"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_coef"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            n_epochs=self.config["update_epochs"],
            max_grad_norm=self.config["max_grad_norm"],
            verbose=1,
        )

    def train(self, total_timesteps: int | None = None) -> None:
        timesteps = (
            self.config["total_timesteps"]
            if total_timesteps is None
            else int(total_timesteps)
        )
        self.model.learn(total_timesteps=timesteps)

    def save(self, model_name: str = "ppo_queue") -> Path:
        output_dir = self.config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / model_name
        self.model.save(output_path)
        if output_path.suffix != ".zip":
            output_path = output_path.with_suffix(".zip")
        return output_path

    def load(self, model_path: Path | str) -> None:
        resolved_path = Path(model_path)
        if resolved_path.suffix != ".zip":
            resolved_path = resolved_path.with_suffix(".zip")
        self.model = MaskablePPO.load(resolved_path, env=self.vec_env)

    def evaluate(
        self,
        n_episodes: int,
        base_seed: int,
        deterministic: bool = True,
    ) -> dict:
        eval_env = QueueEnv(
            max_jobs=self.config["max_jobs"],
            max_steps=self.config["max_steps"],
        )

        episode_results = []
        for episode_idx in range(n_episodes):
            obs, info = eval_env.reset(seed=base_seed + episode_idx)

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
                action_masks = get_action_masks(eval_env)
                action, _ = self.model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=action_masks,
                )
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
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
            "name": "ppo",
            "episodes": n_episodes,
            "mean_reward": float(sum(ep["reward"] for ep in episode_results) / n_episodes),
            "mean_wait_time": float(
                sum(ep["wait_time"] for ep in episode_results) / n_episodes
            ),
            "mean_sojourn_time": float(
                sum(ep["sojourn_time"] for ep in episode_results) / n_episodes
            ),
            "mean_tardiness": float(
                sum(ep["tardiness"] for ep in episode_results) / n_episodes
            ),
            "mean_deadline_miss_rate": float(
                sum(ep["deadline_miss_rate"] for ep in episode_results) / n_episodes
            ),
            "mean_invalid_action_rate": float(
                sum(ep["invalid_action_rate"] for ep in episode_results) / n_episodes
            ),
            "mean_dropped_jobs": float(
                sum(ep["dropped_jobs"] for ep in episode_results) / n_episodes
            ),
            "mean_completed_jobs": float(
                sum(ep["completed_jobs"] for ep in episode_results) / n_episodes
            ),
            "success_rate": float(
                sum(
                    1.0
                    for ep in episode_results
                    if ep["terminated"] and not ep["truncated"]
                )
                / n_episodes
            ),
            "episode_results": episode_results,
        }


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()
    saved_path = trainer.save()
    print(f"Saved PPO model to {saved_path}")
