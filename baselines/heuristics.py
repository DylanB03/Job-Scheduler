from abc import ABC, abstractmethod
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np

from config.base_config import hyperparams_config


def _resolve_eval_defaults(
    n_episodes: Optional[int], base_seed: Optional[int]
) -> tuple[int, int]:
    config = hyperparams_config()
    resolved_episodes = config["episodes"] if n_episodes is None else n_episodes
    resolved_seed = config["seed"] if base_seed is None else base_seed
    return resolved_episodes, resolved_seed


# all baselines return the discrete index int as described in the action space
class HeuristicBaseline(ABC):
    name = "heuristic"

    def _active_indices(self, env: gym.Env) -> np.ndarray:
        return np.arange(env.queue_length, dtype=np.int32)

    @abstractmethod
    def select_action(self, env: gym.Env, obs: np.ndarray) -> int:
        """Choose the next job slot to process."""

    def run_episode(
        self,
        env: gym.Env,
        *,
        seed: Optional[int] = None,
        jobs: Optional[np.ndarray] = None,
    ) -> dict:
        
        if jobs is not None:
            obs, info = env.reset(seed=seed, options={"jobs": jobs})
        else:
            obs, info = env.reset(seed=seed)

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
            action = self.select_action(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)

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

        return {
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

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: Optional[int] = None,
        *,
        base_seed: Optional[int] = None,
        job_sets: Optional[Sequence[np.ndarray]] = None,
    ) -> dict:
        n_episodes, base_seed = _resolve_eval_defaults(n_episodes, base_seed)
        
        if job_sets is not None and len(job_sets) < n_episodes:
            raise ValueError("job_sets must contain at least n_episodes items")

        episode_results = []
        for episode_idx in range(n_episodes):
            jobs = None if job_sets is None else job_sets[episode_idx]
            seed = base_seed + episode_idx
            episode_results.append(self.run_episode(env, seed=seed, jobs=jobs))

        return {
            "name": self.name,
            "episodes": n_episodes,
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


class RandomBaseline(HeuristicBaseline):
    name = "random"

    def select_action(self, env: gym.Env, obs: np.ndarray) -> int:
        active_indices = self._active_indices(env)
        if active_indices.size == 0:
            return 0
        return int(env.np_random.choice(active_indices))


class FifoBaseline(HeuristicBaseline):
    name = "fifo"

    def select_action(self, env: gym.Env, obs: np.ndarray) -> int:
        active_indices = self._active_indices(env)
        if active_indices.size == 0:
            return 0

        arrival_times = env.queue[active_indices, 3]
        order = np.lexsort((active_indices, arrival_times))
        return int(active_indices[order[0]])


class TimeBaseline(HeuristicBaseline):
    name = "shortest_processing_time"

    def select_action(self, env: gym.Env, obs: np.ndarray) -> int:
        active_indices = self._active_indices(env)
        if active_indices.size == 0:
            return 0

        processing_times = env.queue[active_indices, 0]
        order = np.lexsort((active_indices, processing_times))
        return int(active_indices[order[0]])


class DeadlineBaseline(HeuristicBaseline):
    name = "earliest_deadline"

    def select_action(self, env: gym.Env, obs: np.ndarray) -> int:
        active_indices = self._active_indices(env)
        if active_indices.size == 0:
            return 0

        deadlines = env.queue[active_indices, 2]
        order = np.lexsort((active_indices, deadlines))
        return int(active_indices[order[0]])


class PriorityBaseline(HeuristicBaseline):
    name = "highest_priority"

    def select_action(self, env: gym.Env, obs: np.ndarray) -> int:
        active_indices = self._active_indices(env)
        if active_indices.size == 0:
            return 0

        priorities = env.queue[active_indices, 1]
        order = np.lexsort((active_indices, -priorities))
        return int(active_indices[order[0]])


def build_default_baselines() -> list[HeuristicBaseline]:
    return [
        RandomBaseline(),
        FifoBaseline(),
        TimeBaseline(),
        DeadlineBaseline(),
        PriorityBaseline(),
    ]


def evaluate_baselines(
    baselines: Sequence[HeuristicBaseline],
    env: gym.Env,
    n_episodes: Optional[int] = None,
    *,
    base_seed: Optional[int] = None,
    job_sets: Optional[Sequence[np.ndarray]] = None,
) -> list[dict]:
    n_episodes, base_seed = _resolve_eval_defaults(n_episodes, base_seed)
    
    return [
        baseline.evaluate(
            env,
            n_episodes=n_episodes,
            base_seed=base_seed,
            job_sets=job_sets,
        )
        for baseline in baselines
    ]
