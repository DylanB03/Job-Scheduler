from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env


class QueueEnv(gym.Env):
    
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_jobs: int = 10,
        max_steps: Optional[int] = None,
        total_jobs: Optional[int] = None,
    ):
        
        self.max_jobs = max_jobs
        self.total_jobs = total_jobs if total_jobs is not None else max_jobs * 3
        self.max_steps = max_steps
        self.num_job_features = 4

        self.current_time = 0.0
        self.steps_taken = 0
        self.completed_jobs = 0
        self.invalid_actions = 0
        self.dropped_jobs = 0
        self.queue_length = 0
        self.next_job_idx = 0
        self.max_steps_limit = 0

        # Each job row is:
        # [processing_time, priority, deadline, arrival_time]
        self.jobs = np.zeros((self.total_jobs, self.num_job_features), dtype=np.float32)

        # The queue is the agent-facing state: zero-padded waiting jobs only.
        self.queue = np.zeros((self.max_jobs, self.num_job_features), dtype=np.float32)

        # current time + queue size + zero-padded queue snapshot
        obs_size = 2 + self.max_jobs * self.num_job_features
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Action i means "serve the job currently stored in queue slot i".
        self.action_space = gym.spaces.Discrete(self.max_jobs)

    def _sample_jobs(self, num_jobs: int) -> np.ndarray:
        
        processing_times = self.np_random.integers(
            low=1, high=10, size=num_jobs
        ).astype(np.float32)
        
        priorities = self.np_random.integers(
            low=1, high=4, size=num_jobs
        ).astype(np.float32)

        interarrival_times = self.np_random.integers(
            low=0, high=4, size=num_jobs
        ).astype(np.float32)
        interarrival_times[0] = 0.0
        #arrival times since the first jobs arrival time -> accumulate / time
        arrival_times = np.cumsum(interarrival_times)

        # time until a job is considered late, consider arrival + processing time + buffer for prev jobs
        deadline_slack = self.np_random.integers(
            low=2, high=10, size=num_jobs
        ).astype(np.float32)
        deadlines = arrival_times + processing_times + deadline_slack

        return np.stack(
            (processing_times, priorities, deadlines, arrival_times), axis=1
        )

    def _get_obs(self) -> np.ndarray:
        
        return np.concatenate(
            (
                np.array(
                    [self.current_time, float(self.queue_length)],
                    dtype=np.float32,
                ),
                self.queue.flatten().astype(np.float32),
            )
        )

    def _get_info(
        self,
        *,
        invalid_action: bool = False,
        wait_time: float = 0.0,
        tardiness: float = 0.0,
        dropped_this_step: int = 0,
    ) -> dict:
        
        jobs_left = (len(self.jobs) - self.next_job_idx) + self.queue_length
        return {
            "invalid_action": invalid_action,
            "wait_time": float(wait_time),
            "tardiness": float(tardiness),
            "queue_length": int(self.queue_length),
            "completed_jobs": int(self.completed_jobs),
            "jobs_left": int(jobs_left),
            "current_time": float(self.current_time),
            "invalid_action_count": int(self.invalid_actions),
            "dropped_jobs": int(self.dropped_jobs),
            "dropped_this_step": int(dropped_this_step),
        }

    def _clear_queue_slot(self, slot_idx: int) -> None:
        
        if slot_idx < self.queue_length - 1:
            self.queue[slot_idx : self.queue_length - 1] = self.queue[
                slot_idx + 1 : self.queue_length
            ]
        self.queue[self.queue_length - 1] = 0.0
        self.queue_length -= 1

    def _admit_arrivals(self, up_to_time: float) -> int:
        
        dropped_this_step = 0
        total_jobs = len(self.jobs)

        while (
            self.next_job_idx < total_jobs
            and self.jobs[self.next_job_idx, 3] <= up_to_time
        ):
            job = self.jobs[self.next_job_idx]
            if self.queue_length < self.max_jobs:
                self.queue[self.queue_length] = job
                self.queue_length += 1
            else:
                self.dropped_jobs += 1
                dropped_this_step += 1

            self.next_job_idx += 1

        return dropped_this_step

    def _advance_to_next_arrival(self) -> int:
        if self.queue_length > 0 or self.next_job_idx >= len(self.jobs):
            return 0

        self.current_time = float(self.jobs[self.next_job_idx, 3])
        return self._admit_arrivals(self.current_time)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_time = 0.0
        self.steps_taken = 0
        self.completed_jobs = 0
        self.invalid_actions = 0
        self.dropped_jobs = 0
        self.queue_length = 0
        self.next_job_idx = 0
        self.queue.fill(0.0)

        raw_jobs = None
        if options is not None:
            raw_jobs = options.get("jobs") if isinstance(options, dict) else options

        if raw_jobs is not None:
            jobs = np.asarray(raw_jobs, dtype=np.float32)
            if jobs.ndim != 2 or jobs.shape[1] != self.num_job_features:
                raise ValueError(
                    "Expected jobs with shape (n_jobs, 4) for reset options."
                )
            order = np.argsort(jobs[:, 3], kind="stable")
            self.jobs = jobs[order].copy()
        else:
            self.jobs = self._sample_jobs(self.total_jobs)

        self.max_steps_limit = (
            self.max_steps if self.max_steps is not None else max(len(self.jobs) * 2, 1)
        )

        self._admit_arrivals(self.current_time)
        self._advance_to_next_arrival()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.max_jobs:
            raise ValueError(f"Action must be in [0, {self.max_jobs - 1}], got {action}")

        self.steps_taken += 1
        dropped_this_step = self._advance_to_next_arrival()

        terminated = bool(self.next_job_idx >= len(self.jobs) and self.queue_length == 0)
        if terminated:
            return (
                self._get_obs(),
                0.0,
                True,
                False,
                self._get_info(dropped_this_step=dropped_this_step),
            )

        if action >= self.queue_length:
            self.invalid_actions += 1
            reward = -2.0
            terminated = bool(self.next_job_idx >= len(self.jobs) and self.queue_length == 0)
            truncated = bool(self.steps_taken >= self.max_steps_limit and not terminated)
            return (
                self._get_obs(),
                reward,
                terminated,
                truncated,
                self._get_info(
                    invalid_action=True,
                    dropped_this_step=dropped_this_step,
                ),
            )

        processing_time, priority, deadline, arrival_time = self.queue[action].copy()
        wait_time = max(0.0, self.current_time - float(arrival_time))

        self._clear_queue_slot(action)

        service_end_time = self.current_time + float(processing_time)
        self.current_time = service_end_time
        dropped_this_step += self._admit_arrivals(service_end_time)

        tardiness = max(0.0, service_end_time - float(deadline))
        reward = (
            2.0 * float(priority)
            - 0.1 * float(processing_time)
            - 0.2 * wait_time
            - tardiness
            - 2.0 * dropped_this_step
        )

        self.completed_jobs += 1

        terminated = bool(self.next_job_idx >= len(self.jobs) and self.queue_length == 0)
        truncated = bool(self.steps_taken >= self.max_steps_limit and not terminated)

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(
                invalid_action=False,
                wait_time=wait_time,
                tardiness=tardiness,
                dropped_this_step=dropped_this_step,
            ),
        )


if __name__ == "__main__":
    env = QueueEnv()
    check_env(env)
