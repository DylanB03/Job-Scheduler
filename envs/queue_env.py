from typing import Optional

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import numpy as np


class QueueEnv(gym.Env):

    def __init__(self, max_jobs: int = 10, max_steps: Optional[int] = None):
        self.max_jobs = max_jobs
        self.max_steps = max_steps if max_steps is not None else max_jobs * 2
        self.num_job_features = 4

        self.current_time = 0.0
        self.steps_taken = 0
        self.completed_jobs = 0
        self.invalid_actions = 0

        # Each row stores one job as:
        # [processing_time, priority, deadline, arrival_time]
        #actual values for each job
        self.jobs = np.zeros((self.max_jobs, self.num_job_features), dtype=np.float32)
        #active mask shows which jobs have been queued
        #ex. 3 jobs, job #2 is complete at index 1, active = [1,0,1]
        self.active_mask = np.zeros(self.max_jobs, dtype=np.int8)

        # current time + jobs left + (max jobs * num features)
        # flattened array of jobs
        obs_size = 2 + self.max_jobs * self.num_job_features
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Action i means "process the job currently stored in slot i".
        # discrete is from 0 - n -1 on samples
        self.action_space = gym.spaces.Discrete(self.max_jobs)

    # randomize the jobs and their features
    def _sample_jobs(self) -> np.ndarray:
        processing_times = self.np_random.integers(
            low=1, high=10, size=self.max_jobs
        ).astype(np.float32)
        priorities = self.np_random.integers(
            low=1, high=4, size=self.max_jobs
        ).astype(np.float32)
        arrival_times = np.zeros(self.max_jobs, dtype=np.float32)
        deadline_slack = self.np_random.integers(
            low=2, high=10, size=self.max_jobs
        ).astype(np.float32)
        deadlines = processing_times + deadline_slack


        #assemble along axis 1 s.t. format matches jobs expectation in init
        return np.stack(
            (processing_times, priorities, deadlines, arrival_times), axis=1
        )

    #return n direction array as max jobs and features is variable fixed
    def _get_obs(self) -> np.ndarray:
        visible_jobs = self.jobs.copy()
        visible_jobs[self.active_mask == 0] = 0.0

        #see observation space description for concatenation desc. in __init__
        return np.concatenate(
            (
                np.array(
                    [self.current_time, float(self.active_mask.sum())],
                    dtype=np.float32,
                ),
                visible_jobs.flatten().astype(np.float32),
            )
        )

    # not used by learning algorithm, relevant for extra insight and debug
    def _get_info(
        self,
        *,
        invalid_action: bool = False,
        wait_time: float = 0.0,
        tardiness: float = 0.0,
    ) -> dict:
        return {
            "invalid_action": invalid_action,
            "wait_time": float(wait_time),
            "tardiness": float(tardiness),
            "completed_jobs": int(self.completed_jobs),
            "jobs_left": int(self.active_mask.sum()),
            "current_time": float(self.current_time),
            "invalid_action_count": int(self.invalid_actions),
        }

    #starts a new epside
    def reset(self, *, seed: Optional[int] = None, options: Optional[list] = None):
        super().reset(seed=seed)

        self.current_time = 0.0
        self.steps_taken = 0
        self.completed_jobs = 0
        self.invalid_actions = 0

        # expect options as a python 2D array representing the jobs
        if options is not None:
            jobs = np.asarray(options, dtype=np.float32)
            expected_shape = (self.max_jobs, self.num_job_features)
            if jobs.shape != expected_shape:
                raise ValueError(
                    f"Expected jobs with shape {expected_shape}, got {jobs.shape}"
                )
            self.jobs = jobs.copy()
        else:
            # pull random values for all
            self.jobs = self._sample_jobs()

    
        #all jobs active
        self.active_mask = np.ones(self.max_jobs, dtype=np.int8)

        return self._get_obs(), self._get_info()

    # take action, update env state, return result
    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.max_jobs:
            raise ValueError(f"Action must be in [0, {self.max_jobs - 1}], got {action}")

        self.steps_taken += 1
        
        #result format: obvs, reward, terminated, truncated, info
        #terminated == if all jobs are complete naturally
        #truncated == unnatural episode end ; max steps , time limit
        
        if self.active_mask[action] == 0:
            self.invalid_actions += 1
            reward = -2.0
            terminated = bool(self.active_mask.sum() == 0)
            truncated = bool(self.steps_taken >= self.max_steps and not terminated)
            return (
                self._get_obs(),
                reward,
                terminated,
                truncated,
                self._get_info(invalid_action=True),
            )

        processing_time, priority, deadline, arrival_time = self.jobs[action]

        start_time = self.current_time
        self.current_time += float(processing_time)

        wait_time = max(0.0, start_time - float(arrival_time))
        tardiness = max(0.0, self.current_time - float(deadline))
        #intermediate feedback, so every queue is assigned a reward
        reward = 2.0 * float(priority) - 0.1 * float(processing_time) - tardiness

        self.active_mask[action] = 0
        self.completed_jobs += 1

        terminated = bool(self.active_mask.sum() == 0)
        truncated = bool(self.steps_taken >= self.max_steps and not terminated)

    
        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(
                invalid_action=False,
                wait_time=wait_time,
                tardiness=tardiness,
            ),
        )


if __name__ == "__main__":
    #verify against SB3, warnings are outputed if they exist
    env = QueueEnv()
    check_env(env)