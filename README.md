RL project using a custom Gymnasium queueing environment, heuristic baselines, and PPO.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Run scripts with:

```bash
uv run python path/to/script.py
```

## Project Layout

- `envs/queue_env.py`: single-server queueing environment
- `baselines/heuristics.py`: FIFO, SPT, EDF, priority, and random baselines
- `baselines/train_ppo.py`: Stable-Baselines3 PPO trainer
- `baselines/ppo_agent.py`: custom PyTorch PPO implementation
- `baselines/eval_all.py`: combined benchmark runner and plot generation
- `remote_train/collab.ipynb`: notebook for local smoke tests, training, and evaluation
- `remote_train/modal.py`: optional Modal remote-training entrypoint

## Local Training

Train the SB3 PPO baseline:

```bash
uv run python baselines/train_ppo.py
```

Train the custom PPO agent:

```bash
uv run python baselines/ppo_agent.py
```

## Benchmarking

Benchmark heuristics and evaluate any saved PPO models:

```bash
uv run python baselines/eval_all.py
```

The benchmark writes:

- `results/benchmark_results.json`
- `results/benchmark_results.png`

Reported metrics include reward, average wait time, average sojourn time,
average tardiness, deadline-miss rate, dropped jobs, and success rate.

## Notebook Flow

Open [remote_train/collab.ipynb](/home/dylan/projects/rlQueueHandler/remote_train/collab.ipynb) and run:

1. optional heuristic smoke test
2. SB3 PPO training
3. optional custom PPO training
4. combined benchmark evaluation against any saved PPO models

## References

- Gymnasium custom environment guide: https://gymnasium.farama.org/introduction/create_custom_env/
- Stable-Baselines3 PPO docs: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#
