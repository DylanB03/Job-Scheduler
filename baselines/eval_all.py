import json
from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.base_config import hyperparams_config
from baselines.heuristics import build_default_baselines, evaluate_baselines
from baselines.ppo_agent import PPOAgent
from baselines.train_ppo import PPOTrainer
from envs.queue_env import QueueEnv


def print_summary_table(results: list[dict]) -> None:
    headers = [
        "method",
        "reward",
        "wait",
        "tardiness",
        "miss_rate",
        "drops",
        "success",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result["name"],
                f'{result["mean_reward"]:.2f}',
                f'{result["mean_wait_time"]:.2f}',
                f'{result["mean_tardiness"]:.2f}',
                f'{result["mean_deadline_miss_rate"]:.2f}',
                f'{result["mean_dropped_jobs"]:.2f}',
                f'{result["success_rate"]:.2f}',
            ]
        )

    widths = [
        max(len(header), max(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "  ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    )
    print(header_line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def save_results(results: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "benchmark_results.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    return output_path


def plot_results(results: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [result["name"] for result in results]
    metrics = [
        ("mean_reward", "Mean Reward"),
        ("mean_wait_time", "Avg Wait Time"),
        ("mean_sojourn_time", "Avg Sojourn Time"),
        ("mean_tardiness", "Avg Tardiness"),
        ("mean_deadline_miss_rate", "Deadline Miss Rate"),
        ("mean_dropped_jobs", "Dropped Jobs"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for axis, (metric_key, metric_label) in zip(axes, metrics):
        values = [result[metric_key] for result in results]
        axis.bar(labels, values, color="#4C78A8")
        axis.set_title(metric_label)
        axis.tick_params(axis="x", rotation=25)

    for axis in axes[len(metrics) :]:
        axis.axis("off")

    fig.suptitle("Queue Benchmark Comparison", fontsize=14)
    fig.tight_layout()

    output_path = out_dir / "benchmark_results.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _resolve_existing_model_path(out_dir: Path) -> Optional[Path]:
    candidate_names = [
        "ppo_queue_notebook.zip",
        "ppo_queue_eval.zip",
        "ppo_queue.zip",
    ]
    for candidate_name in candidate_names:
        candidate_path = out_dir / candidate_name
        if candidate_path.exists():
            return candidate_path
    return None


def _resolve_existing_custom_model_path(out_dir: Path) -> Optional[Path]:
    candidate_names = [
        "custom_ppo_notebook.pt",
        "custom_ppo_eval.pt",
        "custom_ppo_queue.pt",
    ]
    for candidate_name in candidate_names:
        candidate_path = out_dir / candidate_name
        if candidate_path.exists():
            return candidate_path
    return None


def main(
    train_ppo: bool = False,
    model_path: Optional[str | Path] = None,
    *,
    train_custom_ppo: bool = False,
    custom_model_path: Optional[str | Path] = None,
) -> None:
    args = hyperparams_config()

    env = QueueEnv(max_jobs=args["max_jobs"], max_steps=args["max_steps"])
    baselines = build_default_baselines()
    baseline_results = evaluate_baselines(
        baselines,
        env,
        n_episodes=args["episodes"],
        base_seed=args["seed"],
    )

    results = list(baseline_results)

    if train_ppo:
        trainer = PPOTrainer()
        trainer.train()
        saved_model_path = trainer.save("ppo_queue_eval")
        print(f"Trained and saved PPO model to {saved_model_path}")
        ppo_result = trainer.evaluate(
            n_episodes=args["episodes"],
            base_seed=args["seed"],
            deterministic=True,
        )
        results.append(ppo_result)
    else:
        explicit_model_path = Path(model_path) if model_path is not None else None
        resolved_model_path = explicit_model_path or _resolve_existing_model_path(
            args["output_dir"]
        )
        if resolved_model_path is not None and resolved_model_path.exists():
            trainer = PPOTrainer()
            trainer.load(resolved_model_path)
            print(f"Loaded PPO model from {resolved_model_path}")
            ppo_result = trainer.evaluate(
                n_episodes=args["episodes"],
                base_seed=args["seed"],
                deterministic=True,
            )
            results.append(ppo_result)
        else:
            print("No saved PPO model found; running heuristic-only benchmark.")

    if train_custom_ppo:
        custom_trainer = PPOAgent()
        custom_trainer.train()
        saved_custom_path = custom_trainer.save("custom_ppo_eval")
        print(f"Trained and saved custom PPO model to {saved_custom_path}")
        custom_result = custom_trainer.evaluate(
            n_episodes=args["episodes"],
            base_seed=args["seed"],
            deterministic=True,
        )
        results.append(custom_result)
    else:
        explicit_custom_path = (
            Path(custom_model_path) if custom_model_path is not None else None
        )
        resolved_custom_path = explicit_custom_path or _resolve_existing_custom_model_path(
            args["output_dir"]
        )
        if resolved_custom_path is not None and resolved_custom_path.exists():
            custom_trainer = PPOAgent()
            custom_trainer.load(resolved_custom_path)
            print(f"Loaded custom PPO model from {resolved_custom_path}")
            custom_result = custom_trainer.evaluate(
                n_episodes=args["episodes"],
                base_seed=args["seed"],
                deterministic=True,
            )
            results.append(custom_result)
        else:
            print("No saved custom PPO model found; skipping custom PPO benchmark.")

    print_summary_table(results)

    json_path = save_results(results, args["output_dir"])
    plot_path = plot_results(results, args["output_dir"])

    print()
    print(f"Saved benchmark data to {json_path}")
    print(f"Saved benchmark plot to {plot_path}")


if __name__ == "__main__":
    main(train_ppo=False)
