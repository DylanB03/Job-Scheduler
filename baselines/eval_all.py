import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.base_config import hyperparams_config
from baselines.heuristics import build_default_baselines, evaluate_baselines
from envs.queue_env import QueueEnv


def print_summary_table(results: list[dict]) -> None:
    headers = [
        "baseline",
        "reward",
        "wait",
        "tardiness",
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
    output_path = out_dir / "heuristic_benchmark.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    return output_path


def plot_results(results: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [result["name"] for result in results]
    metrics = [
        ("mean_reward", "Mean Reward"),
        ("mean_wait_time", "Mean Wait Time"),
        ("mean_tardiness", "Mean Tardiness"),
        ("mean_dropped_jobs", "Mean Dropped Jobs"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for axis, (metric_key, metric_label) in zip(axes, metrics):
        values = [result[metric_key] for result in results]
        axis.bar(labels, values, color="#4C78A8")
        axis.set_title(metric_label)
        axis.tick_params(axis="x", rotation=25)

    fig.suptitle("Queue Heuristic Benchmark", fontsize=14)
    fig.tight_layout()

    output_path = out_dir / "heuristic_benchmark.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = hyperparams_config()

    env = QueueEnv(max_jobs=args["max_jobs"], max_steps=args["max_steps"])
    baselines = build_default_baselines()
    results = evaluate_baselines(
        baselines,
        env,
        n_episodes=args["episodes"],
        base_seed=args["seed"],
    )

    print_summary_table(results)

    json_path = save_results(results, args["output_dir"])
    plot_path = plot_results(results, args["output_dir"])

    print()
    print(f"Saved benchmark data to {json_path}")
    print(f"Saved benchmark plot to {plot_path}")


if __name__ == "__main__":
    main()
