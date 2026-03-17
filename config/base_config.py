from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class HyperParamsConfig:
    episodes: int
    max_jobs: int
    max_steps: Optional[int]
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    update_epochs: int
    num_minibatches: int
    max_grad_norm: float
    kl_target: float
    seed: int
    output_dir: Path


def _parse_optional_int(raw_value: str) -> Optional[int]:
    normalized = raw_value.strip().lower()
    if normalized in {"none", "null", ""}:
        return None
    return int(raw_value)


def hyperparams_config(config_path: Optional[Path] = None) -> HyperParamsConfig:
    cfg_path = config_path or (Path(__file__).resolve().parent / "config.ini")

    parser = ConfigParser()
    if not parser.read(cfg_path):
        raise FileNotFoundError(f"Could not read config file at {cfg_path}")

    max_steps = _parse_optional_int(parser.get("training", "max_steps", fallback="null"))

    return HyperParamsConfig(
        # training
        episodes=parser.getint("training", "episodes"),
        max_jobs=parser.getint("training", "max_jobs"),
        max_steps=max_steps,
        learning_rate=parser.getfloat("training", "learning_rate"),
        # ppo
        gamma=parser.getfloat("ppo", "gamma", fallback=0.99),
        gae_lambda=parser.getfloat("ppo", "gae_lambda", fallback=0.95),
        clip_coef=parser.getfloat("ppo", "clip_coef", fallback=0.2),
        ent_coef=parser.getfloat("ppo", "ent_coef", fallback=0.01),
        vf_coef=parser.getfloat("ppo", "vf_coef", fallback=0.5),
        update_epochs=parser.getint("ppo", "update_epochs", fallback=10),
        num_minibatches=parser.getint("ppo", "num_minibatches", fallback=16),
        max_grad_norm=parser.getfloat("ppo", "max_grad_norm", fallback=0.5),
        kl_target=parser.getfloat("ppo", "kl_target", fallback=0.015),
        # system
        seed=parser.getint("system", "seed", fallback=0),
        output_dir=Path(parser.get("system", "output_dir", fallback="results")),
    )
