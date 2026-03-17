from configparser import ConfigParser
from pathlib import Path


def hyperparams_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.ini"
    project_root = cfg_path.parent.parent

    parser = ConfigParser()
    if not parser.read(cfg_path):
        raise FileNotFoundError(f"Could not read config file at {cfg_path}")

    raw_max_steps = parser.get("training", "max_steps", fallback="null").strip().lower()
    max_steps = None if raw_max_steps in {"none", "null", ""} else int(raw_max_steps)

    raw_output_dir = Path(parser.get("system", "output_dir", fallback="results"))
    output_dir = (
        raw_output_dir
        if raw_output_dir.is_absolute()
        else project_root / raw_output_dir
    )

    return {
        # training
        "total_timesteps" : parser.getint("training", "total_timesteps"),
        "episodes": parser.getint("training", "episodes"),
        "max_jobs": parser.getint("training", "max_jobs"),
        "max_steps": max_steps,
        "learning_rate": parser.getfloat("training", "learning_rate"),
        # ppo
        "gamma": parser.getfloat("ppo", "gamma", fallback=0.99),
        "gae_lambda": parser.getfloat("ppo", "gae_lambda", fallback=0.95),
        "clip_coef": parser.getfloat("ppo", "clip_coef", fallback=0.2),
        "ent_coef": parser.getfloat("ppo", "ent_coef", fallback=0.01),
        "vf_coef": parser.getfloat("ppo", "vf_coef", fallback=0.5),
        "update_epochs": parser.getint("ppo", "update_epochs", fallback=10),
        "num_minibatches": parser.getint("ppo", "num_minibatches", fallback=4),
        "max_grad_norm": parser.getfloat("ppo", "max_grad_norm", fallback=0.5),
        "kl_target": parser.getfloat("ppo", "kl_target", fallback=0.015),
        # system
        "device" : parser.get("system", "device", fallback = "cuda"),
        "cuda" : parser.getboolean("system", "cuda", fallback = False),
        "seed": parser.getint("system", "seed", fallback=0),
        "output_dir": output_dir,
    }
