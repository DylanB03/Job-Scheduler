from pathlib import Path

import modal


MODELS_DIR = Path("/models")
checkpoints_volume = modal.Volume.from_name(
    "ppo-checkpoints", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "gymnasium",
        "stable-baselines3",
        "numpy",
        "matplotlib",
        "tqdm",
    )
    .add_local_python_source("baselines", "config", "envs")
)

app = modal.App(name="ppo-queue", image=image)


@app.function(
    image=image,
    gpu="T4",
    volumes={str(MODELS_DIR): checkpoints_volume},
    timeout=86_400,
)
def train_remote(
    model_name: str = "ppo_queue_modal",
    total_timesteps: int | None = None,
) -> str:
    from baselines.train_ppo import PPOTrainer

    trainer = PPOTrainer()
    trainer.config["output_dir"] = MODELS_DIR

    if total_timesteps is not None:
        trainer.config["total_timesteps"] = int(total_timesteps)

    trainer.train()
    saved_path = trainer.save(model_name)
    checkpoints_volume.commit()
    return str(saved_path)


@app.local_entrypoint()
def main(
    model_name: str = "ppo_queue_modal",
    total_timesteps: int = 0,
) -> None:
    timesteps_override = None if total_timesteps <= 0 else total_timesteps
    saved_path = train_remote.remote(
        model_name=model_name,
        total_timesteps=timesteps_override,
    )
    print(f"Saved PPO model to {saved_path}")
