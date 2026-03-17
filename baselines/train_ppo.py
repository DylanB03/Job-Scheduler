from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

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

        self.model = PPO(
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

    def train(self) -> None:
        self.model.learn(total_timesteps=self.config["total_timesteps"])

    def save(self, model_name: str = "ppo_queue") -> Path:
        output_dir = self.config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / model_name
        self.model.save(output_path)
        return output_path


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()
    saved_path = trainer.save()
    print(f"Saved PPO model to {saved_path}")
