import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.queue_env import QueueEnv
from config.base_config import hyperparams_config, HyperParamsConfig

class PPOTrainer:
    
    def __init__(self):
        
        config = hyperparams_config()
        
        #vectorized parallel environments
        vec_env = make_vec_env(QueueEnv(
            max_jobs= config.max_jobs,
            max_steps=config.max_steps | None
        ),
        n_envs=4
        )
        
        
        self.model = PPO(
            #input is a vector , CNN is intended for audio/visual / CV , multi input is both
            policy = "MlpPolicy",
            env = QueueEnv,
            
            
        )