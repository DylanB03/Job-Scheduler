import gymnasium as gym
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
from torch.utils.data import DataLoader, TensorDataset

class ActorCritic(nn.Module):
    
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        obs_dim = int(np.array(obs_space.shape).prod())
        action_dim = action_space.shape[0]
        
        # Conv2d : input channels, output channels, kernel size, convoultion stride
        self.head = nn.Sequential(
            nn.Conv2d(
                4, 16, 8, stride = 4
            ), nn.Tanh(),
            nn.Conv2d(
                16, 32, 4, stride = 2
            ), nn.Tanh(),
            nn.Flatten(), nn.Linear(2592, 256), nn.Tanh()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(256, nb_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(256,1)
        )
        
    def forward(self, x):
        
    