import gymnasium as gym

#necessary for PPO to interact with custon environments through SB3
class QueueWrapper(gym.Wrapper):
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = gym.spaces.Discrete(env.action_space.n, start = 0)
        