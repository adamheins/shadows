from stable_baselines3 import PPO, SAC

# patch to include double DQN
from .dqn import DQN

ALGOS = {
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
}
