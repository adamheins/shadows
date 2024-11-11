from .math import *
from .collision import *
from .gui import Text, Color
from .entity import Agent, Action, Projectile
from .obstacle import Obstacle
from .taggame import TagGame, TagAIPolicy
from .env import TagItEnv

import gymnasium as gym

gym.register(
    id="TagIt-v0",
    entry_point=TagItEnv,
)
