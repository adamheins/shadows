import pygame
import numpy as np
import gymnasium as gym

from .collision import point_poly_dist
from .entity import Action
from .taggame import TagGame

# TODO perhaps we want to have separate envs for it and not it
class TagItEnv(gym.Env):
    """Environment where the agent is 'it'."""

    def __init__(self, shape):
        pygame.init()
        self.game = TagGame(
            shape, player_it=True, invert_agent_colors=True, display=False
        )
        self.player = self.game.player
        self.enemy = self.game.enemies[0]

        # directions to move
        self.action_space = gym.spaces.Dict(
            {
                "lindir": gym.spaces.Discrete(3),
                "angdir": gym.spaces.Discrete(3),
            }
        )

        # RGB pixels
        self.observation_space = gym.spaces.Box(0, 255, (shape + (3,)), dtype=np.uint8)

    def _get_info(self):
        return {}

    def _translate_action(self, action):
        # no movement, forward, backward
        if action["lindir"] == 0:
            lindir = 0
        elif action["lindir"] == 1:
            lindir = 1
        elif action["lindir"] == 2:
            lindir = -1

        # no movement, turn right, turn left
        if action["angdir"] == 0:
            angdir = 0
        elif action["angdir"] == 1:
            angdir = 1
        elif action["angdir"] == 2:
            angdir = -1

        actions = self.game.enemy_policy.compute()
        actions[self.player.id] = Action(
            lindir=[lindir, 0],
            angdir=angdir,
            target=None,
            reload=False,
            frame=Action.LOCAL,
            lookback=False,
        )
        return actions

    def reset(self, seed=None):
        """Reset the game environment."""
        super().reset(seed=seed)

        # TODO we could reset the obstacles as well...

        # reset positions as long as not in obstacles
        for agent in self.game.agents:
            while True:
                agent.position = self.np_random.uniform(self.game.shape)
                agent.angle = self.np_random.uniform(-np.pi, np.pi)
                for obstacle in self.game.obstacles:
                    if point_poly_dist(p, obstacle) <= agent.radius:
                        continue

        # reset who is it
        for agent in self.game.agents:
            agent.it = False
        self.game.player.it = True
        self.game.it_id = self.game.player.id

        obs = self.game.draw()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.game.step(self._translate_action(action))

        # round terminates when the enemy is tagged
        d = self.player.radius + self.enemy.radius
        terminated = np.linalg.norm(self.player.position - self.enemy.position) < d
        truncated = False

        # only reward for tagging the enemy
        reward = 1 if terminated else 0

        obs = self.game.draw()
        info = self._get_info()
        return obs, reward, terminated, truncated, info
