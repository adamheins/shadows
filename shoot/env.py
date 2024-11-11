import pygame
import numpy as np
import gymnasium as gym

from .collision import point_poly_query
from .entity import Action
from .taggame import TagGame


class TagItEnv(gym.Env):
    """Environment where the agent is 'it'."""

    def __init__(self):
        pygame.init()
        self.game = TagGame(player_it=True, invert_agent_colors=True, display=False)
        self.player = self.game.player
        self.enemy = self.game.enemies[0]

        # directions to move
        # first channel is linear direction
        # second channel is angular direction
        self.action_space = gym.spaces.MultiDiscrete((3, 3))

        # RGB pixels
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3,) + self.game.shape, dtype=np.uint8
        )

    def _get_info(self):
        return {}

    def _translate_action(self, action):
        # lindir
        # no movement, forward, backward
        if action[0] == 0:
            lindir = 0
        elif action[0] == 1:
            lindir = 1
        elif action[0] == 2:
            lindir = -1

        # angdir
        # no movement, turn right, turn left
        if action[1] == 0:
            angdir = 0
        elif action[1] == 1:
            angdir = 1
        elif action[1] == 2:
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

    def reset(self, seed=None, options=None):
        """Reset the game environment."""
        super().reset(seed=seed)

        # TODO we could reset the obstacles as well...
        n = 1000

        # reset positions as long as not in obstacles
        w, h = self.game.shape
        for agent in self.game.agents:
            agent.angle = self.np_random.uniform(low=-np.pi, high=np.pi)
            # while True:
            for i in range(n):
                r = agent.radius
                agent.position = self.np_random.uniform(low=(r, r), high=(w - r, h - r))
                collision = False
                for obstacle in self.game.obstacles:
                    Q = point_poly_query(agent.position, obstacle)
                    if Q.distance <= r:
                        collision = True
                        break
                if not collision:
                    break
                if i == n - 1:
                    raise ValueError("failed to generate agent position!")

        # reset who is it
        for agent in self.game.agents:
            agent.it = False
        self.game.player.it = True
        self.game.it_id = 0

        obs = self.game.draw()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.game.step(self._translate_action(action))

        # round terminates when the enemy is tagged
        r = self.player.radius + self.enemy.radius
        d = np.linalg.norm(self.player.position - self.enemy.position)
        terminated = d < r
        truncated = False

        # big reward for tagging the enemy, small cost for being farther away
        # from it
        reward = 100 if terminated else -0.1 * d

        obs = self.game.draw()
        info = self._get_info()
        return obs, reward, terminated, truncated, info


# class TagNotItEnv(gym.Env):
#     """Environment where the agent is not 'it'."""
#
#     def __init__(self):
#         pygame.init()
#         self.game = TagGame(player_it=False, invert_agent_colors=True, display=False)
#         self.player = self.game.player
#         self.enemy = self.game.enemies[0]
#
#         # directions to move
#         # first channel is linear direction
#         # second channel is angular direction
#         # third channel is lookback
#         self.action_space = gym.spaces.MultiDiscrete((3, 3, 2))
#
#         # RGB pixels
#         self.observation_space = gym.spaces.Box(
#             0, 255, (self.game.shape + (3,)), dtype=np.uint8
#         )
#
#     def _get_info(self):
#         return {}
#
#     def _translate_action(self, action):
#         # no movement, forward, backward
#         if action["lindir"] == 0:
#             lindir = 0
#         elif action["lindir"] == 1:
#             lindir = 1
#         elif action["lindir"] == 2:
#             lindir = -1
#
#         # no movement, turn right, turn left
#         if action["angdir"] == 0:
#             angdir = 0
#         elif action["angdir"] == 1:
#             angdir = 1
#         elif action["angdir"] == 2:
#             angdir = -1
#
#         lookback = bool(action["lookback"][0])
#
#         actions = self.game.enemy_policy.compute()
#         actions[self.player.id] = Action(
#             lindir=[lindir, 0],
#             angdir=angdir,
#             target=None,
#             reload=False,
#             frame=Action.LOCAL,
#             lookback=lookback,
#         )
#         return actions
#
#     def reset(self, seed=None):
#         """Reset the game environment."""
#         super().reset(seed=seed)
#
#         # TODO we could reset the obstacles as well...
#
#         # reset positions as long as not in obstacles
#         for agent in self.game.agents:
#             while True:
#                 agent.position = self.np_random.uniform(
#                     low=(0, 0), high=self.game.shape
#                 )
#                 agent.angle = self.np_random.uniform(low=-np.pi, high=np.pi)
#                 for obstacle in self.game.obstacles:
#                     if point_poly_query(p, obstacle).distance <= agent.radius:
#                         continue
#
#         # reset who is it
#         for agent in self.game.agents:
#             agent.it = False
#         self.game.player.it = True
#         self.game.it_id = self.game.player.id
#
#         obs = self.game.draw()
#         info = self._get_info()
#         return obs, info
#
#     def step(self, action):
#         self.game.step(self._translate_action(action))
#
#         # round terminates when the enemy is tagged
#         d = self.player.radius + self.enemy.radius
#         terminated = np.linalg.norm(self.player.position - self.enemy.position) < d
#         truncated = False
#
#         # negative reward for getting tagged
#         reward = -1 if terminated else 0
#
#         obs = self.game.draw()
#         info = self._get_info()
#         return obs, reward, terminated, truncated, info
