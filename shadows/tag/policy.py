import gymnasium as gym
import pygame
import numpy as np
from collections import deque

from ..math import *
from ..entity import Action
from ..gui import Color


class ImageObserver:
    def __init__(self, screen, agent, n_stack=1):
        self.screen = screen
        self.agent = agent

        self.n_stack = n_stack

        # initialize the stack of observations
        obs = self._get_single_observation()
        self._past_obs = deque([obs] * n_stack, maxlen=n_stack)

    def space(self, shape, grayscale=True):
        if grayscale:
            img_shape = shape + (1,)
        else:
            img_shape = shape + (3,)

        return gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=np.zeros(2, dtype=np.float32),
                    high=np.array(shape, dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "angle": gym.spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
                "image": gym.spaces.Box(
                    low=0, high=255, shape=img_shape, dtype=np.uint8
                ),
            }
        )

    def _get_rgb(self):
        """Get RGB pixel values from the given screen."""
        return np.array(pygame.surfarray.pixels3d(self.screen), dtype=np.uint8)

    def _get_single_observation(self):
        """Get a single observation."""
        rgb = self._get_rgb()
        shape = rgb.shape[:2] + (1,)

        # TODO need to confirm the colors are right
        gray = np.zeros(shape, dtype=np.uint8)

        enemy_mask = np.all(rgb == Color.ENEMY, axis=-1)
        gray[enemy_mask, 0] = 85

        player_mask = np.all(rgb == Color.PLAYER, axis=-1)
        gray[player_mask, 0] = 170

        obs_mask = np.all(rgb == Color.OBSTACLE, axis=-1)
        gray[obs_mask, 0] = 255

        return {
            "position": self.agent.position.astype(np.float32),
            "angle": np.array([self.agent.angle], dtype=np.float32),
            "image": gray,
        }

    def get_observation(self):
        """Get an observation for input to the model, which may be stacked."""
        obs = self._get_single_observation()
        if self.n_stack == 1:
            return obs

        # add the new observation
        self._past_obs.append(obs)

        stack = {}
        for key in obs.keys():
            stack[key] = np.concatenate([obs[key] for obs in self._past_obs], axis=-1)
        return stack

        # concatenate the queue to produce the stacked observation
        # position = np.concatenate([obs["position"] for obs in self._past_obs])
        # angle = np.concatenate([obs["angle"] for obs in self._past_obs])
        # image = np.concatenate([obs["image"] for obs in self._past_obs], axis=-1)
        # return {
        #     "position": position,
        #     "angle": angle,
        #     "image": image,
        # }


class FullStateObserver:
    def __init__(self, agent, enemy, treasures=None, n_stack=1):
        self.agent = agent
        self.enemy = enemy
        self.n_stack = n_stack

        if treasures is None:
            treasures = []
        self.treasures = treasures

    def space(self, shape):
        space = {
            "agent_position": gym.spaces.Box(
                low=np.zeros(2, dtype=np.float32),
                high=np.array(shape, dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            ),
            "agent_angle": gym.spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
            "enemy_position": gym.spaces.Box(
                low=np.zeros(2, dtype=np.float32),
                high=np.array(shape, dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            ),
            # "enemy_angle": gym.spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
        }

        # add treasures
        n = len(self.treasures)
        if n > 0:
            low = np.zeros(2 * n, dtype=np.float32)
            high = np.tile(shape, n).astype(np.float32)
            space["treasure_positions"] = gym.spaces.Box(
                low=low, high=high, shape=(2 * n,), dtype=np.float32
            )

        return gym.spaces.Dict(space)

    def get_observation(self):
        obs = {
            "agent_position": self.agent.position.astype(np.float32),
            "agent_angle": np.array([self.agent.angle], dtype=np.float32),
            "enemy_position": self.enemy.position.astype(np.float32),
            # "enemy_angle": np.array([self.enemy.angle], dtype=np.float32),
        }
        if len(self.treasures) > 0:
            obs["treasure_positions"] = np.concatenate(
                [t.center for t in self.treasures]
            ).astype(np.float32)
        return obs


class TagAIPolicy:
    """Basic AI policy for the tag game."""

    def __init__(
        self,
        screen,
        agent,
        player,
        obstacles,
        shape,
        observer,
        it_model=None,
        not_it_model=None,
    ):
        self.screen = screen
        self.shape = shape
        self.agent = agent
        self.player = player
        self.obstacles = obstacles

        self.observer = observer
        self.it_model = it_model
        self.not_it_model = not_it_model

    def _translate_action(self, action):
        # if action < 3:
        #     lindir = 1
        # else:
        #     lindir = 0
        #
        # m = action % 3
        # if m == 0:
        #     angdir = 1
        # elif m == 1:
        #     angdir = 0
        # elif m == 2:
        #     angdir = -1
        #
        # return Action(
        #     lindir=[lindir, 0],
        #     angdir=angdir,
        #     target=None,
        #     reload=False,
        #     frame=Action.LOCAL,
        #     lookback=False,
        # )
        return Action(
            lindir=[1, 0],
            angdir=action,
            target=None,
            reload=False,
            frame=Action.LOCAL,
        )

    def _default_it_policy(self):
        """Policy for the agent that is "it"."""
        r = self.player.position - self.agent.position

        # steer toward the player
        a = angle2pi(r, start=self.agent.angle)
        if a < np.pi:
            angvel = 1
        elif a > np.pi:
            angvel = -1
        else:
            angvel = 0

        return Action(
            lindir=[1, 0],
            angdir=angvel,
            target=None,
            reload=False,
            frame=Action.LOCAL,
        )

    def _learned_it_policy(self):
        obs = self.observer.get_observation()

        # it model ignores treasures entirely
        if "treasure_positions" in obs:
            obs["treasure_positions"].fill(0)

        action, _ = self.it_model.predict(obs, deterministic=False)
        return self._translate_action(action)

    def _it_policy(self):
        if self.it_model is None:
            return self._default_it_policy()
        return self._learned_it_policy()

    def _default_not_it_policy(self):
        """Policy for agents that are not "it"."""
        r = self.player.position - self.agent.position
        d = self.agent.direction()

        if d @ r < 0:
            # we are already facing away from the player, so take whichever
            # direction orthogonal to center point moves us farther away from
            # the player
            p = self.agent.position - 0.5 * np.array(self.shape)
            v = orth(p)
            if v @ r > 0:
                v = -v
            a = angle2pi(v, start=self.agent.angle)
            if a < np.pi:
                angvel = 1
            elif a > np.pi:
                angvel = -1
        else:
            # steer away from the player
            a = angle2pi(r, start=self.agent.angle)
            if a < np.pi:
                angvel = -1
            elif a > np.pi:
                angvel = 1

        return Action(
            lindir=[1, 0],
            angdir=angvel,
            target=None,
            reload=False,
            frame=Action.LOCAL,
        )

    def _learned_not_it_policy(self):
        obs = self.observer.get_observation()
        action, _ = self.not_it_model.predict(obs, deterministic=False)
        return self._translate_action(action)

    def _not_it_policy(self):
        if self.not_it_model is None:
            return self._default_not_it_policy()
        return self._learned_not_it_policy()

    def compute(self):
        """Evaluate the policy at the current state."""
        if self.agent.it:
            return self._it_policy()
        return self._not_it_policy()
