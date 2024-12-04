import pygame
import numpy as np

from ..math import *
from ..entity import Action
from ..gui import Color


class TagAIPolicy:
    """Basic AI policy for the tag game."""

    def __init__(self, agent, player, obstacles, shape):
        self.shape = shape
        self.agent = agent
        self.player = player
        self.obstacles = obstacles

    def _it_policy(self):
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

    def _not_it_policy(self):
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

    def compute(self):
        """Evaluate the policy at the current state."""
        if self.agent.it:
            return self._it_policy()
        return self._not_it_policy()


class LearnedTagAIPolicy:
    def __init__(self, screen, agent, player, obstacles, shape, model):
        self.screen = screen
        self.shape = shape
        self.agent = agent
        self.player = player
        self.obstacles = obstacles
        self.model = model

    def _get_rgb(self, screen):
        """Get RGB pixel values from the given screen."""
        return np.array(pygame.surfarray.pixels3d(screen), dtype=np.uint8)

    def _get_obs(self):
        """Construct the current observation."""
        rgb = self._get_rgb(self.screen)

        # TODO need to confirm the colors are right
        gray = np.zeros(self.shape + (1,), dtype=np.uint8)

        player_mask = np.all(rgb == self.agent.color, axis=-1)
        gray[player_mask, 0] = 1

        enemy_mask = np.all(rgb == self.player.color, axis=-1)
        gray[enemy_mask, 0] = 2

        obs_mask = np.all(rgb == Color.OBSTACLE, axis=-1)
        gray[obs_mask, 0] = 3

        return {
            "position": self.agent.position.astype(np.float32),
            "angle": np.array([self.agent.angle], dtype=np.float32),
            "image": gray,
        }

    def _translate_action(self, action):
        if action == 0:
            angdir = 1
        elif action == 1:
            angdir = 0
        elif action == 2:
            angdir = -1

        return Action(
            lindir=[1, 0],
            angdir=angdir,
            target=None,
            reload=False,
            frame=Action.LOCAL,
            lookback=False,
        )

    def _it_policy(self):
        """Policy for the agent that is "it"."""
        obs = self._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        return self._translate_action(action)

    def _not_it_policy(self):
        """Policy for agents that are not "it".

        Still hand-crafted, since I haven't trained an agent for this yet.
        """
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

    def compute(self):
        """Evaluate the policy at the current state."""
        if self.agent.it:
            return self._it_policy()
        return self._not_it_policy()
