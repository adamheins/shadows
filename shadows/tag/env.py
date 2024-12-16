import pygame
import numpy as np
import gymnasium as gym

from ..collision import point_poly_query
from ..entity import Action
from .game import TagGame


# put the RGB channel dimension first in the observation array, otherwise put
# it last
# RGB_CHANNEL_FIRST = False

# only give rewards when the other agent is successfully tagged
USE_SPARSE_REWARD = True

# use one flat discrete action space rather than a multidiscrete space
FLATTEN_ACTION_SPACE = True

# always move forward, only choice of action is the angular direction
REDUCE_ACTION_SPACE = True

# truncate each episode to at most this many timesteps (if the enemy is not
# tagged first)
MAX_STEPS_PER_EPISODE = 1000

USE_AI_POLICY = False

DRAW_OCCLUSIONS = False


class TagItEnv(gym.Env):
    """Environment where the agent is 'it'."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="rgb_array"):
        pygame.init()

        self.game = TagGame(
            player_it=True,
            invert_agent_colors=True,
            display=False,
            draw_occlusions=DRAW_OCCLUSIONS,
        )
        self.player = self.game.player
        self.enemy = self.game.enemies[0]

        # TODO don't really like this
        if render_mode == "human":
            self.game.screen = pygame.display.set_mode(
                self.game.shape, flags=pygame.SCALED
            )

        # directions to move
        # first channel is linear direction
        # second channel is angular direction
        if REDUCE_ACTION_SPACE:
            self.action_space = gym.spaces.Discrete(3 * 2)
        elif FLATTEN_ACTION_SPACE:
            self.action_space = gym.spaces.Discrete(3 * 3)
        else:
            self.action_space = gym.spaces.MultiDiscrete((3, 3))

        # RGB pixels
        # if RGB_CHANNEL_FIRST:
        #     shape = (3,) + self.game.shape
        # else:
        shape = self.game.shape + (3,)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        self.render_mode = render_mode

        # steps per episode
        self._steps = 0

        self._resets = 0

    def _get_info(self):
        return {
            "player_position": self.player.position,
            "enemy_position": self.enemy.position,
        }

    def _get_obs(self):
        return self.game.rgb()
        # if not RGB_CHANNEL_FIRST:
        #     rgb = np.moveaxis(rgb, 0, -1)
        # return rgb

    def _translate_action(self, action):
        if FLATTEN_ACTION_SPACE:
            if action < 3:
                lindir = 1
            elif 3 <= action < 6:
                lindir = 0
            else:
                lindir = -1

            m = action % 3
            if m == 0:
                angdir = 1
            elif m == 1:
                angdir = 0
            elif m == 2:
                angdir = -1
        else:
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

        # print(f"action = {action}")
        # print(f"lindir = {lindir}")
        # print(f"angdir = {angdir}")

        if USE_AI_POLICY:
            actions = self.game.enemy_policy.compute()
        else:
            actions = {}
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

        self._steps = 0

        # TODO we could reset the obstacles as well...
        n = 1000

        # randomly reset agent positions to collision-free positions
        w, h = self.game.shape
        for idx, agent in enumerate(self.game.agents):
            agent.angle = self.np_random.uniform(low=-np.pi, high=np.pi)
            # while True:
            for i in range(n):
                r = agent.radius
                agent.position = self.np_random.uniform(low=(0, 0), high=(w, h))
                collision = False

                # check for collision with other agents
                if idx > 0:
                    for other in self.game.agents[:idx]:
                        d = np.linalg.norm(agent.position - other.position)
                        if d < 2 * r + 2 * r:
                            collision = True
                            break
                if collision:
                    print("avoiding collision with other agent")
                    continue

                # check for collision with obstacles
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

        self.game.draw()
        obs = self._get_obs()
        info = self._get_info()
        self._resets += 1
        return obs, info

    def step(self, action):
        self._steps += 1
        self.game.step(self._translate_action(action))

        # round terminates when the enemy is tagged
        r = self.player.radius + self.enemy.radius
        d = np.linalg.norm(self.player.position - self.enemy.position)
        terminated = bool(d < r)

        truncated = self._steps >= MAX_STEPS_PER_EPISODE

        if USE_SPARSE_REWARD:
            reward = 1 if terminated else 0
        else:
            # big reward for tagging the enemy, small cost for being farther away
            # from it
            reward = 1000 if terminated else 2 - 0.01 * d

        if terminated:
            print("tagged!")
            print(f"  steps = {self._steps}")
            print(f"  d = {d}")
            print(f"  r = {reward}")

        self.game.draw()
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return self.game.rgb()


gym.register(
    id="TagIt-v0",
    entry_point=TagItEnv,
)
