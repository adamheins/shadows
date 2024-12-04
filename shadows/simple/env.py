import pygame
import numpy as np
import gymnasium as gym

from ..entity import Agent, Action
from ..gui import Color
from ..obstacle import Obstacle
from ..collision import point_poly_query, AARect
from ..math import *


FRAMERATE = 60
TIMESTEP = 1.0 / FRAMERATE

SHAPE = (50, 50)

MAX_STEPS_PER_EPISODE = 500
FIX_ENEMY_POSITION = False
APPEND_POSITION_TO_STATE = True
USE_LOCAL_FRAME_ACTIONS = True
DRAW_DIRECTION = False
DRAW_OCCLUSIONS = True
USE_AI_POLICY = True
VERBOSE = True

# scale up rendering by this value
RENDER_SCALE = 4


class SimpleNotItPolicy:
    def __init__(self, agent, player, obstacles, shape):
        self.shape = shape
        self.agent = agent
        self.player = player
        self.obstacles = obstacles

    def compute(self):
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


class SimpleEnv(gym.Env):
    """Environment where the agent is 'it'."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FRAMERATE}

    def __init__(self, render_mode="rgb_array", grayscale=True, sparse_reward=False):
        pygame.init()

        self.shape = SHAPE
        self.render_shape = tuple(int(RENDER_SCALE * s) for s in self.shape)
        self.render_mode = render_mode
        self.grayscale = grayscale
        self.sparse_reward = sparse_reward
        self._diag = np.linalg.norm(self.shape)

        self.screen = pygame.Surface(self.shape)
        self.screen_rect = AARect(0, 0, self.shape[0], self.shape[1])

        if render_mode == "human":
            # self.render_screen = pygame.display.set_mode(self.render_shape)
            self.render_screen = pygame.display.set_mode(
                self.render_shape, flags=pygame.SCALED
            )
            self.render_screen_rect = AARect(
                0, 0, self.render_shape[0], self.render_shape[1]
            )

        self.player = Agent.player(position=[10, 10], radius=3, it=True)
        self.enemy = Agent.enemy(position=[40, 40], radius=3)

        # just for learning purposes
        self.player.color = Color.ENEMY
        self.enemy.color = Color.PLAYER

        # self.obstacles = []
        self.obstacles = [Obstacle(20, 20, 10, 10)]
        # self.obstacles = [
        #     Obstacle(20, 20, 10, 10),
        #     Obstacle(8, 8, 5, 5),
        #     Obstacle(8, 37, 5, 5),
        #     Obstacle(37, 37, 5, 5),
        #     Obstacle(37, 8, 5, 5),
        # ]

        self.enemy_policy = SimpleNotItPolicy(
            agent=self.enemy,
            player=self.player,
            obstacles=self.obstacles,
            shape=self.shape,
        )

        if USE_LOCAL_FRAME_ACTIONS:
            self.action_space = gym.spaces.Discrete(3)
        else:
            self.action_space = gym.spaces.Discrete(4)

        if grayscale:
            img_shape = self.shape + (1,)
        else:
            img_shape = self.shape + (3,)

        if APPEND_POSITION_TO_STATE:
            self.observation_space = gym.spaces.Dict(
                {
                    "position": gym.spaces.Box(
                        low=np.zeros(2, dtype=np.float32),
                        high=np.array(self.shape, dtype=np.float32),
                        shape=(2,),
                        dtype=np.float32,
                    ),
                    "angle": gym.spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
                    "image": gym.spaces.Box(
                        low=0, high=255, shape=img_shape, dtype=np.uint8
                    ),
                }
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )

        # steps per episode
        self._steps = 0

    def _get_info(self):
        return {
            "player_position": self.player.position,
            "enemy_position": self.enemy.position,
        }

    def _get_rgb(self, screen):
        """Get RGB pixel values from the given screen."""
        return np.array(pygame.surfarray.pixels3d(screen), dtype=np.uint8)

    def _get_obs(self):
        """Construct the current observation."""
        rgb = self._get_rgb(self.screen)

        if self.grayscale:
            gray = np.zeros(self.shape + (1,), dtype=np.uint8)

            player_mask = np.all(rgb == self.player.color, axis=-1)
            gray[player_mask, 0] = 1

            enemy_mask = np.all(rgb == self.enemy.color, axis=-1)
            gray[enemy_mask, 0] = 2

            obs_mask = np.all(rgb == Color.OBSTACLE, axis=-1)
            gray[obs_mask, 0] = 3

            img = gray
        else:
            img = rgb

        if APPEND_POSITION_TO_STATE:
            return {
                "position": self.player.position.astype(np.float32),
                "angle": np.array([self.player.angle], dtype=np.float32),
                "image": img,
            }
        return img

    def _translate_action(self, action):
        if USE_LOCAL_FRAME_ACTIONS:
            if action < 3:
                lindir = 1
            else:
                lindir = 0

            m = action % 3
            if m == 0:
                angdir = 1
            elif m == 1:
                angdir = 0
            elif m == 2:
                angdir = -1
            return Action(
                lindir=[lindir, 0],
                angdir=angdir,
                target=None,
                reload=False,
                frame=Action.LOCAL,
                lookback=False,
            )
        else:
            if action == 0:
                v = [1, 0]
            elif action == 1:
                v = [-1, 0]
            elif action == 2:
                v = [0, 1]
            else:
                v = [0, -1]
            return Action(
                lindir=v,
                angdir=0,
                target=None,
                reload=False,
                frame=Action.WORLD,
                lookback=False,
            )

    def reset(self, seed=None, options=None):
        """Reset the game environment."""
        super().reset(seed=seed)

        self._steps = 0

        r = self.player.radius
        w, h = self.shape

        agents = [self.player]
        if not FIX_ENEMY_POSITION:
            agents.append(self.enemy)

        for agent in agents:
            if USE_LOCAL_FRAME_ACTIONS:
                agent.angle = self.np_random.uniform(low=-np.pi, high=np.pi)

            # generate collision-free position for each agent
            while True:
                agent.position = self.np_random.uniform(low=(r, r), high=(w - r, h - r))
                collision = False
                for obstacle in self.obstacles:
                    Q = point_poly_query(agent.position, obstacle)
                    if Q.distance <= r:
                        collision = True
                        break
                if not collision:
                    break

        self._draw(self.screen, self.screen_rect)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _potential(self):
        """Potential for current state."""
        d = np.linalg.norm(self.player.position - self.enemy.position)
        return 1 - d / self._diag

    def step(self, action):
        self._steps += 1

        self.player.command(self._translate_action(action))
        if USE_AI_POLICY:
            self.enemy.command(self.enemy_policy.compute())

        agents = [self.player, self.enemy]
        for agent in agents:
            v = agent.velocity
            if np.linalg.norm(v) > 0:
                # don't leave the screen
                if agent.position[0] >= self.shape[0] - agent.radius:
                    v[0] = min(0, v[0])
                elif agent.position[0] <= agent.radius:
                    v[0] = max(0, v[0])
                if agent.position[1] >= self.shape[1] - agent.radius:
                    v[1] = min(0, v[1])
                elif agent.position[1] <= agent.radius:
                    v[1] = max(0, v[1])

                # don't penetrate obstacles
                normal = None
                for obstacle in self.obstacles:
                    Q = point_poly_query(agent.position, obstacle)
                    if Q.distance < agent.radius:
                        normal = Q.normal
                        break

                if normal is not None and normal @ v < 0:
                    tan = orth(normal)  # tangent velocity
                    v = (tan @ v) * tan

            agent.velocity = v

        p0 = self._potential()
        for agent in agents:
            agent.step(TIMESTEP)
        p1 = self._potential()

        # round terminates when the enemy is tagged
        r = self.player.radius + self.enemy.radius
        d = np.linalg.norm(self.player.position - self.enemy.position)
        terminated = bool(d < r)

        truncated = self._steps >= MAX_STEPS_PER_EPISODE

        reward = 1 if terminated else 0

        # use potential to shape reward
        if not self.sparse_reward:
            F = p1 - p0
            reward += F
            # reward = 1 if terminated else -0.01 * d

        if VERBOSE and terminated:
            print("tagged!")
            print(f"  steps = {self._steps}")
            print(f"  d = {d}")
            print(f"  r = {reward}")

        self._draw(self.screen, self.screen_rect)
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _draw(self, screen, screen_rect, scale=1):
        """Draw the screen."""
        screen.fill(Color.BACKGROUND)

        self.player.draw(
            screen, scale=scale, draw_direction=DRAW_DIRECTION, draw_outline=False
        )
        self.enemy.draw(
            screen, scale=scale, draw_direction=DRAW_DIRECTION, draw_outline=False
        )

        for obstacle in self.obstacles:
            obstacle.draw(screen, scale=scale)
            if DRAW_OCCLUSIONS:
                obstacle.draw_occlusion(
                    screen,
                    viewpoint=self.player.position,
                    screen_rect=screen_rect,
                    scale=scale,
                )

    def render(self):
        self._draw(self.render_screen, self.render_screen_rect, scale=RENDER_SCALE)

        if self.render_mode == "human":
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return self._get_rgb(self.render_screen)


gym.register(id="Simple-v0", entry_point=SimpleEnv)
