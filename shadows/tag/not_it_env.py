import pygame
import numpy as np
import gymnasium as gym

from ..entity import Agent, Action, PLAYER_FORWARD_VEL
from ..gui import Color
from ..obstacle import Obstacle
from ..collision import point_in_rect, point_poly_query, AARect
from ..math import *
from .policy import TagAIPolicy


FRAMERATE = 60
TIMESTEP = 1.0 / FRAMERATE

SHAPE = (50, 50)

MAX_STEPS_PER_EPISODE = 1000

# don't move the enemy between episodes
FIX_ENEMY_POSITION = False

# only place the enemy in corners
ENEMY_ONLY_IN_CORNERS = False

# include the player's position and angle in the observation
APPEND_POSITION_TO_STATE = True

# use local frame for movement rather than the world frame
USE_LOCAL_FRAME_ACTIONS = True

# use a position target as the action rather than velocity commands
USE_TARGET_AS_ACTION = False

# draw the direction line onto the agents
DRAW_DIRECTION = False

# draw occlusions behind obstacles
DRAW_OCCLUSIONS = True

# use a policy for the computer-controlled agent
# this may be learned or handcrafted
USE_AI_POLICY = True

# print extra information
VERBOSE = True

# render the grayscale observation directly, rather than a
# nice-looking human version. RENDER_SCALE should be set to 1 if this
# is True.
RENDER_OBSERVATION = False

# scale up rendering by this value
RENDER_SCALE = 1

FRAME_SKIP = 1


# TODO basically just need to support a model in here for the enemy AI policy
class TagNotItEnv(gym.Env):
    """Environment where the agent is 'it'."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FRAMERATE}

    def __init__(
        self, render_mode="rgb_array", grayscale=True, it_model=None, not_it_model=None
    ):
        pygame.init()

        self.shape = SHAPE
        self.render_shape = tuple(int(RENDER_SCALE * s) for s in self.shape)
        self.render_mode = render_mode
        self.grayscale = grayscale
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

        # now the enemy is it
        self.player = Agent.player(position=[10, 10], radius=3)
        self.enemy = Agent.enemy(position=[47, 47], radius=3, it=True)

        # just for learning purposes
        self.player.color = Color.ENEMY
        self.enemy.color = Color.PLAYER

        # self.obstacles = []
        # self.obstacles = [Obstacle(20, 20, 10, 10)]
        self.obstacles = [
            Obstacle(20, 20, 10, 10),
            Obstacle(8, 8, 5, 5),
            Obstacle(8, 37, 5, 5),
            Obstacle(37, 37, 5, 5),
            Obstacle(37, 8, 5, 5),
        ]

        self.enemy_policy = TagAIPolicy(
            screen=self.screen,
            agent=self.enemy,
            player=self.player,
            obstacles=self.obstacles,
            shape=self.shape,
            it_model=it_model,
            not_it_model=None,
        )

        if USE_TARGET_AS_ACTION:
            self.action_space = gym.spaces.Box(
                low=-np.ones(2, dtype=np.float32),
                high=np.ones(2, dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
        elif USE_LOCAL_FRAME_ACTIONS:
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

            player_mask = np.all(rgb == Color.ENEMY, axis=-1)
            gray[player_mask, 0] = 85

            enemy_mask = np.all(rgb == Color.PLAYER, axis=-1)
            gray[enemy_mask, 0] = 170

            obs_mask = np.all(rgb == Color.OBSTACLE, axis=-1)
            gray[obs_mask, 0] = 255

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
        if USE_TARGET_AS_ACTION:
            target = 0.5 * (action + 1) * self.shape
            r = target - self.player.position

            # don't go anywhere if target is on top of the player
            if np.linalg.norm(r) < self.player.radius:
                linvel = 0
            else:
                linvel = 1

            # steer toward the player
            a = angle2pi(r, start=self.player.angle)
            if a < np.pi:
                angvel = 1
            elif a > np.pi:
                angvel = -1
            else:
                angvel = 0

            return Action(
                lindir=[linvel, 0],
                angdir=angvel,
                target=None,
                reload=False,
                frame=Action.LOCAL,
            )
        elif USE_LOCAL_FRAME_ACTIONS:
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
        if not FIX_ENEMY_POSITION and not ENEMY_ONLY_IN_CORNERS:
            agents.append(self.enemy)

        if ENEMY_ONLY_IN_CORNERS:
            corner = self.np_random.integers(4)
            if corner == 0:
                p = (3, 3)
            elif corner == 1:
                p = (3, 47)
            elif corner == 2:
                p = (47, 3)
            else:
                p = (47, 47)
            self.enemy.position = np.array(p)

        for agent_idx, agent in enumerate(agents):
            if USE_LOCAL_FRAME_ACTIONS:
                agent.angle = self.np_random.uniform(low=-np.pi, high=np.pi)

            # generate collision-free position for each agent
            while True:
                agent.position = self.np_random.uniform(low=(0, 0), high=(w, h))

                # avoid collision with obstacles
                collision = False
                for obstacle in self.obstacles:
                    # Q = point_poly_query(agent.position, obstacle)
                    # if Q.distance <= r:
                    #     collision = True
                    #     break
                    if point_in_rect(agent.position, obstacle):
                        collision = True
                        break

                # avoid collision with other agents
                if agent_idx > 0:
                    for other in agents[:agent_idx]:
                        d = np.linalg.norm(agent.position - other.position)
                        if d <= 2 * r:
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

        # negative of the It environment
        return -(1 - d / self._diag)

    def step(self, action):
        p0 = self._potential()
        for _ in range(FRAME_SKIP):
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

            for agent in agents:
                agent.step(TIMESTEP)

            # round terminates when the player is caught
            r = self.player.radius + self.enemy.radius
            d = np.linalg.norm(self.player.position - self.enemy.position)
            terminated = bool(d < r)

            truncated = self._steps >= MAX_STEPS_PER_EPISODE

            # stop frame skip if episode ends
            if terminated or truncated:
                break
        p1 = self._potential()

        # negative reward if caught
        reward = -1 if terminated else 0

        # shaped reward
        F = p1 - p0
        reward += F

        # positive reward if not caught
        # reward = 1 if not terminated else 0
        # reward = self.player.last_vel_mag / PLAYER_FORWARD_VEL if not terminated else 0

        # TODO add some reward if a treasure is taken

        if VERBOSE and terminated:
            print("tagged!")
            print(f"  steps = {self._steps}")

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
        if RENDER_OBSERVATION:
            # 2D grayscale array
            img = self._get_obs()["image"].squeeze()

            # make into RGB array but still grayscale, so pygame can
            # render it properly
            rgb = np.stack([img, img, img], axis=-1)
            surf = pygame.surfarray.make_surface(rgb)
            self.render_screen.blit(surf, dest=(0, 0))
        else:
            # draw the human-friendly version
            self._draw(self.render_screen, self.render_screen_rect, scale=RENDER_SCALE)

        if self.render_mode == "human":
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return self._get_rgb(self.render_screen)


gym.register(id="TagNotIt-v0", entry_point=TagNotItEnv)
