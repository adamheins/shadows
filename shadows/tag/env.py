"""Learning environments."""

import pygame
import numpy as np
import gymnasium as gym

from ..entity import Agent, Action, PLAYER_FORWARD_VEL
from ..gui import Color
from ..obstacle import Obstacle
from ..collision import point_in_rect, point_poly_query, AARect
from ..math import *
from ..treasure import Treasure
from .policy import TagAIPolicy, ImageObserver, FullStateObserver


FRAMERATE = 60
TIMESTEP = 1.0 / FRAMERATE

SHAPE = (50, 50)

# use continuous linear and angular velocity as the actions
USE_CONTINUOUS_ACTIONS = True

# learn from observations of the screen pixels
USE_IMAGE_OBSERVATIONS = False

# draw the direction line onto the agents
DRAW_DIRECTION = False

# draw occlusions behind obstacles
DRAW_OCCLUSIONS = False

# number of treasures to collect
N_TREASURES = 2
TREASURE_RADIUS = 1

# print extra information
VERBOSE = True

# render the grayscale observation directly, rather than a
# nice-looking human version. RENDER_SCALE should be set to 1 if this
# is True.
RENDER_OBSERVATION = False

# scale up rendering by this value
RENDER_SCALE = 1

FRAME_SKIP = 1


class TagBaseEnv(gym.Env):
    """Environment where the agent is 'it'."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FRAMERATE}

    def __init__(
        self,
        render_mode="rgb_array",
        grayscale=True,
        sparse_reward=False,
        player_it=True,
        stationary_enemy=False,
        it_model=None,
        not_it_model=None,
        n_stack=1,
        max_steps=1000,
    ):
        pygame.init()

        self.shape = SHAPE
        self.render_shape = tuple(int(RENDER_SCALE * s) for s in self.shape)
        self.render_mode = render_mode
        self.grayscale = grayscale
        self.sparse_reward = sparse_reward
        self.stationary_enemy = stationary_enemy
        self.player_it = player_it
        self.max_steps = max_steps
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

        self.player = Agent.player(position=[10, 10], radius=3, it=player_it)
        self.enemy = Agent.enemy(position=[47, 47], radius=3, it=not player_it)

        # just for learning purposes
        self.player.color = Color.ENEMY
        self.enemy.color = Color.PLAYER

        # self.obstacles = []
        # self.obstacles = [Obstacle(20, 20, 10, 10)]
        self.obstacles = [
            Obstacle(20, 27, 10, 10),
            Obstacle(8, 8, 5, 5),
            # Obstacle(8, 37, 5, 5),
            Obstacle(0, 37, 13, 13),
            Obstacle(37, 37, 5, 5),
            # Obstacle(37, 8, 5, 5),
            Obstacle(20, 8, 5, 7),
            Obstacle(20, 15, 22, 5),
        ]

        self.treasures = [
            Treasure(center=[0, 0], radius=TREASURE_RADIUS) for _ in range(N_TREASURES)
        ]

        if USE_CONTINUOUS_ACTIONS:
            self.action_space = gym.spaces.Box(
                low=-np.ones(1, dtype=np.float32),
                high=np.ones(1, dtype=np.float32),
                shape=(1,),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Discrete(3)

        if USE_IMAGE_OBSERVATIONS:
            self.observer = ImageObserver(self.screen, self.player, n_stack=n_stack)
            self.observation_space = self.observer.space(
                self.shape, grayscale=grayscale
            )
        else:
            self.observer = FullStateObserver(
                self.player, self.enemy, treasures=self.treasures, n_stack=n_stack
            )
            self.observation_space = self.observer.space(self.shape)

        self.enemy_policy = TagAIPolicy(
            screen=self.screen,
            agent=self.enemy,
            player=self.player,
            obstacles=self.obstacles,
            shape=self.shape,
            observer=self.observer,
            it_model=it_model,
            not_it_model=None,
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

    def _translate_action(self, action):
        if USE_CONTINUOUS_ACTIONS:
            return Action(
                lindir=[1, 0],
                angdir=action,
                target=None,
                reload=False,
                frame=Action.LOCAL,
            )
        else:
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

    def reset(self, seed=None, options=None):
        """Reset the game environment."""
        super().reset(seed=seed)

        self._steps = 0

        r = self.player.radius

        agents = [self.player, self.enemy]
        for agent_idx, agent in enumerate(agents):
            agent.angle = self.np_random.uniform(low=-np.pi, high=np.pi)

            # generate collision-free position for each agent
            while True:
                agent.position = self.np_random.uniform(low=(0, 0), high=self.shape)

                # avoid collision with obstacles
                collision = False
                for obstacle in self.obstacles:
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

        # update treasure positions
        if not self.player_it:
            for treasure in self.treasures:
                treasure.update_position(
                    shape=self.shape, obstacles=self.obstacles, rng=self.np_random
                )

        self._draw(self.screen, self.screen_rect)
        obs = self.observer.get_observation()
        info = self._get_info()
        return obs, info

    def _potential(self):
        """Potential for current state."""
        d = np.linalg.norm(self.player.position - self.enemy.position)
        # potential for when player is it
        p = 1 - d / self._diag

        # when not it, potential is negated
        if not self.player_it:
            p = -p
        return p

    def step(self, action):
        treasures_collected = 0
        p0 = self._potential()
        for _ in range(FRAME_SKIP):
            self._steps += 1

            self.player.command(self._translate_action(action))
            if not self.stationary_enemy:
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
                    for obstacle in self.obstacles:
                        Q = point_poly_query(agent.position, obstacle)
                        if Q.distance < agent.radius and Q.normal @ v < 0:
                            tan = orth(Q.normal)
                            v = (tan @ v) * tan

                agent.velocity = v

            # check if player has collected a treasure
            # for treasure in self.treasures:
            #     d = np.linalg.norm(self.player.position - treasure.center)
            #     if d <= self.player.radius + treasure.radius:
            #         treasures_collected += 1
            #         treasure.update_position(
            #             shape=self.shape,
            #             obstacles=self.obstacles,
            #             rng=self.np_random,
            #         )

            # check if treasures have been collected
            if not self.player_it:
                for agent in agents:
                    if agent.it:
                        continue

                    for treasure in self.treasures:
                        d = np.linalg.norm(agent.position - treasure.center)
                        if d <= agent.radius + treasure.radius:
                            treasures_collected += 1
                            treasure.update_position(
                                shape=self.shape,
                                obstacles=self.obstacles,
                                rng=self.np_random,
                            )

            for agent in agents:
                agent.step(TIMESTEP)

            # round terminates when the player is caught
            r = self.player.radius + self.enemy.radius
            d = np.linalg.norm(self.player.position - self.enemy.position)
            terminated = bool(d < r)

            truncated = self._steps >= self.max_steps

            # stop frame skip if episode ends
            if terminated or truncated:
                break
        p1 = self._potential()

        # when it, there is a positive reward for catching the enemy
        reward = 1 if terminated else 0

        # when not it, there is a negative reward for being caught
        if not self.player_it:
            reward = -reward
            reward += 0.5 * treasures_collected

        # shape reward with potential function
        if not self.sparse_reward:
            F = p1 - p0
            reward += F

        # encourage high velocities
        # reward += self.player.last_vel_mag / PLAYER_FORWARD_VEL / self.max_steps

        if VERBOSE:
            if terminated:
                print("tagged!")
                print(f"  steps = {self._steps}")
            if treasures_collected > 0:
                print(f"treasures = {treasures_collected}")

        self._draw(self.screen, self.screen_rect)
        obs = self.observer.get_observation()
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
        if USE_IMAGE_OBSERVATIONS and RENDER_OBSERVATION:
            # 2D grayscale array
            obs = self.observer.get_observation()
            img = obs["image"].squeeze()

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


# register the environments
gym.register(
    id="TagIt-v0",
    entry_point=TagBaseEnv,
    kwargs=dict(player_it=True, stationary_enemy=True, max_steps=500),
)
gym.register(
    id="TagNotIt-v0",
    entry_point=TagBaseEnv,
    kwargs=dict(player_it=False, stationary_enemy=False, max_steps=1000),
)
