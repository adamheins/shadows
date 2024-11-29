import pygame
import pygame.gfxdraw
import numpy as np

from ..collision import *
from ..math import *
from ..gui import Text, Color
from ..entity import Agent, Action, Projectile
from ..obstacle import Obstacle


FRAMERATE = 60
TIMESTEP = 1.0 / FRAMERATE

TAG_COOLDOWN = 120  # ticks

# for more efficiency we can turn off continuous collision detection
USE_CCD = False

USE_AI_POLICY = True


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

        return {
            self.agent.id: Action(
                lindir=[1, 0],
                angdir=angvel,
                target=None,
                reload=False,
                frame=Action.LOCAL,
            )
        }

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

        return {
            self.agent.id: Action(
                lindir=[1, 0],
                angdir=angvel,
                target=None,
                reload=False,
                frame=Action.LOCAL,
            )
        }

    def compute(self):
        """Evaluate the policy at the current state."""
        if self.agent.it:
            return self._it_policy()
        return self._not_it_policy()


class TagGame:
    def __init__(
        self,
        shape=(200, 200),
        player_it=False,
        invert_agent_colors=False,
        display=True,
        draw_occlusions=True,
    ):
        self.shape = shape
        self.display = display
        self.draw_occlusions = draw_occlusions

        if self.display:
            self.screen = pygame.display.set_mode(shape, flags=pygame.SCALED)
        else:
            self.screen = pygame.Surface(shape)
        self.clock = pygame.time.Clock()
        self.screen_rect = AARect(0, 0, shape[0], shape[1])

        self.keys_down = set()

        # self.obstacles = [
        #     Obstacle(80, 80, 80, 80),
        #     Obstacle(0, 230, 160, 40),
        #     Obstacle(80, 340, 80, 80),
        #     Obstacle(250, 80, 40, 220),
        #     Obstacle(290, 80, 100, 40),
        #     Obstacle(390, 260, 110, 40),
        #     Obstacle(250, 380, 200, 40),
        # ]

        # self.obstacles = [
        #     Obstacle(75, 75, 50, 50),
        # ]

        # self.obstacles = [
        #     Obstacle(75, 75, 50, 50),
        #     Obstacle(25, 25, 25, 25),
        #     Obstacle(150, 25, 25, 25),
        #     Obstacle(150, 150, 25, 25),
        #     Obstacle(25, 150, 25, 25),
        # ]
        self.obstacles = []

        # player and enemy agents
        self.player = Agent.player(position=[100, 50])
        self.enemies = [Agent.enemy(position=[150, 150])]
        self.agents = [self.player] + self.enemies

        # just for learning purposes
        if invert_agent_colors:
            self.player.color = Color.ENEMY
            self.enemies[0].color = Color.PLAYER

        # id of the agent that is "it"
        if player_it:
            self.player.it = True
            self.it_id = 0
        else:
            self.enemies[0].it = True
            self.it_id = 1

        self.tag_cooldown = 0

        self.enemy_policy = TagAIPolicy(
            agent=self.enemies[0],
            player=self.player,
            obstacles=self.obstacles,
            shape=self.shape,
        )

    def draw(self):
        """Render the game.

        If ``display`` is ``False``, the screen image is returned as an RGB
        array.
        """
        self.screen.fill(Color.BACKGROUND)

        for enemy in self.enemies:
            enemy.draw(self.screen)

        if self.draw_occlusions:
            self.player.draw_view_occlusion(self.screen, self.screen_rect)
        self.player.draw(self.screen)

        if self.draw_occlusions:
            for obstacle in self.obstacles:
                obstacle.draw_occlusion(
                    self.screen,
                    viewpoint=self.player.position,
                    screen_rect=self.screen_rect,
                )

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        if self.display:
            pygame.display.flip()

    def rgb(self):
        """Return the current screen image as a RGB pixel array."""
        return np.array(pygame.surfarray.pixels3d(self.screen), dtype=np.uint8)

    def step(self, actions):
        """Step the game forward in time."""
        self.tag_cooldown = max(0, self.tag_cooldown - 1)

        for agent in self.agents:
            if agent.id in actions:
                action = actions[agent.id]
                agent.command(action)

        # agents cannot walk off the screen and into obstacles
        for agent in self.agents:
            v = agent.velocity

            # don't leave the screen
            if agent.position[0] >= self.screen_rect.w - agent.radius:
                v[0] = min(0, v[0])
            elif agent.position[0] <= agent.radius:
                v[0] = max(0, v[0])
            if agent.position[1] >= self.screen_rect.h - agent.radius:
                v[1] = min(0, v[1])
            elif agent.position[1] <= agent.radius:
                v[1] = max(0, v[1])

            # don't walk into an obstacle
            if np.linalg.norm(v) > 0:
                # collision time and normal
                min_time = None
                normal = None
                if USE_CCD:
                    path = Segment(agent.position, agent.position + TIMESTEP * v)
                    for obstacle in self.obstacles:
                        Q = swept_circle_poly_query(path, agent.radius, obstacle)
                        if Q.intersect and (min_time is None or t < min_time):
                            min_time = Q.time
                            normal = Q.normal

                else:
                    for obstacle in self.obstacles:
                        Q = point_poly_query(agent.position, obstacle)
                        if Q.distance < agent.radius:
                            min_time = 0
                            normal = Q.normal
                            break

                if min_time is not None and normal @ v < 0:
                    # tangent velocity
                    tan = orth(normal)
                    vtan = (tan @ v) * tan
                    v = min_time * v + (1 - min_time) * vtan

            agent.velocity = v

        # check if someone has been tagged
        if self.tag_cooldown == 0:
            it_agent = self.agents[self.it_id]
            for i, agent in enumerate(self.agents):
                if i == self.it_id:
                    continue

                # switch who is "it"
                d = agent.radius + it_agent.radius
                if np.linalg.norm(agent.position - it_agent.position) < d:
                    self.tag_cooldown = TAG_COOLDOWN
                    it_agent.it = False
                    agent.it = True
                    self.it_id = i
                    break

        # cannot move after just being tagged
        if self.tag_cooldown > 0:
            self.agents[self.it_id].velocity = np.zeros(2)

        for agent in self.agents:
            agent.step(TIMESTEP)

    def loop(self):
        """Main game loop."""
        while True:

            # process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    self.keys_down.add(event.key)
                elif event.type == pygame.KEYUP:
                    self.keys_down.discard(event.key)

            # respond to events
            lindir = 0
            angdir = 0
            if pygame.K_d in self.keys_down:
                angdir -= 1
            if pygame.K_a in self.keys_down:
                angdir += 1
            if pygame.K_w in self.keys_down:
                lindir += 1
            if pygame.K_s in self.keys_down:
                lindir -= 1

            lookback = pygame.K_SPACE in self.keys_down

            if USE_AI_POLICY:
                actions = self.enemy_policy.compute()
            else:
                actions = {}
            actions[self.player.id] = Action(
                lindir=[lindir, 0],
                angdir=angdir,
                target=None,
                reload=False,
                frame=Action.LOCAL,
                lookback=lookback,
            )

            self.step(actions)
            self.draw()
            self.clock.tick(FRAMERATE)