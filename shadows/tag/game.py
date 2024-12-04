import pygame
import numpy as np

from ..collision import *
from ..math import *
from ..gui import Text, Color
from ..entity import Agent, Action
from ..obstacle import Obstacle
from .policy import TagAIPolicy, LearnedTagAIPolicy


FRAMERATE = 60
TIMESTEP = 1.0 / FRAMERATE

TAG_COOLDOWN = 120  # ticks

# for more efficiency we can turn off continuous collision detection
USE_CCD = False

USE_AI_POLICY = True

RENDER_SCALE = 4


class TagGame:
    def __init__(
        self,
        shape=(50, 50),
        invert_agent_colors=False,
        display=True,
        model=None,
    ):
        self.display = display

        self.shape = shape
        self.render_shape = tuple(int(RENDER_SCALE * s) for s in self.shape)

        self.screen = pygame.Surface(self.shape)
        self.screen_rect = AARect(0, 0, self.shape[0], self.shape[1])

        if self.display:
            self.render_screen = pygame.display.set_mode(
                self.render_shape, flags=pygame.SCALED
            )
            self.render_screen_rect = AARect(
                0, 0, self.render_shape[0], self.render_shape[1]
            )

        self.clock = pygame.time.Clock()
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
        # self.obstacles = []
        self.obstacles = [
            Obstacle(20, 20, 10, 10),
            Obstacle(8, 8, 5, 5),
            Obstacle(8, 37, 5, 5),
            Obstacle(37, 37, 5, 5),
            Obstacle(37, 8, 5, 5),
        ]

        # player and enemy agents
        self.player = Agent.player(position=[10, 25], radius=3, it=False)
        self.enemy = Agent.enemy(position=[40, 25], radius=3, it=True)
        self.agents = [self.player, self.enemy]
        self.it_id = 1

        # just for learning purposes
        if invert_agent_colors:
            self.player.color = Color.ENEMY
            self.enemies[0].color = Color.PLAYER

        # id of the agent that is "it"
        # if player_it:
        #     self.player.it = True
        #     self.it_id = 0
        # else:
        #     self.enemy.it = True
        #     self.it_id = 1

        self.tag_cooldown = 0

        # self.enemy_policy = TagAIPolicy(
        #     agent=self.enemy,
        #     player=self.player,
        #     obstacles=self.obstacles,
        #     shape=self.shape,
        # )
        if model is None:
            self.enemy_policy = TagAIPolicy(
                agent=self.enemy,
                player=self.player,
                obstacles=self.obstacles,
                shape=self.shape,
            )
        else:
            self.enemy_policy = LearnedTagAIPolicy(
                screen=self.screen,
                agent=self.enemy,
                player=self.player,
                obstacles=self.obstacles,
                shape=self.shape,
                model=model,
            )

    def _draw(
        self,
        screen,
        screen_rect,
        viewpoint,
        scale=1,
        draw_direction=True,
        draw_outline=True,
    ):
        screen.fill(Color.BACKGROUND)

        for agent in self.agents:
            agent.draw(
                surface=screen,
                draw_direction=draw_direction,
                draw_outline=draw_outline,
                scale=scale,
            )

        # if self.draw_occlusions:
        #     self.player.draw_view_occlusion(self.screen, self.screen_rect)
        # self.player.draw(
        #     screen,
        #     draw_direction=draw_direction,
        #     draw_outline=draw_outline,
        #     scale=scale,
        # )

        # TODO this is wrong because it does not scale properly
        for obstacle in self.obstacles:
            obstacle.draw(surface=screen, scale=scale)
            obstacle.draw_occlusion(
                surface=screen,
                viewpoint=viewpoint,
                # screen_rect=self.screen_rect,  # TODO
                screen_rect=screen_rect,
                scale=scale,
            )

    def draw_enemy_screen(self):
        self._draw(
            screen=self.screen,
            screen_rect=self.screen_rect,
            viewpoint=self.enemy.position,
            scale=1,
            draw_direction=False,
            draw_outline=False,
        )

    def draw_player_screen(self):
        self._draw(
            screen=self.render_screen,
            screen_rect=self.render_screen_rect,
            viewpoint=self.player.position,
            scale=RENDER_SCALE,
            draw_direction=True,
            draw_outline=True,
        )

    def render_display(self):
        self.draw_player_screen()
        pygame.display.flip()

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

            # TODO hardcoded indices here
            actions = {}
            if USE_AI_POLICY:
                self.draw_enemy_screen()
                actions[1] = self.enemy_policy.compute()
            actions[0] = Action(
                lindir=[lindir, 0],
                angdir=angdir,
                target=None,
                reload=False,
                frame=Action.LOCAL,
                lookback=lookback,
            )

            self.step(actions)
            self.render_display()
            self.clock.tick(FRAMERATE)