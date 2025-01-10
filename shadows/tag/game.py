import pygame
import numpy as np

from ..collision import *
from ..math import *
from ..gui import Text, Color
from ..entity import Agent, Action
from ..obstacle import Obstacle
from .policy import TagAIPolicy, FullStateObserver
from .treasure import Treasure


FRAMERATE = 60
TIMESTEP = 1.0 / FRAMERATE

TAG_COOLDOWN = 60  # ticks

# for more efficiency we can turn off continuous collision detection
USE_CCD = False
USE_AI_POLICY = True
ALLOW_TAG_SWITCH = True

N_TREASURES = 2
TREASURE_RADIUS = 1
OCCLUDE_TREASURES = True
DRAW_OCCLUSIONS = False

RENDER_SCALE = 8


class TagGame:
    def __init__(
        self,
        shape=(50, 50),
        display=True,
        it_model=None,
        not_it_model=None,
        rng=None,
    ):
        self.rng = np.random.default_rng(rng)

        self.shape = shape
        self.render_shape = tuple(int(RENDER_SCALE * s) for s in self.shape)

        self.screen = pygame.Surface(self.shape)
        self.screen_rect = AARect(0, 0, self.shape[0], self.shape[1])

        if display:
            self.render_screen = pygame.display.set_mode(
                self.render_shape, flags=pygame.SCALED
            )
            # self.screen = pygame.display.set_mode(
            #     self.shape, flags=pygame.SCALED
            # )
            # self.render_screen = pygame.Surface(self.render_shape)

        self.font = pygame.font.SysFont(None, 3 * RENDER_SCALE)
        self.clock = pygame.time.Clock()
        self.keys_down = set()

        # self.obstacles = []
        # self.obstacles = [
        #     Obstacle(20, 20, 10, 10),
        #     Obstacle(8, 8, 5, 5),
        #     Obstacle(8, 37, 5, 5),
        #     Obstacle(37, 37, 5, 5),
        #     Obstacle(37, 8, 5, 5),
        # ]
        # self.obstacles = [Obstacle(20, 20, 10, 10)]
        self.obstacles = [
            Obstacle(20, 27, 10, 10),
            Obstacle(8, 8, 5, 5),
            Obstacle(0, 37, 13, 13),
            Obstacle(37, 37, 5, 5),
            Obstacle(20, 8, 5, 7),
            Obstacle(20, 15, 22, 5),
        ]

        # player and enemy agents
        self.player = Agent.player(position=[10, 25], radius=3, it=False)
        self.enemy = Agent.enemy(position=[40, 25], radius=3, it=True)
        self.agents = [self.player, self.enemy]
        self.it_id = 1

        self.score = 0
        self.treasures = [
            Treasure(center=[0, 0], radius=TREASURE_RADIUS) for _ in range(N_TREASURES)
        ]
        for treasure in self.treasures:
            treasure.update_position(
                shape=self.shape, obstacles=self.obstacles, rng=self.rng
            )

        self.tag_cooldown = 0

        self.observer = FullStateObserver(
            agent=self.enemy, enemy=self.player, treasures=self.treasures
        )
        self.enemy_policy = TagAIPolicy(
            screen=self.screen,
            agent=self.enemy,
            player=self.player,
            obstacles=self.obstacles,
            shape=self.shape,
            observer=self.observer,
            it_model=it_model,
            not_it_model=not_it_model,
        )

    def _draw(
        self,
        screen,
        viewpoint,
        scale=1,
        draw_direction=True,
        draw_outline=True,
        draw_occlusion=True,
        draw_treasure=True,
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
        if draw_treasure and OCCLUDE_TREASURES:
            for treasure in self.treasures:
                treasure.draw(surface=screen, scale=scale)

        # NOTE screen_rect is always the unscaled version
        for obstacle in self.obstacles:
            obstacle.draw(surface=screen, scale=scale)
            if draw_occlusion:
                obstacle.draw_occlusion(
                    surface=screen,
                    viewpoint=viewpoint,
                    screen_rect=self.screen_rect,
                    scale=scale,
                )

        if draw_treasure and not OCCLUDE_TREASURES:
            for treasure in self.treasures:
                treasure.draw(surface=screen, scale=scale)

        if draw_treasure:
            # TODO render on a background?
            text = f"Score: {int(self.score)}"
            image = self.font.render(text, True, (255, 255, 255))
            screen.blit(image, scale * np.array([2, 45]))

    def draw_enemy_screen(self):
        self._draw(
            screen=self.screen,
            viewpoint=self.enemy.position,
            scale=1,
            draw_direction=False,
            draw_outline=False,
            draw_occlusion=DRAW_OCCLUSIONS,
            draw_treasure=False,
        )

    def draw_player_screen(self):
        self._draw(
            screen=self.render_screen,
            viewpoint=self.player.position,
            scale=RENDER_SCALE,
            draw_direction=True,
            draw_outline=True,
            draw_occlusion=DRAW_OCCLUSIONS,
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
                if USE_CCD:
                    path = Segment(agent.position, agent.position + TIMESTEP * v)
                    for obstacle in self.obstacles:
                        Q = swept_circle_poly_query(path, agent.radius, obstacle)
                        if Q.intersect and Q.normal @ v < 0:
                            tan = orth(Q.normal)
                            vtan = (tan @ v) * tan
                            v = Q.time * v + (1 - Q.time) * vtan

                else:
                    for obstacle in self.obstacles:
                        Q = point_poly_query(agent.position, obstacle)
                        if Q.distance < agent.radius and Q.normal @ v < 0:
                            tan = orth(Q.normal)
                            v = (tan @ v) * tan

            agent.velocity = v

        # check if someone has collected a treasure
        for agent in self.agents:
            if agent.it:
                continue

            for treasure in self.treasures:
                d = np.linalg.norm(agent.position - treasure.center)
                if d <= agent.radius + treasure.radius:
                    if agent is self.player:
                        self.score += 1
                    else:
                        self.score -= 1

                    treasure.update_position(
                        shape=self.shape, obstacles=self.obstacles, rng=self.rng
                    )

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

                    # manually switch who is it, for testing purposes
                    if ALLOW_TAG_SWITCH and event.key == pygame.K_t:
                        self.it_id = (self.it_id + 1) % len(self.agents)
                        self.agents[0].it = not self.agents[0].it
                        self.agents[1].it = not self.agents[1].it

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
