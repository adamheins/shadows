#!/usr/bin/env python3
import pygame
import pygame.gfxdraw
import numpy as np

import gymnasium as gym

from shoot import *


SCREEN_SHAPE = (500, 500)
BACKGROUND_COLOR = (219, 200, 184)

FRAMERATE = 60
TIMESTEP = 1.0 / FRAMERATE

TAG_COOLDOWN = 120  # ticks


class TagItAIPolicy:
    def __init__(self, agent_id, player_id, agents, obstacles):
        self.agent = agents[agent_id]
        self.player = agents[player_id]

        self.agents = agents
        self.obstacles = obstacles

    def step(self, dt):
        if not self.agent.it:
            return {}  # do nothing

        r = self.player.position - self.agent.position

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


class TagGame:
    def __init__(self, shape, display=True):
        self.display = display

        if self.display:
            self.screen = pygame.display.set_mode(shape)
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
        self.obstacles = [
            Obstacle(200, 200, 100, 100),
        ]

        # player and enemy agents
        self.player = Agent.player(position=[100, 100])
        self.enemies = [Agent.enemy(position=[400, 400], it=True)]
        self.agents = [self.player] + self.enemies

        # id of the agent that is "it"
        self.it_id = self.enemies[0].id
        self.tag_cooldown = 0

        self.enemy_policy = TagItAIPolicy(
            self.enemies[0].id, self.player.id, self.agents, self.obstacles
        )

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)

        for enemy in self.enemies:
            enemy.draw(self.screen)

        self.player.draw_view_occlusion(self.screen, self.screen_rect)
        self.player.draw(self.screen)

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
        else:
            # TODO build a semantic map
            color = self.screen.map_rgb(BACKGROUND_COLOR)
            raw = np.array(pygame.PixelArray(self.screen))
            print(np.sum(raw == color))
            rgb = np.array([raw >> 16, raw >> 8, raw]) & 0xFF
            rgb = np.moveaxis(rgb, 0, -1)
            return rgb

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
                for obstacle in self.obstacles:
                    n = obstacle.compute_collision_normal(agent.position, agent.radius)
                    if n is not None and n @ v < 0:
                        t = orth(n)
                        v = (t @ v) * t

            agent.velocity = v

        # check if someone has been tagged
        if self.tag_cooldown == 0:
            it_agent = self.agents[self.it_id]
            for agent in self.agents:
                if agent.id == self.it_id:
                    continue

                # switch who is "it"
                d = agent.radius + it_agent.radius
                if np.linalg.norm(agent.position - it_agent.position) < d:
                    self.tag_cooldown = TAG_COOLDOWN
                    it_agent.it = False
                    agent.it = True
                    self.it_id = agent.id
                    break

        for agent in self.agents:
            agent.step(TIMESTEP)

    def loop(self):
        while True:

            # process events
            viewtarget = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    self.keys_down.add(event.key)
                elif event.type == pygame.KEYUP:
                    self.keys_down.discard(event.key)
                # elif event.type == pygame.MOUSEMOTION:
                #     viewtarget = pygame.mouse.get_pos()

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

            actions = self.enemy_policy.step(TIMESTEP)
            # actions = {}
            actions[self.player.id] = Action(
                lindir=[lindir, 0],
                angdir=angdir,
                target=None,
                reload=False,
                frame=Action.LOCAL,
                viewtarget=viewtarget,
            )

            self.step(actions)
            self.draw()
            self.clock.tick(FRAMERATE)


class ShootEnv(gym.Env):
    def __init__(self, shape):
        self.game = Game(shape, display=False)

        # get enemy id
        self.agent_id = self.game.enemies[0].id

        self.action_space = gym.spaces.Dict(
            {
                "direction": gym.spaces.Discrete(8),
                "target": gym.spaces.Box(low=[0, 0], high=shape, dtype=np.float32),
                "reload": gym.spaces.MultiBinary(1),
            }
        )

        # semantically labelled pixels
        self.observation_space = gym.spaces.Dict(
            {
                "health": gym.spaces.Box(low=1, high=5, dtype=np.uint8),
                "ammo": gym.spaces.Box(low=0, high=20, dtype=np.uint8),
                "field": gym.spaces.Box(
                    low=np.zeros(shape), high=6 * np.ones(shape), dtype=np.uint8
                ),
            }
        )

    def reset(self, seed=None):
        # reset the game environment
        # TODO randomly populate the game
        super().reset(seed=seed)

        # TODO agents needs to be randomly placed

        obs = self.game.draw()
        pass

    def step(self, action):
        self.game.step()
        obs = self.game.draw()
        # step the game forward in time, extracting the observation space

        info = None
        return obs, reward, terminated, truncated, info


def main():
    pygame.init()

    game = TagGame(SCREEN_SHAPE, display=True)
    game.loop()


main()
