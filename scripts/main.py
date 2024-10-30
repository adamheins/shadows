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


class DefaultAIPolicy:
    def __init__(self, agent_id, player_id, agents, obstacles):
        self.agent = agents[agent_id]
        self.player = agents[player_id]

        self.agents = agents
        self.obstacles = obstacles

    def step(self, dt):
        target = self.player.position
        reload = False

        # move toward the player if they are not too close by
        delta = self.player.position - self.agent.position
        dist = np.linalg.norm(delta)
        if dist <= 3 * self.player.radius:
            velocity = np.zeros(2)
        else:
            velocity = PLAYER_VELOCITY * delta / dist

        # check if player is behind an obstacle, in which case don't shoot at
        # them
        segment = Segment(self.agent.position, self.player.position)
        for obstacle in self.obstacles:
            if segment_rect_intersect(segment, obstacle):
                target = None

        # reload if not shooting and clip is low
        if target is None and self.agent.ammo <= 5:
            reload = True

        return {
            self.agent.id: Action(
                velocity=velocity, target=target, reload=False
            )
        }


class Game:
    def __init__(self, shape, display=True):
        self.display = display
        self.font = pygame.font.SysFont(None, 28)
        self.ammo_text = Text(
            text=f"Ammo: 0",
            font=self.font,
            position=(20, shape[1] - 60),
            color=(0, 0, 0),
        )
        self.health_text = Text(
            text="Health: 0",
            font=self.font,
            position=(20, shape[1] - 30),
            color=(0, 0, 0),
        )

        self.texts = [self.ammo_text, self.health_text]

        if self.display:
            self.screen = pygame.display.set_mode(shape)
        else:
            self.screen = pygame.Surface(shape)
        self.clock = pygame.time.Clock()
        self.screen_rect = AARect(0, 0, shape[0], shape[1])

        self.keys_down = set()

        self.projectiles = {}

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
        self.enemies = [Agent.enemy(position=[400, 400])]
        self.agents = {enemy.id: enemy for enemy in self.enemies}
        self.agents[self.player.id] = self.player

        self.enemy_policy = DefaultAIPolicy(
            self.enemies[0].id, self.player.id, self.agents, self.obstacles
        )

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)

        for projectile in self.projectiles.values():
            projectile.draw(self.screen)

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
            # text
            if self.player.reload_ticks > 0:
                self.ammo_text.update(text=f"Reloading...", color=(255, 0, 0))
            else:
                self.ammo_text.update(text=f"Ammo: {self.player.ammo}", color=(0, 0, 0))

            if self.player.health > 2:
                self.health_text.update(text=f"Health: {self.player.health}")
            else:
                self.health_text.update(
                    text=f"Health: {self.player.health}", color=(255, 0, 0)
                )

            for text in self.texts:
                text.draw(self.screen)

            pygame.display.flip()
        else:
            raw = np.array(pygame.PixelArray(self.screen))
            rgb = np.array([raw >> 16, raw >> 8, raw]) & 0xFF
            rgb = np.moveaxis(rgb, 0, -1)
            return rgb

    def step(self, actions):
        """Step the game forward in time."""
        for agent_id, agent in self.agents.items():
            if agent_id in actions:
                action = actions[agent_id]

                if action.target is not None:
                    projectile = agent.shoot(action.target)
                    if projectile is not None:
                        self.projectiles[projectile.id] = projectile

                if action.reload:
                    agent.reload()

                agent.move(action.velocity)

        # agents cannot walk off the screen and into obstacles
        for agent in self.agents.values():
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

        # process projectiles
        projectiles_to_remove = set()
        agents_to_remove = set()
        for idx, projectile in self.projectiles.items():
            # projectile has left the screen
            if not point_in_rect(projectile.position, self.screen_rect):
                projectiles_to_remove.add(idx)
                continue

            # path of projectile's motion over the timestep
            segment = projectile.path(TIMESTEP)

            # check for collision with obstacle
            obs_dist = np.inf
            for obstacle in self.obstacles:
                if segment_rect_intersect(segment, obstacle):
                    d = segment_rect_intersect_dist(segment, obstacle)
                    obs_dist = min(obs_dist, d)
                    projectiles_to_remove.add(idx)

            # check for collision with an agent
            for agent_id, agent in self.agents.items():

                # agent cannot be hit by its own bullets
                if agent_id == projectile.agent_id:
                    continue

                # check for collision with the bullet's path
                # TODO segment_segment_dist might be better here
                circle = agent.circle()
                if circle_segment_intersect(circle, segment):
                    # if the projectile hit an obstacle first, then the agent
                    # is fine
                    d = circle_segment_intersect_dist(circle, segment)
                    if d > obs_dist:
                        continue

                    projectiles_to_remove.add(idx)

                    agent.velocity += 0.5 * PLAYER_VELOCITY * unit(projectile.velocity)
                    agent.health -= 1
                    if agent.health <= 0:
                        agents_to_remove.add(agent_id)

        # remove projectiles that have hit something
        for idx in projectiles_to_remove:
            self.projectiles.pop(idx)

        # remove agents that have died
        for idx in agents_to_remove:
            print("dead!")
            self.agents.pop(idx)

        # integrate the game state forward in time
        for projectile in self.projectiles.values():
            projectile.step(TIMESTEP)

        # TODO don't like that the obstacles gets passed in here
        for agent in self.agents.values():
            agent.step(TIMESTEP)

    def loop(self):
        while True:

            target = None
            reload = False

            # process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    self.keys_down.add(event.key)
                elif event.type == pygame.KEYUP:
                    self.keys_down.discard(event.key)
                    reload = event.key == pygame.K_r
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    target = np.array(pygame.mouse.get_pos())

            # respond to events
            velocity = np.zeros(2)
            linvel = 0
            angvel = 0
            if pygame.K_d in self.keys_down:
                # velocity[0] += 1
                angvel -= 1
            if pygame.K_a in self.keys_down:
                # velocity[0] -= 1
                angvel += 1
            if pygame.K_w in self.keys_down:
                linvel += 1
                # velocity[1] -= 1
            if pygame.K_s in self.keys_down:
                # velocity[1] += 1
                linvel -= 0.5

            # norm = np.linalg.norm(velocity)
            # if norm > 0:
            #     velocity = PLAYER_VELOCITY * velocity / norm
            # TODO need to wrap to pi better
            self.player.angle += TIMESTEP * 5 * angvel
            if self.player.angle > np.pi:
                self.player.angle -= 2 * np.pi
            elif self.player.angle < -np.pi:
                self.player.angle += 2 * np.pi

            if np.abs(linvel) > 0:
                velocity = linvel * PLAYER_VELOCITY * rotmat(self.player.angle) @ [1, 0]

            # TODO I wonder if the action should just be the command ("go
            # left") rather than the actual velocity vector (the latter has
            # more DOFs than are actually available)
            # actions = self.enemy_policy.step(TIMESTEP)
            actions = {}
            actions[self.player.id] = Action(velocity, target, reload)

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

    game = Game(SCREEN_SHAPE)
    game.loop()


main()
