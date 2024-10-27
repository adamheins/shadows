#!/usr/bin/env python3
import pygame
import pygame.gfxdraw
import numpy as np

from shoot import *


SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
SCREEN_VERTS = np.array(
    [[0, 0], [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], [0, SCREEN_HEIGHT]]
)

BACKGROUND_COLOR = (219, 200, 184)
OBSTACLE_COLOR = (0, 0, 0)

FRAMERATE = 60
TIMESTEP = 1.0 / FRAMERATE


def line_screen_edge_intersection(p, v):
    ts = []

    # vertical edges
    if not np.isclose(v[0], 0):
        ts.extend([-p[0] / v[0], (SCREEN_WIDTH - p[0]) / v[0]])

    # horizontal edges
    if not np.isclose(v[1], 0):
        ts.extend([-p[1] / v[1], (SCREEN_HEIGHT - p[1]) / v[1]])

    # return the smallest positive value
    ts = np.array(ts)
    t = np.min(ts[ts >= 0])
    return p + t * v


class Obstacle(AARect):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.color = OBSTACLE_COLOR
        self.rect = pygame.Rect(x, y, w, h)

    def _compute_witness_vertices(self, point, tol=1e-3):
        right = None
        left = None
        for i in range(len(self.vertices)):
            vert = self.vertices[i]
            delta = vert - point
            normal = orth(delta)
            dists = (self.vertices - point) @ normal
            if np.all(dists >= -tol):
                right = vert
            elif np.all(dists <= tol):
                left = vert
            if left is not None and right is not None:
                break
        return right, left

    def _compute_occlusion(self, point, tol=1e-3):
        right, left = self._compute_witness_vertices(point, tol=tol)

        delta_right = right - point
        extra_right = line_screen_edge_intersection(right, delta_right)
        normal_right = orth(delta_right)

        delta_left = left - point
        extra_left = line_screen_edge_intersection(left, delta_left)
        normal_left = orth(delta_left)

        screen_dists = []
        screen_vs = []
        for v in SCREEN_VERTS:
            if -(v - point) @ normal_left < 0:
                continue
            dist = (v - point) @ normal_right
            if dist >= 0:
                if len(screen_dists) > 0 and screen_dists[0] > dist:
                    screen_vs = [v, screen_vs[0]]
                    break
                else:
                    screen_dists.append(dist)
                    screen_vs.append(v)

        return [right, extra_right] + screen_vs + [extra_left, left]

    def compute_collision_normal(self, point, r):
        if point[0] >= self.x and point[0] <= self.x + self.w:
            if point[1] >= self.y - r and point[1] <= self.y + 0.5 * self.h:
                return (0, -1)
            elif point[1] <= self.y + self.h + r and point[1] > self.y + 0.5 * self.h:
                return (0, 1)
        elif point[1] >= self.y and point[1] <= self.y + self.h:
            if point[0] >= self.x - r and point[0] <= self.x + 0.5 * self.w:
                return (-1, 0)
            elif point[0] <= self.x + self.w + r and point[0] > self.x + 0.5 * self.w:
                return (1, 0)
        else:
            # closest point is one of the vertices
            v2 = np.sum((self.vertices - point) ** 2, axis=1)
            idx = np.argmin(v2)
            if v2[idx] <= r**2:
                return unit(point - self.vertices[idx, :])
        return None

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def draw_occlusion(self, surface, viewpoint):
        ps = self._compute_occlusion(viewpoint)
        pygame.gfxdraw.aapolygon(surface, ps, (100, 100, 100))
        pygame.gfxdraw.filled_polygon(surface, ps, (100, 100, 100))


class Game:
    def __init__(self):
        self.font = pygame.font.SysFont(None, 28)
        self.ammo_text = Text(
            text=f"Ammo: {CLIP_SIZE}",
            font=self.font,
            position=(20, SCREEN_HEIGHT - 60),
            color=(0, 0, 0),
        )
        self.health_text = Text(
            text="Health: 5",
            font=self.font,
            position=(20, SCREEN_HEIGHT - 30),
            color=(0, 0, 0),
        )

        self.texts = [self.ammo_text, self.health_text]

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.screen.fill(BACKGROUND_COLOR)
        self.screen_rect = self.screen.get_bounding_rect()
        pygame.display.flip()

        self.keys_down = set()

        self.projectiles = {}

        self.obstacles = [
            Obstacle(80, 80, 80, 80),
            Obstacle(0, 230, 160, 40),
            Obstacle(80, 340, 80, 80),
            Obstacle(250, 80, 40, 220),
            Obstacle(290, 80, 100, 40),
            Obstacle(390, 260, 110, 40),
            Obstacle(250, 380, 200, 40),
        ]

        self.player = Agent(position=[200, 200], color=(255, 0, 0, 255))
        enemies = [Agent(position=[200, 300], color=(0, 0, 255, 255))]

        self.agents = {enemy.id: enemy for enemy in enemies}
        self.agents[self.player.id] = self.player

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)

        for projectile in self.projectiles.values():
            projectile.draw(self.screen)

        for agent in self.agents.values():
            agent.draw(self.screen)

        for obstacle in self.obstacles:
            obstacle.draw_occlusion(self.screen, viewpoint=self.player.position)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

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

        # NOTE: this is how to extract the RGB values
        # raw = np.array(pygame.PixelArray(self.screen))
        # rgb = np.array([raw >> 16, raw >> 8, raw]) & 0xff
        # rgb = np.moveaxis(rgb, 0, -1)

    def step(self, actions):
        """Step the game forward in time."""
        for agent_id, agent in self.agents.items():
            if agent_id in actions:
                action = actions[agent_id]

                if action.target is not None:
                    projectile = self.player.shoot(action.target)
                    if projectile is not None:
                        self.projectiles[projectile.id] = projectile

                if action.reload:
                    agent.reload()

                agent.move(action.velocity)

        # agents cannot walk off the screen and into obstacles
        for agent in self.agents.values():
            v = agent.velocity

            # don't leave the screen
            if agent.position[0] >= SCREEN_WIDTH - agent.radius:
                v[0] = min(0, v[0])
            elif agent.position[0] <= agent.radius:
                v[0] = max(0, v[0])
            if agent.position[1] >= SCREEN_HEIGHT - agent.radius:
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
            if not self.screen_rect.collidepoint(projectile.position):
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
            if pygame.K_d in self.keys_down:
                velocity[0] += 1
            if pygame.K_a in self.keys_down:
                velocity[0] -= 1
            if pygame.K_w in self.keys_down:
                velocity[1] -= 1
            if pygame.K_s in self.keys_down:
                velocity[1] += 1
            norm = np.linalg.norm(velocity)
            if norm > 0:
                velocity = PLAYER_VELOCITY * velocity / norm

            # TODO I wonder if the action should just be the command ("go
            # left") rather than the actual velocity vector (the latter has
            # more DOFs than are actually available)
            actions = {self.player.id: Action(velocity, target, reload)}

            self.step(actions)
            self.draw()
            self.clock.tick(FRAMERATE)


def main():
    pygame.init()

    game = Game()
    game.loop()


main()
