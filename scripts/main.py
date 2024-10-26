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

PLAYER_VELOCITY = 200  # px per second
BULLET_VELOCITY = 1000

PLAYER_MASS = 10
BULLET_MASS = 0.1

CLIP_SIZE = 20  # bullets per reload
SHOT_COOLDOWN_TICKS = 1
RELOAD_TICKS = 2 * FRAMERATE


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

    # TODO move to collision code
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


class Entity:
    """Pymunk entity."""

    _current_id = 0

    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

        self.id = Entity._current_id
        Entity._current_id += 1


class Agent(Entity):
    def __init__(self, position, color):
        super().__init__(position, np.zeros(2))

        self.color = color
        self.radius = 10

        self.health = 5
        self.ammo = CLIP_SIZE
        self.shot_cooldown = 0
        self.reload_ticks = 0

    def draw(self, surface):
        pygame.gfxdraw.aacircle(
            surface,
            int(self.position[0]),
            int(self.position[1]),
            int(self.radius),
            self.color,
        )
        pygame.gfxdraw.filled_circle(
            surface,
            int(self.position[0]),
            int(self.position[1]),
            int(self.radius),
            self.color,
        )

    def move(self, velocity):
        self.velocity += velocity

    def step(self, dt, obstacles):
        self.shot_cooldown = max(0, self.shot_cooldown - 1)

        # if done reloading, fill clip
        if self.reload_ticks == 1:
            self.ammo = CLIP_SIZE
        self.reload_ticks = max(0, self.reload_ticks - 1)

        v = self.velocity

        # don't leave the screen
        if self.position[0] >= SCREEN_WIDTH - self.radius:
            v[0] = min(0, v[0])
        elif self.position[0] <= self.radius:
            v[0] = max(0, v[0])
        if self.position[1] >= SCREEN_HEIGHT - self.radius:
            v[1] = min(0, v[1])
        elif self.position[1] <= self.radius:
            v[1] = max(0, v[1])

        # don't walk into an obstacle
        for obstacle in obstacles:
            n = obstacle.compute_collision_normal(self.position, self.radius)
            if n is not None and n @ v < 0:
                t = orth(n)
                v = (t @ v) * t

        self.position += dt * v
        self.velocity = np.zeros(2)

    def reload(self):
        self.reload_ticks = RELOAD_TICKS

    def shoot(self, target):
        if self.shot_cooldown > 0 or self.reload_ticks > 0:
            return None

        norm = np.linalg.norm(target - self.position)
        if norm > 0:
            direction = (target - self.position) / norm
            self.shot_cooldown = SHOT_COOLDOWN_TICKS
            self.ammo -= 1
            if self.ammo == 0:
                self.reload()
            self.velocity -= 0.5 * PLAYER_VELOCITY * direction
            return Projectile(
                position=self.position.copy(),
                velocity=BULLET_VELOCITY * direction,
                agent_id=self.id,
            )
        else:
            return None

    def circle(self):
        return Circle(self.position, self.radius)


class Projectile(Entity):
    def __init__(self, position, velocity, agent_id):
        super().__init__(position, velocity)
        self.agent_id = agent_id
        self.color = (0, 0, 0, 255)
        self.radius = 3

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position, self.radius)

    def path(self, dt):
        return Segment(self.position, self.position + dt * self.velocity)

    def step(self, dt):
        self.position += dt * self.velocity


class AgentAction:
    def __init__(self, velocity, target):
        self.velocity = velocity
        self.target = target


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

                agent.move(action.velocity)

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

            # check for collisions with obstacles
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

        # integrate forward
        for projectile in self.projectiles.values():
            projectile.step(TIMESTEP)

        # TODO don't like that the obstacles gets passed in here
        for agent in self.agents.values():
            agent.step(TIMESTEP, self.obstacles)

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

            # TODO I guess reload can be dumped into the action as well
            # TODO I wonder if the action should just be the command ("go
            # left") rather than the actual velocity vector (the latter has
            # more DOFs than are actually available)
            if reload:
                self.player.reload()

            actions = {self.player.id: AgentAction(velocity, target)}

            self.step(actions)
            self.draw()
            self.clock.tick(FRAMERATE)


def main():
    pygame.init()

    game = Game()
    game.loop()


main()
