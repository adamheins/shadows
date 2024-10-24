#!/usr/bin/env python3
import pygame
import pygame.gfxdraw
import numpy as np
import pymunk
import pymunk.pygame_util


SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
DIAG = np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
SCREEN_VERTS = np.array(
    [[0, 0], [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], [0, SCREEN_HEIGHT]]
)

FRAMERATE = 60
PHYSICS_STEP_PER_TICK = 10
TIMESTEP = 1.0 / (FRAMERATE * PHYSICS_STEP_PER_TICK)

PLAYER_VELOCITY = 200  # px per second
BULLET_VELOCITY = 1000

PLAYER_MASS = 10
BULLET_MASS = 0.1

# TODO shouldn't this be mapped to real time?
SHOT_COOLDOWN_TICKS = 10


def orth(v):
    """Generate a 2D orthogonal to v."""
    return np.array([v[1], -v[0]])


def unit(v):
    """Normalize to a unit vector."""
    norm = np.linalg.norm(v)
    if np.isclose(norm, 0):
        return np.zeros_like(v)
    return v / norm


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


class Obstacle:
    def __init__(self, space, x, y, w, h, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.rect = pygame.Rect(x, y, w, h)

        # vertices of the box
        self.verts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

        self.shape = pymunk.Poly(body=space.static_body, vertices=self.verts.tolist())
        self.shape.collision_type = 0
        space.add(self.shape)

    def _compute_witness_vertices(self, point, tol=1e-3):
        right = None
        left = None
        for i in range(len(self.verts)):
            vert = self.verts[i]
            delta = vert - point
            normal = orth(delta)
            dists = (self.verts - point) @ normal
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

    def draw(self, surface, viewpoint=None):
        if viewpoint is not None:
            ps = self._compute_occlusion(viewpoint)
            pygame.gfxdraw.aapolygon(surface, ps, (100, 100, 100))
            pygame.gfxdraw.filled_polygon(surface, ps, (100, 100, 100))

        pygame.draw.rect(surface, self.color, self.rect)


class Entity:
    """Pymunk entity."""

    def __init__(self, space, body, shape):
        self.body = body
        self.shape = shape
        space.add(body, shape)

    @property
    def id(self):
        return self.body.id

    @property
    def position(self):
        return self.body.position

    @property
    def velocity(self):
        return self.body.velocity

    @velocity.setter
    def velocity(self, value):
        self.body.velocity = tuple(value)

    def remove(self):
        space = self.body.space
        space.remove(self.shape, self.body)


class Agent(Entity):
    def __init__(self, space, position, color):
        body = pymunk.Body(PLAYER_MASS, float("inf"))
        body.position = position

        shape = pymunk.Circle(body, 10, (0, 0))
        shape.color = color
        shape.collision_type = 1

        super().__init__(space, body, shape)

        self.shot_cooldown = 0

    def draw(self, surface):
        # pygame.gfxdraw.aacircle(surface, int(self.position[0]), int(self.position[1]),
        #                         int(self.shape.radius), self.shape.color)
        pygame.draw.circle(surface, self.shape.color, self.position, self.shape.radius)

    def tick(self):
        self.shot_cooldown = max(0, self.shot_cooldown - 1)

    def shoot(self, target):
        if self.shot_cooldown > 0:
            return None

        norm = np.linalg.norm(target - self.position)
        if norm > 0:
            direction = (target - self.position) / norm
            self.shot_cooldown = SHOT_COOLDOWN_TICKS
            return Projectile(
                space=self.body.space,
                position=self.position,
                velocity=BULLET_VELOCITY * direction,
                agent_id=self.id,
            )
        else:
            return None


class Projectile(Entity):
    def __init__(self, space, position, velocity, agent_id):
        self.agent_id = agent_id

        body = pymunk.Body(BULLET_MASS, float("inf"))
        body.position = tuple(position)
        body.velocity = tuple(velocity)

        shape = pymunk.Circle(body, 3, (0, 0))
        shape.color = (0, 0, 0, 255)
        shape.collision_type = 2

        super().__init__(space, body, shape)

    def draw(self, surface):
        pygame.draw.circle(
            surface, self.shape.color, self.body.position, self.shape.radius
        )


class Blood:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class Game:
    def __init__(self):
        self.space = pymunk.Space()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))
        self.screen_rect = self.screen.get_bounding_rect()
        pygame.display.flip()

        self.keys_down = set()

        self.projectiles = {}

        self.obstacles = [
            Obstacle(self.space, 300, 300, 100, 100, color=(0, 0, 0)),
            Obstacle(self.space, 500, 300, 100, 20, color=(0, 0, 0)),
        ]

        self.player = Agent(
            space=self.space, position=[200, 200], color=(255, 0, 0, 255)
        )
        enemies = [Agent(space=self.space, position=[200, 300], color=(0, 0, 255, 255))]

        self.agents = {enemy.body.id: enemy for enemy in enemies}
        self.agents[self.player.body.id] = self.player

        self._shot_cooldown = 0

        self.bloods = []

        def bullet_obstacle_handler(arbiter, space, data):
            # delete bullet when it hits an obstacle
            body_id = arbiter.shapes[1].body.id
            self.projectiles[body_id].remove()
            self.projectiles.pop(body_id)
            return False

        def bullet_agent_handler(arbiter, space, data):
            bullet_id = arbiter.shapes[1].body.id
            agent_id = arbiter.shapes[0].body.id

            bullet = self.projectiles[bullet_id]

            # an agent cannot be hit by its own bullet
            if bullet.agent_id == agent_id:
                return False

            if agent_id == self.player.body.id:
                print("you lose!")
            else:
                agent = self.agents[agent_id]
                agent.remove()
                self.agents.pop(agent_id)

                start = arbiter.contact_point_set.points[0].point_a
                blood = Blood(start, start + 20 * unit(bullet.velocity))
                self.bloods.append(blood)

            return False

        self.space.add_collision_handler(0, 2).begin = bullet_obstacle_handler
        self.space.add_collision_handler(1, 2).begin = bullet_agent_handler

    def draw(self):
        self.screen.fill((255, 255, 255))

        for projectile in self.projectiles.values():
            projectile.draw(self.screen)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen, viewpoint=self.player.position)

        for agent in self.agents.values():
            agent.draw(self.screen)

        for blood in self.bloods:
            pygame.draw.line(self.screen, (255, 0, 0), blood.p1, blood.p2, width=3)

        pygame.display.flip()

    def step(self):
        # TODO we want to be able to step the sim forward once to facilitate
        # learning
        pass

    def loop(self):
        while True:

            target = None

            # process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    self.keys_down.add(event.key)
                elif event.type == pygame.KEYUP:
                    self.keys_down.discard(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    target = np.array(pygame.mouse.get_pos())

            # respond to events
            velocity = np.zeros(2)
            # TODO add remaining checks like this and to the bullet as well
            if (
                pygame.K_d in self.keys_down
                and self.player.position[0] < SCREEN_WIDTH - 10
            ):
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
            self.player.velocity = tuple(velocity)

            # create a new projectile
            if target is not None:
                projectile = self.player.shoot(target)
                if projectile is not None:
                    self.projectiles[projectile.id] = projectile

            # remove projectiles that are outside of the screen
            projectiles_to_remove = []
            for idx, projectile in self.projectiles.items():
                if not self.screen_rect.collidepoint(projectile.body.position):
                    projectiles_to_remove.append(idx)
            for idx in projectiles_to_remove:
                self.projectiles[idx].remove()
                self.projectiles.pop(idx)

            self.draw()

            # physics
            for _ in range(PHYSICS_STEP_PER_TICK):
                self.space.step(TIMESTEP)

            for agent in self.agents.values():
                agent.tick()

            # tick forward
            self.clock.tick(FRAMERATE)


def main():
    pygame.init()

    game = Game()
    game.loop()


main()
