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

PHYSICS_STEP_PER_FRAME = 10
FRAMERATE = 60
TIMESTEP = 1.0 / (FRAMERATE * PHYSICS_STEP_PER_FRAME)

PLAYER_VELOCITY = 100  # px per second
BULLET_VELOCITY = 1000


class Circle:
    def __init__(self, position, radius, color):
        self.position = position
        self.radius = radius
        self.color = color

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position, self.radius)


def orth(v):
    return np.array([v[1], -v[0]])


def unit(v):
    norm = np.linalg.norm(v)
    if np.isclose(v, 0):
        return np.zeros_like(v)
    return v / norm


# TODO need this to properly do the occlusions
def line_line_intersection(p1, v1, p2, v2):
    # parallel
    if np.isclose(orth(v1) @ v2, 0):
        return None

    A = np.array([[-v1 @ v1, v1 @ v2], [-v1 @ v2, v2 @ v2]])
    b = np.array([v1 @ (p1 - p2), v2 @ (p1 - p2)])
    t = np.linalg.solve(A, b)
    return p1 + t[0] * v1, ts


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

    def compute_occlusion(self, point, tol=1e-3):
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
            ps = self.compute_occlusion(viewpoint)
            pygame.gfxdraw.aapolygon(surface, ps, (100, 100, 100))
            pygame.gfxdraw.filled_polygon(surface, ps, (100, 100, 100))

        pygame.draw.rect(surface, self.color, self.rect)


class Player:
    def __init__(self, space, position):
        # self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body = pymunk.Body(5, float("inf"))
        self.body.position = position

        self.shape = pymunk.Circle(self.body, 10, (0, 0))
        self.shape.color = (255, 0, 0, 255)
        self.shape.collision_type = 1

        space.add(self.body, self.shape)

    @property
    def position(self):
        return self.body.position

    @property
    def velocity(self):
        return self.body.velocity

    @velocity.setter
    def velocity(self, value):
        self.body.velocity = tuple(value)

    def draw(self, surface):
        # pygame.gfxdraw.aacircle(surface, int(self.position[0]), int(self.position[1]),
        #                         int(self.shape.radius), self.shape.color)
        pygame.draw.circle(surface, self.shape.color, self.position, self.shape.radius)


class Projectile:
    def __init__(self, space, position, velocity):
        self.body = pymunk.Body(0.1, float("inf"))
        self.body.position = tuple(position)
        self.body.velocity = tuple(velocity)

        self.shape = pymunk.Circle(self.body, 3, (0, 0))
        self.shape.color = (0, 0, 0, 255)
        self.shape.collision_type = 2

        space.add(self.body, self.shape)

    def draw(self, surface):
        pygame.draw.circle(surface, self.shape.color, self.body.position, self.shape.radius)


class Controller:
    def __init__(self):
        pass


class Game:
    def __init__(self):
        self.space = pymunk.Space()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))
        pygame.display.flip()

        self.keys_down = set()

        # TODO put in Controller
        self.projectiles = []

        # TODO
        self.obstacles = [
            Obstacle(self.space, 300, 300, 100, 100, color=(0, 0, 0)),
            Obstacle(self.space, 500, 300, 100, 20, color=(0, 0, 0)),
        ]

        self.player = Player(space=self.space, position=[200, 200])

        def bullet_obstacle_handler(arbiter, space, data):
            # print(arbiter.shapes[1].body.velocity)
            # v = arbiter.shapes[1].body.velocity
            # n = arbiter.normal
            # t = orth(n)
            # # TODO apparently this does not work
            # arbiter.shapes[1].body.velocity = tuple((v @ t) * t)
            print("pow")
            # TODO need to remove the projectile itself too...
            space.remove(arbiter.shapes[1].body, arbiter.shapes[1])
            return False

        self.space.add_collision_handler(0, 2).begin = bullet_obstacle_handler
        # self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def draw(self):
        self.screen.fill((255, 255, 255))

        self.player.draw(self.screen)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen, viewpoint=self.player.position)

        for projectile in self.projectiles:
            projectile.draw(self.screen)

        pygame.display.flip()

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
            self.player.velocity = tuple(velocity)

            if target is not None:
                # TODO make a new projectile with appropriate direction and
                # velocity
                norm = np.linalg.norm(target - self.player.position)
                if norm > 0:
                    direction = (target - self.player.position) / norm

                    projectile = Projectile(
                        space=self.space,
                        position=self.player.position,
                        velocity=BULLET_VELOCITY * direction,
                    )
                self.projectiles.append(projectile)
            #
            # for projectile in self.projectiles:
            #     projectile.position += projectile.velocity
            #     if (
            #         projectile.position[0] < -projectile.radius
            #         or projectile.position[1] < -projectile.radius
            #         or projectile.position[0] > SCREEN_WIDTH + projectile.radius
            #         or projectile.position[1] > SCREEN_HEIGHT + projectile.radius
            #     ):
            #         pass
            #     # TODO delete if outside the screen
            #
            self.draw()

            # physics
            for i in range(PHYSICS_STEP_PER_FRAME):
                self.space.step(TIMESTEP)

            # graphics
            self.clock.tick(FRAMERATE)

            # print(self.clock.get_fps())
            # pygame.time.wait(DELAY)


def main():
    pygame.init()

    game = Game()
    game.loop()


main()
