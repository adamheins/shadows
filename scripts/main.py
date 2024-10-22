#!/usr/bin/env python3
import pygame
import pygame.gfxdraw
import numpy as np

import IPython


SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
DIAG = np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
SCREEN_VERTS = np.array(
    [[0, 0], [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], [0, SCREEN_HEIGHT]]
)

DELAY = 10  # milliseconds

MOVE_STEP = 2


class Circle:
    def __init__(self, position, radius, color):
        self.position = position
        self.radius = radius
        self.color = color

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position, self.radius)


class Projectile(Circle):
    def __init__(self, position, velocity, radius, color):
        self.velocity = velocity
        super().__init__(position, radius, color)


def orth(v):
    return np.array([v[1], -v[0]])


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
    def __init__(self, x, y, w, h, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.rect = pygame.Rect(x, y, w, h)

        # vertices of the box
        self.verts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

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

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)


class Controller:
    def __init__(self):
        pass


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))

        self.keys_down = set()

        # TODO put in Controller
        self.player = Circle(position=[200, 200], radius=10, color=(255, 0, 0))
        self.projectiles = []

        # TODO
        self.obstacles = [
            Obstacle(300, 300, 100, 100, color=(0, 0, 0)),
            Obstacle(500, 300, 100, 20, color=(0, 0, 0)),
        ]

        pygame.display.flip()

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.player.draw(self.screen)

        for projectile in self.projectiles:
            projectile.draw(self.screen)

        for obstacle in self.obstacles:
            # obstacle.draw(self.screen)
            # v1, v2 = obstacle._compute_witness_vertices(self.player.position)
            # pygame.draw.circle(self.screen, (0, 0, 255), v1, 3)
            # pygame.draw.circle(self.screen, (0, 255, 0), v2, 3)

            ps = obstacle.compute_occlusion(self.player.position)
            # pygame.draw.polygon(self.screen, (100, 100, 100), ps)
            pygame.gfxdraw.aapolygon(self.screen, ps, (100, 100, 100))
            pygame.gfxdraw.filled_polygon(self.screen, ps, (100, 100, 100))
            # pygame.draw.aalines(self.screen, color=(100, 100, 100), closed=True, points=ps)
            obstacle.draw(self.screen)

            # for w in ws:
            #     pygame.draw.circle(self.screen, (0, 0, 255), w, 3)
            # for e in es:
            #     pygame.draw.circle(self.screen, (0, 0, 255), w, 3)

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
            move_step = np.zeros(2)
            if pygame.K_d in self.keys_down:
                move_step[0] += 1
            if pygame.K_a in self.keys_down:
                move_step[0] -= 1
            if pygame.K_w in self.keys_down:
                move_step[1] -= 1
            if pygame.K_s in self.keys_down:
                move_step[1] += 1
            norm = np.linalg.norm(move_step)
            if norm > 0:
                move_step = MOVE_STEP * move_step / norm
            self.player.position += move_step

            if target is not None:
                # TODO make a new projectile with appropriate direction and
                # velocity
                norm = np.linalg.norm(target - self.player.position)
                if norm > 0:
                    direction = (target - self.player.position) / norm
                    projectile = Projectile(
                        position=self.player.position.copy(),
                        velocity=10 * direction,
                        radius=3,
                        color=(0, 0, 0),
                    )
                self.projectiles.append(projectile)

            for projectile in self.projectiles:
                projectile.position += projectile.velocity
                if (
                    projectile.position[0] < -projectile.radius
                    or projectile.position[1] < -projectile.radius
                    or projectile.position[0] > SCREEN_WIDTH + projectile.radius
                    or projectile.position[1] > SCREEN_HEIGHT + projectile.radius
                ):
                    pass
                # TODO delete if outside the screen

            self.draw()
            pygame.time.wait(DELAY)


def main():
    pygame.init()

    game = Game()
    game.loop()


main()
