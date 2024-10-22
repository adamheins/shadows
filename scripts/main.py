#!/usr/bin/env python3
import pygame
import numpy as np

import IPython


SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
DIAG = np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)

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
# def line_line_intersection(p11, p12, p21, p22):
#     q1 = p2 - p1
#     q2 = p4 - p3
#
#     d = q1[0] * q2[1] - q1[1] * q2[0]
#     if np.isclose(d, 0):
#         pass
#
#     x = (p1[0] * p2[1] - p1[1] * p2[0]) *


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

    def compute_occlusion(self, point, tol=1e-3):
        points = []
        for i in range(len(self.verts)):
            vert = self.verts[i]
            delta = vert - point
            normal = orth(delta)
            dists = (self.verts - point) @ normal
            if np.all(dists >= -tol):
                extra = vert + DIAG * delta / np.linalg.norm(delta)
                points.append(vert)
                points.append(extra)
            elif np.all(dists <= tol):
                extra = vert + DIAG * delta / np.linalg.norm(delta)
                points.append(extra)
                points.append(vert)
        return points

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
            Obstacle(300, 300, 100, 100, color=(100, 100, 100)),
            Obstacle(500, 300, 100, 20, color=(100, 100, 100)),
        ]

        pygame.display.flip()

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.player.draw(self.screen)

        for projectile in self.projectiles:
            projectile.draw(self.screen)

        for obstacle in self.obstacles:
            ps = obstacle.compute_occlusion(self.player.position)
            pygame.draw.polygon(self.screen, (100, 100, 100), ps)
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
