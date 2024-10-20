#!/usr/bin/env python3
import pygame
import numpy as np


SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640

DELAY = 10  # milliseconds

MOVE_STEP = 2


class Circle:
    def __init__(self, position, radius, color):
        self.position = position
        self.radius = radius
        self.color = color

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position, self.radius, 0)


class Projectile(Circle):
    def __init__(self, position, velocity, radius, color):
        self.velocity = velocity
        super().__init__(position, radius, color)


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
        self.obstacles = []

        pygame.display.flip()

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.player.draw(self.screen)
        for projectile in self.projectiles:
            projectile.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
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
