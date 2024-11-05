#!/usr/bin/env python3
import pygame
from shoot import TagGame

SCREEN_SHAPE = (200, 200)


def main():
    pygame.init()
    game = TagGame(SCREEN_SHAPE, display=True)
    game.loop()


main()
