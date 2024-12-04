#!/usr/bin/env python3
import pygame
from shoot import TagGame

SCREEN_SHAPE = (200, 200)


def main():
    pygame.init()
    game = TagGame(display=True)
    game.loop()


main()
