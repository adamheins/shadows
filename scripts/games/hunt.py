#!/usr/bin/env python3
import argparse
import pygame

import shadows


def main():
    pygame.init()
    game = shadows.HuntGame(display=True)
    game.loop()


main()
