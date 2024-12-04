#!/usr/bin/env python3
import argparse
import pygame

from stable_baselines3 import PPO
from shadows import TagGame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to the trained model.")
    args = parser.parse_args()

    model = None
    if args.model is not None:
        model = PPO.load(args.model)

    pygame.init()
    game = TagGame(display=True, model=model)
    game.loop()


main()
