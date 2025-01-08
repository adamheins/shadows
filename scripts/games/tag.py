#!/usr/bin/env python3
import argparse
import pygame

import shadows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--it-model", help="Path to the trained model for 'it' agent.")
    parser.add_argument(
        "--not-it-model", help="Path to the trained model for 'not it' agent."
    )
    parser.add_argument(
        "--algo", default="dqn", help="The algorithm used by the trained models."
    )
    args = parser.parse_args()

    algo = shadows.ALGOS[args.algo.lower()]
    it_model, not_it_model = None, None
    if args.it_model is not None:
        it_model = algo.load(args.it_model)
    if args.not_it_model is not None:
        not_it_model = algo.load(args.not_it_model)

    pygame.init()
    game = shadows.TagGame(display=True, it_model=it_model, not_it_model=not_it_model)
    game.loop()


main()
