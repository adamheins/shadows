#!/usr/bin/env python3
import gymnasium as gym
import numpy as np

import shoot


def main():
    env = gym.make("TagIt-v0")
    # env = gym.wrappers.GrayscaleObservation(env)
    # env = gym.wrappers.ResizeObservation(env, (50, 50))
    # print(env.observation_space.shape)

    game = shoot.TagGame(display=False)
    game.draw()
    # rgb = game.rgb()

    import IPython
    IPython.embed()


main()
