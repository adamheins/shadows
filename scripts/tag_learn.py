#!/usr/bin/env python3
import gymnasium as gym

from shoot import TagItEnv

SCREEN_SHAPE = (200, 200)


def main():
    # gym.register(
    #     id="shoot/TagIt-v0",
    #     entry_point=TagItEnv,
    # )

    env = gym.make("TagIt-v0")
    # env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, (50, 50))
    print(env.observation_space.shape)

    # IPython.embed()


main()
