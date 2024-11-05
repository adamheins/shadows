#!/usr/bin/env python3
import gymnasium as gym

from shoot import TagItEnv

SCREEN_SHAPE = (200, 200)


def main():
    gym.register(
        id="shoot/TagIt-v0",
        entry_point=TagItEnv,
    )

    env = gym.make("shoot/TagIt-v0", shape=SCREEN_SHAPE)



main()
