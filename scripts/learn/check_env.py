import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import stable_baselines3
from stable_baselines3.common.env_checker import check_env

import shadows

import IPython


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="Environment name.")
    parser.add_argument("--seed", default=0, help="Random seed.")
    args = parser.parse_args()

    # make the env and check that it is okay
    env = gym.make(args.env, render_mode="human")
    obs, info = env.reset(seed=args.seed)
    check_env(env)

    # try some random actions
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obj, info = env.reset()


if __name__ == "__main__":
    main()
