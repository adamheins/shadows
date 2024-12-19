"""Enjoy a trained agent."""

import argparse
import os
import yaml
import pygame

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

import shadows

ALGOS = {
    "dqn": DQN,
    "ppo": PPO,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Path to the logs directory.")
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic policy."
    )
    args = parser.parse_args()

    info_path = os.path.join(args.log_dir, "info.yaml")
    with open(info_path) as f:
        info = yaml.safe_load(f)
    env_name = info["env"]
    algo_name = info["algo"]

    model_path = os.path.join(args.log_dir, env_name + ".zip")

    # load the trained agent
    env = make_vec_env(env_name, env_kwargs={"render_mode": "human"})
    env = VecTransposeImage(env)
    model = ALGOS[algo_name].load(model_path, env=env)

    # enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    steps = 0
    for i in range(5000):
        action, _states = model.predict(obs, deterministic=args.deterministic)
        obs, rewards, dones, info = vec_env.step(action)

        if dones[0]:
            print(f"steps = {steps}")
            steps = 0
        else:
            steps += 1

        # allow nice shutdown
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        vec_env.render("human")


if __name__ == "__main__":
    main()
