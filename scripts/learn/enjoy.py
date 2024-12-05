"""Enjoy a trained agent."""
import argparse

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

import shadows

ENV_KWARGS = {"render_mode": "human"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="Environment name.")
    parser.add_argument("model", help="Path to the trained model.")
    args = parser.parse_args()

    # load the trained agent
    env = make_vec_env(args.env, env_kwargs=ENV_KWARGS)
    env = VecTransposeImage(env)
    model = PPO.load(args.model, env=env)

    # enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    main()
