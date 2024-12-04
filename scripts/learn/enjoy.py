import argparse
import os
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage


ENV_ID = "Simple-v0"
ENV_KWARGS = {"grayscale": True, "render_mode": "human"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the trained model.")
    args = parser.parse_args()

    # load the trained agent
    env = make_vec_env(ENV_ID, env_kwargs=ENV_KWARGS)
    env = VecTransposeImage(env)
    model = DQN.load(args.model, env=env)

    # enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(5000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    main()
