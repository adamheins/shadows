"""Train an agent."""

import argparse
import os
import datetime
import gymnasium as gym
import re
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage

import shadows


TOTAL_TIMESTEPS = 1_000_000
EVAL = True


def linear_schedule(initial_value):
    """Linear learning rate schedule."""

    def func(progress_remaining):
        """Progress will decrease from 1 (beginning) to 0."""
        return progress_remaining * initial_value

    return func


def make_log_dir(env_id):
    pattern = re.compile(f".*{env_id}_([0-9]+)")
    log_dir_root = Path("logs")
    log_dir_root.mkdir(parents=True, exist_ok=True)
    max_count = 0
    for log in log_dir_root.glob(f"{env_id}_*"):
        match = pattern.match(str(log))
        count = int(match.group(1))
        max_count = max(max_count, count)

    new_count = max_count + 1
    log_dir = log_dir_root / f"{env_id}_{new_count:>02}"
    log_dir.mkdir()
    return str(log_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="Environment name.")
    parser.add_argument(
        "--n-envs", type=int, default=8, help="Number of parallel environments."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Number of parallel environments."
    )
    args = parser.parse_args()

    log_dir = make_log_dir(args.env)

    # create environment
    # use VecTransposeImage because SB3 wants channel-first format
    env = make_vec_env(args.env, n_envs=args.n_envs, monitor_dir=log_dir)
    env = VecTransposeImage(env)

    # Instantiate the agent
    # TODO this does not work as well as rl_zoo - why?
    # model = DQN(
    #     policy="CnnPolicy",
    #     env=env,
    #     seed=SEED,
    #     buffer_size=100000,
    #     learning_rate=1e-4,
    #     batch_size=32,
    #     learning_starts=100000,
    #     target_update_interval=1000,
    #     train_freq=4,
    #     gradient_steps=1,
    #     exploration_fraction=0.5,
    #     exploration_final_eps=0.01,
    #     verbose=1,
    # )
    model = PPO(
        env=env,
        seed=args.seed,
        policy="MultiInputPolicy",
        n_steps=128,
        n_epochs=4,
        batch_size=256,
        learning_rate=linear_schedule(2.5e-4),
        clip_range=linear_schedule(0.1),
        vf_coef=0.5,
        ent_coef=0.01,
        verbose=1,
    )

    if EVAL:
        eval_env = make_vec_env(args.env, n_envs=args.n_envs)
        eval_env = VecTransposeImage(eval_env)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=max(25000 // args.n_envs, 1),
            deterministic=True,
            render=False,
        )
    else:
        eval_callback = None

    # train the agent
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=eval_callback
    )

    # save the agent
    model_path = os.path.join(log_dir, args.env)
    model.save(model_path)
    # print(f"saved model to {model_path}")


if __name__ == "__main__":
    main()
