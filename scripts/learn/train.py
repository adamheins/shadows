"""Train an agent."""

import argparse
import os
import datetime
import gymnasium as gym
import re
from pathlib import Path
import yaml
import numpy as np

from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
from stable_baselines3.common.noise import NormalActionNoise

from sb3_contrib import QRDQN

import shadows


# TOTAL_TIMESTEPS = 5_000_000

EVAL = True
EVAL_FREQ = 50_000
N_EVAL_ENVS = 1
N_EVAL_EPISODES = 5

N_STACK = 1


def linear_schedule(initial_value):
    """Linear learning rate schedule."""

    def func(progress_remaining):
        """Progress will decrease from 1 (beginning) to 0."""
        return progress_remaining * initial_value

    return func


def make_log_dir(env_name, log_dir_root):
    pattern = re.compile(f".*{env_name}_([0-9]+)")
    log_dir_root = Path(log_dir_root)

    log_dir_root.mkdir(parents=True, exist_ok=True)
    max_count = 0
    for log in log_dir_root.glob(f"{env_name}_*"):
        match = pattern.match(str(log))
        count = int(match.group(1))
        max_count = max(max_count, count)

    new_count = max_count + 1
    log_dir = log_dir_root / f"{env_name}_{new_count:>02}"
    log_dir.mkdir()

    return str(log_dir)


def make_model(algo_name, env, seed, trained_agent=None):
    kwargs = dict(policy="MultiInputPolicy", env=env, seed=seed, verbose=1)

    algo_name = algo_name.lower()
    if algo_name == "dqn" or algo_name == "qrdqn":
        if algo_name == "dqn":
            algo = shadows.DQN
        else:
            algo = QRDQN
        kwargs.update(
            dict(
                buffer_size=100_000,
                learning_rate=1e-4,
                batch_size=32,
                learning_starts=100_000,
                target_update_interval=1000,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.5,
                exploration_final_eps=0.01,
                double_q=True,
            )
        )
    elif algo_name == "ppo":
        algo = PPO
        kwargs.update(
            dict(
                n_steps=512,
                n_epochs=4,
                batch_size=256,
                learning_rate=linear_schedule(2.5e-4),
                clip_range=linear_schedule(0.1),
                vf_coef=0.5,
                ent_coef=0.01,
            )
        )
    elif algo_name == "sac":
        algo = SAC
        kwargs.update(
            dict(
                learning_rate=7.3e-4,
                buffer_size=100_000,
                batch_size=256,
                ent_coef="auto",
                gamma=0.98,
                tau=0.02,
                train_freq=64,
                gradient_steps=64,
                learning_starts=100_000,
                use_sde=True,
                # policy_kwargs=dict(net_arch=[400, 300]),
            )
        )
    else:
        raise ValueError(f"unknown model type: {algo_name}")

    if trained_agent is None:
        return algo(**kwargs)
    return algo.load(path=trained_agent, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="Environment name.")
    parser.add_argument(
        "--n-envs", type=int, default=1, help="Number of parallel environments."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Number of parallel environments."
    )
    parser.add_argument("-L", "--log-dir", default="logs", help="Logging directory.")
    parser.add_argument(
        "-T", "--trained-agent", help="Existing model to continue training."
    )
    parser.add_argument("--algo", default="dqn", help="The algorithm to use.")
    parser.add_argument(
        "-n", "--timesteps", type=int, required=True, help="Total number of timesteps."
    )
    parser.add_argument("--it-model", help="Path to the trained model for 'it' agent.")
    args = parser.parse_args()

    log_dir = make_log_dir(args.env, args.log_dir)

    it_model, not_it_model = None, None
    if args.it_model is not None:
        it_model = SAC.load(args.it_model)

    # create environment
    # use VecTransposeImage because SB3 wants channel-first format
    env_kwargs = dict(it_model=it_model, not_it_model=None)
    env = make_vec_env(
        args.env,
        seed=args.seed,
        n_envs=args.n_envs,
        monitor_dir=log_dir,
        env_kwargs=env_kwargs,
    )
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=N_STACK)

    # instantiate the agent
    model = make_model(
        algo_name=args.algo, env=env, seed=args.seed, trained_agent=args.trained_agent
    )

    if EVAL:
        eval_env = make_vec_env(
            args.env, seed=args.seed, n_envs=N_EVAL_ENVS, env_kwargs=env_kwargs
        )
        eval_env = VecTransposeImage(eval_env)
        eval_env = VecFrameStack(eval_env, n_stack=N_STACK)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=max(EVAL_FREQ // args.n_envs, 1),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=False,
            render=False,
        )
    else:
        eval_callback = None

    start = datetime.datetime.now()

    # train the agent
    try:
        model.learn(
            total_timesteps=args.timesteps, progress_bar=True, callback=eval_callback
        )
    except KeyboardInterrupt:
        print("goodbye")

    end = datetime.datetime.now()

    info_path = os.path.join(log_dir, "info.yaml")
    info = {
        "start": start,
        "end": end,
        "env": args.env,
        "algo": args.algo,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "n_stack": N_STACK,
    }
    with open(info_path, "w") as f:
        yaml.dump(info, stream=f)

    # save the agent
    model_path = os.path.join(log_dir, args.env)
    model.save(model_path)
    print(f"Saved logs to {log_dir}")


if __name__ == "__main__":
    main()
