"""Plot reward curves from training."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import window_func

import IPython

EPISODES_WINDOW = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Path to the logs directory.")
    args = parser.parse_args()

    data_frame = load_results(args.log_dir)

    timesteps = np.cumsum(data_frame.l.values)
    rewards = data_frame.r.values

    # smoothed reward curve
    t_smooth, r_smooth = window_func(timesteps, rewards, EPISODES_WINDOW, np.mean)

    plt.scatter(timesteps, rewards, s=1, color="b")
    plt.plot(t_smooth, r_smooth, color="r", label="Train")

    # also plot evaluations, if available
    eval_path = Path(args.log_dir) / "evaluations.npz"
    if eval_path.exists():
        eval_data = np.load(eval_path)
        eval_timesteps = eval_data["timesteps"]
        eval_rewards = np.mean(eval_data["results"], axis=-1)
        plt.plot(eval_timesteps, eval_rewards, color="g", label="Eval")

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
