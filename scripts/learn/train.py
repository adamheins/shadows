import os
import datetime
import gymnasium as gym
import shoot

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage


ENV_ID = "Simple-v0"
ENV_KWARGS = {"grayscale": True}
LOG_DIR = "logs"

SEED = 0
TOTAL_TIMESTEPS = 2_000_000
EVAL = False


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(LOG_DIR, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # create environment
    # use VecTransposeImage because SB3 wants channel-first format
    env = make_vec_env(ENV_ID, env_kwargs=ENV_KWARGS, n_envs=1, monitor_dir=log_dir)
    env = VecTransposeImage(env)

    # Instantiate the agent
    # TODO this does not work as well as rl_zoo - why?
    model = DQN(
        policy="CnnPolicy",
        env=env,
        seed=SEED,
        buffer_size=100000,
        learning_rate=1e-4,
        batch_size=32,
        learning_starts=100000,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        # exploration_fraction=0.1,
        exploration_fraction=0.5,
        exploration_final_eps=0.01,
        verbose=1,
    )

    if EVAL:
        eval_env = make_vec_env(ENV_ID, n_envs=1, env_kwargs=ENV_KWARGS)
        eval_env = VecTransposeImage(eval_env)
        eval_callback = EvalCallback(
            eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=25000, deterministic=True, render=False
        )
    else:
        eval_callback = None

    # train the agent
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=eval_callback)

    # save the agent
    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)
    # print(f"saved model to {model_path}")


if __name__ == "__main__":
    main()
