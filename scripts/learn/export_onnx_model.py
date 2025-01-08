import argparse
import inspect
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import gymnasium as gym
import yaml

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

import shadows

import IPython


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, agent_position, agent_angle, enemy_position, treasure_positions):
        observation = dict(
            agent_position=agent_position,
            agent_angle=agent_angle,
            enemy_position=enemy_position,
            treasure_positions=treasure_positions,
        )
        return self.policy._predict(observation, deterministic=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Path to the logs directory.")
    args = parser.parse_args()

    info_path = os.path.join(args.log_dir, "info.yaml")
    with open(info_path) as f:
        info = yaml.safe_load(f)
    env_name = info["env"]
    algo_name = info["algo"].lower()
    algo = shadows.ALGOS[algo_name]

    env = make_vec_env(env_name)
    env = VecTransposeImage(env)
    obs = env.reset()
    model_path = os.path.join(args.log_dir, "best_model.zip")
    model = algo.load(model_path, env=env, device="cpu")

    onnx_policy = OnnxablePolicy(model.policy)

    # key order needs to match forward
    keys = ["agent_position", "agent_angle", "enemy_position", "treasure_positions"]
    args = tuple(torch.tensor(obs[key]) for key in keys)

    onnx_path = env_name + "_" + algo_name + ".onnx"
    torch.onnx.export(
        onnx_policy,
        args=args,
        f=onnx_path,
        opset_version=17,
        input_names=keys,
        external_data=False,
        dynamo=True,
    )
    print(f"exported model to {onnx_path}")

    # check the model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # perform inference
    ort_sess = ort.InferenceSession(onnx_path)
    action = ort_sess.run(None, obs)

    # IPython.embed()


main()
