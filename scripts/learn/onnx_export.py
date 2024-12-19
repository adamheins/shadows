import numpy as np
import onnx
import onnxruntime as ort
import torch
import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

import shadows

import IPython


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, position, angle, image):
        # construct the Dict input
        observation = dict(position=position, angle=angle, image=image)
        return self.policy._predict(observation, deterministic=False)


env = make_vec_env("Simple-v0")
env = VecTransposeImage(env)
obs = env.reset()
model = DQN.load("logs/Simple-v0_47/Simple-v0.zip", env=env, device="cpu")

onnx_policy = OnnxablePolicy(model.policy)

# key order needs to match forward
keys = ["position", "angle", "image"]
args = tuple(torch.tensor(obs[key]) for key in keys)

onnx_path = "dqn.onnx"
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

IPython.embed()
