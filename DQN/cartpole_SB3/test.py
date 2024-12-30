import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = DQN.load("models", env=env)

print("Successfully loaded the trained DQN model")

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs,rewards,dones,info = vec_env.step(action)
    vec_env.render("human")

vec_env.close()