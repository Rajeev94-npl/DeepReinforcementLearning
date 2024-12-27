import torch
from torch import nn 
import gymnasium as gym
import numpy as np
import cv2
from PPO_Agent import ActorNetwork

env = gym.make("CartPole-v1", render_mode="human")


model = ActorNetwork(n_actions=env.action_space.n,
                    input_dims=env.observation_space.shape,
                    alpha=0.0003)

model.load_state_dict(torch.load("temporary\\actor_ppo.pth",weights_only=True))

print("Successfully loaded model!!!!")

observation, info = env.reset()

epsilon = 0.1 
for i in range(1000):
    action_ = model(torch.from_numpy(observation).float()).sample()

    action = torch.squeeze(action_).item()
    
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Time step at termination: {i}")
        break
    print(f"Time step: {i}")
    #env = gym.wrappers.HumanRendering(env)

env.close()