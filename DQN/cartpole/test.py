import torch
from torch import nn 
import gymnasium as gym
import numpy as np
import cv2

env = gym.make("CartPole-v1", render_mode="human")

layer1_size = 4
layer2_size = 150
layer3_size = 100
layer4_size = 2


model = torch.nn.Sequential(
    torch.nn.Linear(layer1_size, layer2_size),
    torch.nn.ReLU(),
    torch.nn.Linear(layer2_size, layer3_size),
    torch.nn.ReLU(),
    torch.nn.Linear(layer3_size,layer4_size)
)

model.load_state_dict(torch.load("DQN1.pth",weights_only=True))

observation, info = env.reset()

epsilon = 0.1 
for i in range(1000):
    action_ = model(torch.from_numpy(observation).float()) 
    if np.random.random() < epsilon:
        action = np.random.choice([0,1])
    else:
        action = torch.argmax(action_.long()).item()
    
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Time step at termination: {i}")
        break
    print(f"Time step: {i}")

env.close()