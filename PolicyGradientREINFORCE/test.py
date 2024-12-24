import torch
from torch import nn 
import gymnasium as gym
import numpy as np
import cv2

env = gym.make("CartPole-v1", render_mode="human")

#Parameters
input_layer = 4 
output_layer = 2 
hidden_layer = 150

model = nn.Sequential(
     nn.Linear(input_layer, hidden_layer),
     nn.LeakyReLU(),
     nn.Linear(hidden_layer, output_layer),
     nn.Softmax(dim=0) 
)

model.load_state_dict(torch.load("PGmodel1.pth",weights_only=True))

observation, info = env.reset()

for i in range(1000):
    action_prob = model(torch.from_numpy(observation).float()) 
    action = np.random.choice(np.array([0,1]), p=action_prob.data.numpy())
    
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Time step at termination: {i}")
        break
    print(f"Time step: {i}")

env.close()