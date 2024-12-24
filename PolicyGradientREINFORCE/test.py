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

# model = nn.Sequential(
#      nn.Linear(input_layer, hidden_layer),
#      nn.LeakyReLU(),
#      nn.Linear(hidden_layer, output_layer),
#      nn.Softmax(dim=0) 
# )

class PolicyGradientModel(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(PolicyGradientModel,self).__init__()
        self.fc1 = nn.Linear(input_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, output_layer)
        self.act1 = nn.LeakyReLU()
        self.output = nn.Softmax(dim=0)
    
    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.output(self.fc2(x))
        return x

model = PolicyGradientModel(input_layer,output_layer,hidden_layer)    

model.load_state_dict(torch.load("PGmodel2.pth",weights_only=True))

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