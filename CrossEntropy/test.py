import torch
from torch import nn 
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from crossentropy import CrossEntropyNetwork

env = gym.make("CartPole-v1", render_mode="human")

#Parameters
input_layer = 4 
output_layer = 2 
hidden_layer = 128

model = CrossEntropyNetwork(input_layer,hidden_layer,output_layer) 
soft_max = nn.Softmax(dim = 1)   

model.load_state_dict(torch.load("crossent1.pth",weights_only=True))

observation, info = env.reset()

for i in range(1000):
    action_ = model(torch.from_numpy(observation).float().unsqueeze(0)) 
    action_probs = soft_max(action_)
    #print("action probas:",action_probs.data)
    action = np.random.choice(np.array([0,1]), p=action_probs.data.numpy()[0])
    
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Time step at termination: {i}")
        break
    print(f"Time step: {i}")

env.close()