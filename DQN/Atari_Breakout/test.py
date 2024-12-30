import torch
from torch import nn 
import gymnasium as gym
import numpy as np
import cv2
import ale_py
from DQNAgent import DQNAgent
from preprocessing import make_env

gym.register_envs(ale_py)
env_name = "ALE/Breakout-v5"
env = make_env(env_name)
state_dim = env.observation_space.shape
n_actions = env.action_space.n

model = DQNAgent(state_dim, n_actions, epsilon=1)

model.load_state_dict(torch.load("Breakout_DQN1.pth",weights_only=True))

print("Successfully loaded agent model!!!")

model.eval()

observation, info = env.reset()

#print(torch.from_numpy(observation).float().unsqueeze(0).permute(0,-1,1,2).shape)

epsilon = 0.1 
for i in range(10000):
    action_ = model(torch.from_numpy(observation).float().unsqueeze(0)) 
    if np.random.random() < epsilon:
        action = np.random.choice(list(range(n_actions)))
    else:
        action = torch.argmax(action_.long()).item()
    
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Time step at termination: {i}")
        break
    print(f"Time step: {i}")
    cv2.imshow("imshow",cv2.cvtColor(cv2.resize(observation[-1],(400,400)),cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()

env.close()