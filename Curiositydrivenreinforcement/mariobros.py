import gymnasium as gym 
import ale_py
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(1000):
    #action = env.action_space.sample()
    action = 0
    observation,reward,terminated,truncated,info = env.step(action)
    if terminated or truncated:
        observation,info = env.reset()
    
    print(info)
    
env.close()
