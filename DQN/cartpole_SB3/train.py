import torch
import gymnasium as gym 

#Creating the DQN agent 
from stable_baselines3 import DQN

#Creating the environment 
env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="rgb_array")

#Defining a policy, the activation function and the network layers size
policy_kwargs = dict(activation_fn = torch.nn.ReLU,
                     net_arch = [256,256])

model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

#Training 
model.learn(total_timesteps=1e5, log_interval=500, progress_bar=True)

#Saving the trained model
model.save("dqn_cartpole")
 