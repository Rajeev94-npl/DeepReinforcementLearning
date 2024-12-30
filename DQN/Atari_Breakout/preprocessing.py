import gymnasium as gym 
import matplotlib.pyplot as plt
import numpy as np 
import ale_py

from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import TransformReward

gym.register_envs(ale_py)

def make_env(env_name,
             clip_rewards=True):
    env = gym.make(env_name,
                   render_mode='rgb_array',
                   frameskip=1
              )
    env = AtariPreprocessing(env, screen_size=84, scale_obs=True)
    env = FrameStackObservation(env, stack_size=4)
    if clip_rewards:
        env = TransformReward(env, lambda r: np.sign(r))
    return env

if __name__ == "__main__":
    env_name = "ALE/Breakout-v5"
    env = make_env(env_name)
    obs, _ = env.reset()
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    print("Observation Shape:", state_shape) #prints 4x84x84
    obs = obs[:] #unpack lazyframe
    obs = np.transpose(obs,[1,0,2]) #move axes
    obs = obs.reshape((obs.shape[0], -1))
    plt.figure(figsize=[15,15])
    plt.title("Agent observation (4 frames left to right)")
    plt.imshow(obs, cmap='gray')
    plt.show()
    plt.close()