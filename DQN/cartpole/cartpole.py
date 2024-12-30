import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

print("First time:",observation,"\n",info)

episode_over = False
for i in range(1000):
    action = env.action_space.sample()  
    
    observation, reward, terminated, truncated, info = env.step(action)
    if i % 100 == 0:
        print(observation,"\n",reward)
    

env.close()