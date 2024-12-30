import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py

gym.register_envs(ale_py)

def make_env(env_name, frameskip=5, repeat_action_prob = 0.25,
              render_mode="rgb_array",mode=0,difficulty=0):
    env = gym.make(env_name,
                   render_mode=render_mode,
                   frameskip=frameskip,
                   repeat_action_probability=repeat_action_prob,
                   mode=mode,
                   difficulty=difficulty).unwrapped
    # Removing time limit wrapper from the environment

    return env

if __name__ == "__main__":
    env_name = "ALE/Breakout-v5"
    env = make_env(env_name)
    env.reset(seed=127)
    n_columns = 4
    n_rows = 2
    fig = plt.figure(figsize=(16,9))

    for row in range(n_rows):
        for col in range(n_columns):
            ax = fig.add_subplot(n_rows,n_columns, row*n_columns + col + 1)
            ax.imshow(env.render())
            env.step(env.action_space.sample())
    plt.show()
