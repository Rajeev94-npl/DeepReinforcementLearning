import numpy as np
import random 

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 

from DQNAgent import DQNAgent
from preprocessing import make_env
from utilities import ReplayBuffer,compute_loss,play_and_record, evaluate

from scipy.signal import convolve
from scipy.signal.windows import gaussian

from IPython.display import clear_output
from tqdm import trange

# Setting a seed
seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "ALE/Breakout-v5"

#Setting up env , agent and target networks
env = make_env(env_name)
state_dim = env.observation_space.shape
n_actions = env.action_space.n
state, _ = env.reset(seed=seed)

agent = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
target_network = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
target_network.load_state_dict(agent.state_dict())


#Filling experience replay with some samples using full random policy

exp_replay = ReplayBuffer(10**4)

for i in range(100):
    play_and_record(state, agent, env, exp_replay, n_steps=10**2)
    if len(exp_replay) == 10**4:
        break
print(len(exp_replay))

#Setting up some parameters for training
timesteps_per_epoch = 1
batch_size = 32
total_steps = 100000 #Make this bigger for more training.

#Initializing Optimizer
opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

#Setting up exploration epsilon
start_epsilon = 1
end_epsilon = 0.05
eps_decay_final_step = 1 * 10**6

# Setting up some frequency for logging and updating target network
loss_freq = 20
refresh_target_network_freq = 100
eval_freq = 10000

# To clip the gradients
max_grad_norm = 5000

mean_rw_history = []
td_loss_history = []

def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step

def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')

state, _ = env.reset(seed=seed)

for step in trange(total_steps + 1):

    # Reducing exploration as we progress
    agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)

    # Taking timesteps_per_epoch and updating experience replay buffer
    _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

    # Training by sampling batch_size of data from experience replay
    states, actions, rewards, next_states, done_flags = exp_replay.sample(batch_size)


    # loss = computing td loss
    loss = compute_loss(agent, target_network,
                           states, actions, rewards, next_states, done_flags,
                           gamma=0.99,
                           device=device)

    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()

    if step % loss_freq == 0:
        td_loss_history.append(loss.data.cpu().item())

    if step % refresh_target_network_freq == 0:
        # Loading agent weights into target_network
        target_network.load_state_dict(agent.state_dict())

    if step % eval_freq == 0 and step != 0:
        # eval the agent
        mean_rw_history.append(evaluate(
            make_env(env_name), agent, n_games=3, greedy=True, t_max=1000)
        )

        #clear_output(False)
        print("buffer size = %i, epsilon = %.5f" %
              (len(exp_replay), agent.epsilon))


plt.figure(figsize=[16, 5])
plt.subplot(1, 2, 1)
plt.title("Mean return per episode")
plt.plot(mean_rw_history)
plt.grid()

assert not np.isnan(td_loss_history[-1])
plt.subplot(1, 2, 2)
plt.title("TD loss history (smoothened)")
plt.plot(smoothen(td_loss_history))
plt.grid()

plt.show()

plt.close()

torch.save(agent.state_dict(),"Breakout_DQN1.pth")

