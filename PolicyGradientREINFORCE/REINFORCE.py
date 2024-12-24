import numpy as np
import torch
from torch import nn,optim
import gymnasium as gym 

#Creating the environment 
env = gym.make("CartPole-v1", render_mode="rgb_array")

#Parameters
input_layer = 4 
output_layer = 2 

#Hyperparameters
learning_rate = 0.009
hidden_layer = 150

class PolicyGradientModel(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(PolicyGradientModel,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act1 = nn.LeakyReLU()
        self.output = nn.Softmax(dim=0)
    
    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.output(self.fc2(x))
        return x

model = PolicyGradientModel(input_layer,hidden_layer,output_layer)
#Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Initial testing of the architecture
# state1,_ = env.reset()
# pred = model(torch.from_numpy(state1)) 
# action = np.random.choice(np.array([0,1]), p=pred.data.numpy()) 
# state2, reward, done,_,info = env.step(action) 
# print(reward)

#Discounting older rewards in policy gradient
def discount_rewards(rewards, gamma=0.99):
    length_of_rewards = len(rewards)
    discounted_return = torch.pow(gamma,torch.arange(length_of_rewards).float()) * rewards 
    discounted_return /= discounted_return.max() 
    return discounted_return

#Defining loss funtion 
def loss_func(predictions, rewards): 
    return -1 * torch.sum(rewards * torch.log(predictions)) 

#Training 

#Hyperparameters
MAX_DURATION = 500  #of an episode
MAX_EPISODES = 1000
gamma = 0.99
score = [] 
expectation = 0.0

for episode in range(MAX_EPISODES):
    current_state = env.reset()[0]
    done = False
    transitions = [] 
    
    for t in range(MAX_DURATION): 
        action_prob = model(torch.from_numpy(current_state).float()) 
        action = np.random.choice(np.array([0,1]), p=action_prob.data.numpy()) 
        prev_state = current_state
        current_state, _, done, _, info = env.step(action) 
        transitions.append((prev_state, action, t+1))
        if done: 
            break

    episode_len = len(transitions) 
    score.append(episode_len)
    reward_batch = torch.Tensor(np.array([r for (s,a,r) in transitions])).flip(dims=(0,)) 
    discounted_returns = discount_rewards(reward_batch) 
    state_batch = torch.Tensor(np.array([s for (s,a,r) in transitions])) 
    action_batch = torch.Tensor(np.array([a for (s,a,r) in transitions])) 
    pred_batch = model(state_batch) 
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() 
    loss = loss_func(prob_batch, discounted_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Training episode: {episode}")

torch.save(model.state_dict(),"PGmodel1.pth")