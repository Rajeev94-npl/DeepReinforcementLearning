
import torch
from torch import nn 
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import cv2

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

class ActorCritic(nn.Module): 
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(4,25)
        self.linear2 = nn.Linear(25,50)
        self.actor_linear1 = nn.Linear(50,2)
        self.linear3 = nn.Linear(50,25)
        self.critic_linear1 = nn.Linear(25,1)

    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(y))
        actor = F.log_softmax(self.actor_linear1(y),dim=0) 
        c = F.relu(self.linear3(y.detach()))
        critic = torch.tanh(self.critic_linear1(c)) 
        return actor, critic 

MasterNode = ActorCritic()
MasterNode.load_state_dict(torch.load("DA2CNStep1.pth", weights_only=True))
print("Successful in loading the model.")

for i in range(1000):
    state_ = np.array(env.reset()[0])
    state = torch.from_numpy(state_).float()
    logits,value = MasterNode(state)
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    state2, reward, done,_, info = env.step(action.detach().numpy())
    if done:
        print("Lost")
        env.reset()
    state_ = np.array(env.reset()[0])
    state = torch.from_numpy(state_).float()
    print(f"Time step {i}")

env.close()