import torch
from torch import nn,optim
import gymnasium as gym
from collections import deque
import numpy as np
import random
import copy

#Creating the environment 
env = gym.make("CartPole-v1", render_mode="rgb_array")

layer1_size = 4
layer2_size = 150
layer3_size = 100
layer4_size = 2


model = torch.nn.Sequential(
    torch.nn.Linear(layer1_size, layer2_size),
    torch.nn.ReLU(),
    torch.nn.Linear(layer2_size, layer3_size),
    torch.nn.ReLU(),
    torch.nn.Linear(layer3_size,layer4_size)
)

model2 = copy.deepcopy(model) 
model2.load_state_dict(model.state_dict()) 

loss_fn = nn.MSELoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9 #Discount Factor
epsilon = 0.3

epochs = 1000
losses = []
mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size)
MAX_MOVES = 400
sync_freq = 500 
status = True
j = 0

for i in range(epochs):
    state1_ = env.reset()[0]
    state1 = torch.from_numpy(state1_).float().unsqueeze(0)
    #print("state1:",state1.shape)
    moves = 0
    status = True
    while status: 
        j += 1
        moves += 1
        qval = model(state1)
        #print("qval:",qval.shape)
        qval_ = qval.data.numpy()
        #print("qval:",qval_)
        if (random.random() < epsilon):
            action_ = np.random.randint(0,2)
        else:
            action_ = np.argmax(qval_)
        #print("action:",action_)
        action = [0,1][action_]
        state2_, _, done, _, info = env.step(action)
        done1or0 = 1 if done == True else 0
        state2 = torch.from_numpy(state2_).float()
        state2 = state2.unsqueeze(0)
        #print(state2.shape)
        #print("First state2:",state2)
        reward = moves
        exp =  (state1, action_, reward, state2, done1or0)
        replay.append(exp) 
        state1 = state2
        
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            #print("Minibatch shape:",minibatch)
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
            #print(state1_batch.size())
            #print(state1_batch)
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
            
            Q1 = model(state1_batch) 
            with torch.no_grad():
                Q2 = model2(state2_batch) 
            #print(Q2.shape)
            #print((torch.max(Q2,dim=1)[0]))
            Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i,loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            if j % sync_freq == 0: 
                model2.load_state_dict(model.state_dict())
        
        if moves > MAX_MOVES or done==True:
            status = False

max_episode = max([r for (s1,a,r,s2,d) in replay])

print("Highest episode length during training:",max_episode)

torch.save(model.state_dict(),"DQN1.pth")