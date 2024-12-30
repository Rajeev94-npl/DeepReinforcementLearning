import torch 
import torch.nn as nn
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        state_dim = state_shape[0]
       
        self.net = nn.Sequential()
        self.net.add_module('conv1', nn.Conv2d(4,16,kernel_size=8, stride=4))
        self.net.add_module('relu1', nn.ReLU())
        self.net.add_module('conv2', nn.Conv2d(16,32,kernel_size=4, stride=2))
        self.net.add_module('relu2', nn.ReLU())
        self.net.add_module('flatten', nn.Flatten())
        self.net.add_module('linear3', nn.Linear(2592, 256)) #2592 calculated as 32*9*9 of final conv layer
        self.net.add_module('relu3', nn.ReLU())
        self.net.add_module('linear4', nn.Linear(256, n_actions))
        #self.parameters = self.net.parameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, state_t):
        #Q(s,a)
        qvalues = self.net(state_t)
        return qvalues
    
    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = torch.tensor(np.asarray(states), device=self.device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()
    
    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)