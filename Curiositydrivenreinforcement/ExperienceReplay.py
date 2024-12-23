from random import shuffle
import torch
import torch.nn.functional as F
import numpy as np
from preprocessinginput import number_of_actions

class ExperienceReplay:
    def __init__(self, max_size=500, batch_size=100):
        self.max_size = max_size
        self.batch_size = batch_size 
        self.memory = [] 
        self.counter = 0
        
    def add_memory(self, state1, action, reward, state2):
        self.counter +=1 
        if self.counter % self.max_size == 0: 
            self.shuffle_memory()
            
        if len(self.memory) < self.max_size: 
            self.memory.append( (state1, action, reward, state2) )
        else:
            rand_index = np.random.randint(0,self.max_size-1)
            self.memory[rand_index] = (state1, action, reward, state2)
    
    def shuffle_memory(self): 
        shuffle(self.memory)
        
    def get_batch(self): 
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
        if len(self.memory) < 1:
            print("Error: No data in memory.")
            return None
        
        indices = np.random.choice(np.arange(len(self.memory)),batch_size,replace=False)
        batch = [self.memory[i] for i in indices] #batch is a list of tuples
        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch],dim=0)
        action_batch = torch.Tensor([x[1] for x in batch]).long()
        reward_batch = torch.Tensor([x[2] for x in batch])
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch],dim=0)
        return state1_batch, action_batch, reward_batch, state2_batch