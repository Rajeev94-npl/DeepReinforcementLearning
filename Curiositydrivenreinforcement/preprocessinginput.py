import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque 

number_of_actions = 6

def downscale_observations(obvs, new_size=(42,42), to_grayscale=True):
    if to_grayscale:
        return resize(obvs,new_size,anti_aliasing=True).max(axis=2)
    else:
        return resize(obvs,new_size,anti_aliasing=True)

def prepare_state(state):
    return torch.from_numpy(downscale_observations(state,to_grayscale=True)).float().unsqueeze(dim=0)

def prepare_multi_state(state1, state2):
    state1 = state1.clone()
    tmp = torch.from_numpy(downscale_observations(state2, to_grayscale=True)).float()
    state1[0][0] = state1[0][1]
    state1[0][1] = state1[0][2]
    state1[0][2] = tmp 
    return state1

def prepare_initial_state(state, N=3):
    state_ = torch.from_numpy(downscale_observations(state, to_grayscale=True)).float()
    tmp = state_.repeat((N,1,1))
    return tmp.unsqueeze(dim=0)