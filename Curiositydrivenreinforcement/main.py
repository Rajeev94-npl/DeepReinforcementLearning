import torch
import torch.nn.functional as F
from torch import nn, optim
from ExperienceReplay import ExperienceReplay
from intrinsiccuriositymodule import encoder_phi, inverse_model, forward_model
from dqn import DeepQNetwork
from preprocessinginput import prepare_initial_state, prepare_multi_state,prepare_state,downscale_observations
from collections import deque
import gymnasium as gym 
import ale_py
from preprocessinginput import number_of_actions

gym.register_envs(ale_py)
#env = gym.make("ALE/MarioBros-v5", render_mode="rgb_array")
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")


params = {
    'batch_size':150,
    'beta':0.2,
    'lambda':0.1,
    'eta': 1.0,
    'gamma':0.2,
    'max_episode_len':100,
    'min_progress':15,
    'action_repeats':6,
    'frames_per_state':3
}

replay = ExperienceReplay(max_size=1000, batch_size=params['batch_size'])
DeepQmodel = DeepQNetwork()
encoder = encoder_phi()
forward_model = forward_model()
inverse_model = inverse_model()
forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')
qloss = nn.MSELoss()
all_model_params = list(DeepQmodel.parameters()) + list(encoder.parameters()) 
all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
opt = optim.Adam(lr=0.001, params=all_model_params)

def loss_fn(q_loss, inverse_loss, forward_loss):
    loss_ = (1 - params['beta']) * inverse_loss
    loss_ += params['beta'] * forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + params['lambda'] * q_loss
    return loss

def reset_env():
    env.reset()
    state1 = prepare_initial_state(env.render())
    return state1

def ICM(state1, action, state2, forward_scale=1., inverse_scale=1e-4):
    state1_hat = encoder(state1) 
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(), action.detach()) 
    forward_pred_error = forward_scale * forward_loss(state2_hat_pred, \
                        state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat, state2_hat) 
    inverse_pred_error = inverse_scale * inverse_loss(pred_action, \
                                        action.detach().flatten()).unsqueeze(dim=1)
    return forward_pred_error, inverse_pred_error

def minibatch_train(use_extrinsic=True):
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
    action_batch = action_batch.view(action_batch.shape[0],1) 
    reward_batch = reward_batch.view(reward_batch.shape[0],1)
    
    forward_pred_error, inverse_pred_error = ICM(state1_batch, action_batch, state2_batch) 
    i_reward = (1. / params['eta']) * forward_pred_error 
    reward = i_reward.detach() 
    if use_extrinsic: 
        reward += reward_batch 
    qvals = DeepQmodel(state2_batch) 
    reward += params['gamma'] * torch.max(qvals)
    reward_pred = DeepQmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack( (torch.arange(action_batch.shape[0]), \
    action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), F.normalize(reward_target.detach()))
    return forward_pred_error, inverse_pred_error, q_loss

def policy(qvalues, eps=None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0,high=number_of_actions - 1,size=(1,))
        else:
            return torch.argmax(qvalues)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvalues),dim=-1), num_samples=1) 

# Training Loop

epochs = 500
env.reset()
state1 = prepare_initial_state(env.render())
eps=0.15
losses = []
episode_length = 0
switch_to_eps_greedy = 1000
state_deque = deque(maxlen=params['frames_per_state'])
e_reward = 0.

ep_lengths = []
use_explicit = False
for i in range(epochs):
    opt.zero_grad()
    episode_length += 1
    q_val_pred = DeepQmodel(state1) 
    if i > switch_to_eps_greedy: 
        action = int(policy(q_val_pred,eps))
    else:
        action = int(policy(q_val_pred))
    for j in range(params['action_repeats']): 
        state2, e_reward_, done,_, info = env.step(action)
       
        if done:
            state1 = reset_env()
            break
        e_reward += e_reward_
        state_deque.append(prepare_state(state2))
    state2 = torch.stack(list(state_deque),dim=1) 
    replay.add_memory(state1, action, e_reward, state2) 
    e_reward = 0
    # if episode_length > params['max_episode_len']: 
    #     if (info['x_pos'] - last_x_pos) < params['min_progress']:
    #         done = True
    #     else:
    #         last_x_pos = info['x_pos']
    # if done:
    #     ep_lengths.append(info['x_pos'])
    #     state1 = reset_env()
    #     last_x_pos = env.env.env._x_position
    #     episode_length = 0
    # else:
    #     state1 = state2
    state1 = state2

    if len(replay.memory) < params['batch_size']:
        continue
    forward_pred_error, inverse_pred_error, q_loss = minibatch_train(use_extrinsic=True) 
    loss = loss_fn(q_loss, forward_pred_error, inverse_pred_error) 
    loss_list = (q_loss.mean(), forward_pred_error.flatten().mean(),\
    inverse_pred_error.flatten().mean())
    losses.append(loss_list)
    loss.backward()
    opt.step()
    print(f"Training epoch:{i}, average_loss:{loss}")


torch.save(DeepQmodel.state_dict(),"DQNPongmodel2.pth")    


