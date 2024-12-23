import torch
import torch.nn.functional as F
from dqn import DeepQNetwork
from preprocessinginput import prepare_initial_state, prepare_multi_state
from collections import deque
import gymnasium as gym 
import ale_py
import numpy as np
import cv2
from preprocessinginput import number_of_actions

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")


def policy(qvalues, eps=None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0,high=number_of_actions-1,size=(1,))
        else:
            return torch.argmax(qvalues)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvalues),dim=-1), num_samples=1) 

DeepQModel = DeepQNetwork()

DeepQModel.load_state_dict(torch.load("DQNPongmodel2.pth",weights_only=True))
print("SUccessfully loaded model!!!")

done = True
eps=0.15
state_deque = deque(maxlen=3)
for step in range(5000):
    if done:
        env.reset()
        state1 = prepare_initial_state(env.render())
    q_val_pred = DeepQModel(state1)
    action = int(policy(q_val_pred,eps))
    state2, reward, done,_, info = env.step(action)
    print(reward)
    state2 = prepare_multi_state(state1,state2)
    state1=state2
    #env.render()
    cv2.imshow("imshow",cv2.cvtColor(env.render(),cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
