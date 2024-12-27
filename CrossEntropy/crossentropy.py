import numpy as np
import gymnasium as gym 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from dataclasses import dataclass 
import typing as tt
from torch.utils.tensorboard.writer import SummaryWriter


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70 

class CrossEntropyNetwork(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(CrossEntropyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]

def iterate_batches(env: gym.Env, net: CrossEntropyNetwork, batch_size: int) -> tt.Generator[tt.List[Episode], None, None]:
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    soft_max = nn.Softmax(dim=1)
    while True:
        observ = torch.tensor(obs, dtype=torch.float32)
        action_probs_v = soft_max(net(observ.unsqueeze(0)))
        action_probs = action_probs_v.data.numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch: tt.List[Episode], percentile: float) -> \
        tt.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        train_act.extend(map(lambda step: step.action, episode.steps))

    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    assert env.observation_space.shape is not None
    observ_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    no_of_actions = int(env.action_space.n)

    net = CrossEntropyNetwork(observ_size, HIDDEN_SIZE, no_of_actions)
    print(net)
    objective_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective_func(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_mean, reward_bound))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        if reward_mean > 475:
            print("Solved!")
            break
    writer.close()
    #tensorboard --logdir=runs
    torch.save(net.state_dict(),"crossent1.pth")

