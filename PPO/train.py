import gymnasium as gym
import numpy as np
from PPO_Agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode = "rgb_array")
    learn_freq = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 500 #Episodes

    figure_file = 'temporary\\cartpole.png'

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        steps = 0
        while not done:
            steps += 1
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done,_, info = env.step(action)
            
            n_steps += 1
            score += reward
            reward = steps
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % learn_freq == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            #agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    agent.save_models()