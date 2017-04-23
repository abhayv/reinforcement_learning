"""Implements Q-learning with a classic table based approach.

author: abhay.vardhan@gmail.com
"""
import gym
import numpy as np

env = gym.make('FrozenLake-v0')

learning_rate = 0.85
gamma = 0.99
epsilon = 0.01
Q = np.zeros([env.observation_space.n,env.action_space.n])
rewards_by_episode = []
num_episodes = 2000
for episode_num in range(num_episodes):
    reward_for_episode = 0
    observation = env.reset()
    for t in range(100):
        randm = np.random.randn(1,env.action_space.n)
        randm *= (1. / (episode_num + 1))
        action = np.argmax(Q[observation, :] + randm)
        next_state, reward, done, info = env.step(action)
        if reward != 0:
            print(reward)
        Q[observation, action] += learning_rate * (reward + gamma * np.max(Q[next_state, :]) - Q[observation, action])
        observation = next_state
        reward_for_episode += reward
        if done:
            break
    rewards_by_episode.append(reward_for_episode)
print("Score over time: " + str(sum(rewards_by_episode) / num_episodes))
print(Q)
