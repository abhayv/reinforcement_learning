"""Implements a simple Q Network without experience replay.

author: abhay.vardhan@gmail.com
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')

# Reward discount factor
gamma = 0.99

# e-greedy factor
epsilon = 0.1

env_state = tf.placeholder(tf.float32, [1, 16])
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Q = tf.matmul(env_state, W)
predicted_action = tf.argmax(Q, 1)[0]

Q_target = tf.placeholder(tf.float32, [1, 4])
loss = tf.reduce_sum(tf.square(Q_target - Q))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

rewards_by_episode = []
steps_for_episode = []
num_episodes = 2000
for episode_num in range(num_episodes):
    reward_for_episode = 0
    observation = env.reset()
    step = 0
    while step < 100:
        action, Q_this_step = sess.run([predicted_action, Q],
                                       feed_dict={env_state: np.identity(16)[observation:observation + 1]})
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        next_Q = sess.run(Q, feed_dict={env_state: np.identity(16)[next_state:next_state + 1]})
        Q_this_step[0][action] = reward + gamma * np.max(next_Q[0])
        sess.run([train_step, W],
                 feed_dict={env_state: np.identity(16)[observation:observation + 1], Q_target: Q_this_step})

        observation = next_state
        reward_for_episode += reward
        if done:
            epsilon = 1. / ((episode_num / 50) + 10)
            break
        step += 1
    steps_for_episode.append(step)
    rewards_by_episode.append(reward_for_episode)
print("Score over time: " + str(sum(rewards_by_episode) / num_episodes))
plt.figure(1)
plt.subplot(211)
plt.plot(rewards_by_episode)
plt.subplot(212)
plt.plot(steps_for_episode)
plt.show()
