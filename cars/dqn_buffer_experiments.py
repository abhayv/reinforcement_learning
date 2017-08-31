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

buffer_size = 1
batch_size = None
sample_size = 1
num_episodes = 2000
experience_buffer_overwrite_pos = 0


class StateProcessor(object):
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.env_state = tf.placeholder(tf.int32, [batch_size])
            self.processed_state = tf.identity(self.env_state)

    def process(self, session, state):
        return session.run(self.processed_state, feed_dict={self.env_state: state})


class Model(object):
    def __init__(self, state):
        with tf.variable_scope("model"):
            self.state = state
            self.W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
            self.Q = tf.matmul(tf.one_hot(state, 16), self.W)
            self.predicted_action = tf.argmax(self.Q, 1)[0]
            self.Q_target = tf.placeholder(tf.float32, [batch_size, 4])
            self.loss = tf.reduce_sum(tf.square(self.Q_target - self.Q))
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                self.loss)

    def train(self, sess, sample_obs, sample_Qs):
        return sess.run([self.train_step, self.W],
                 feed_dict={self.state: sample_obs, self.Q_target: sample_Qs})

    def predict(self, sess, observation):
        return sess.run([self.predicted_action, self.Q],
                 feed_dict={self.state: observation})


def run_session(show_plot=False):
    experience_buffer = list()

    rewards_by_episode = []
    steps_for_episode = []

    state_processor = StateProcessor()
    model = Model(state_processor.processed_state)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    def batch_actions_and_train(observation, Q_this_step):
        global experience_buffer_overwrite_pos
        if len(experience_buffer) < buffer_size:
            experience_buffer.append({'obs': observation, 'Q': Q_this_step[0]})
        else:
            samples = np.random.choice(experience_buffer, sample_size)
            sample_obs = [x['obs'] for x in samples]
            sample_Qs = [x['Q'] for x in samples]
            model.train(sess, sample_obs, sample_Qs)
            experience_buffer[experience_buffer_overwrite_pos] = {'obs': observation,
                                                                  'Q': Q_this_step[0]}
            experience_buffer_overwrite_pos = (experience_buffer_overwrite_pos + 1) % len(
                experience_buffer)

    def run():
        global epsilon
        for episode_num in range(num_episodes):
            reward_for_episode = 0
            observation = env.reset()
            step = 0
            while step < 100:
                observation = state_processor.process(sess, [observation])
                action, Q_this_step = model.predict(sess, observation)
                if np.random.rand(1) < epsilon:
                    action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                observation = next_state
                next_state = state_processor.process(sess, [next_state])
                next_Q = sess.run(model.Q, feed_dict={state_processor.env_state: next_state})
                Q_this_step[0][action] = reward + gamma * np.max(next_Q[0])
                batch_actions_and_train(observation, Q_this_step)

                reward_for_episode += reward
                if done:
                    epsilon = 1. / ((episode_num / 50) + 10)
                    break
                step += 1
            steps_for_episode.append(step)
            rewards_by_episode.append(reward_for_episode)

    run()
    print("Score over time: " + str(sum(rewards_by_episode) / num_episodes))
    if show_plot:
        plt.figure(1)
        plt.subplot(211)
        plt.plot(rewards_by_episode)
        plt.subplot(212)
        plt.plot(steps_for_episode)
        plt.show()


if __name__ == '__main__':
    run_session()
    run_session()
    run_session()
