import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

def init_model(learning_rate):
    activation = tf.keras.layers.LeakyReLU()
    layers = []
    for i in range(5):
        layers.append(tf.keras.layers.Dense(4 << i, activation=activation))
    layers.append(tf.keras.layers.Dense(2, activation=activation))

    model = tf.keras.Sequential(layers)
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

def get_max(arr):
    max_i = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[max_i]:
            max_i = i
    return max_i

class GymModel:
    def __init__(self, train_id, max_episodes):
        self.env = gym.make('CartPole-v0')
        self.train_id = train_id
        self.max_episodes = max_episodes

    def __fini__(self):
        self.env.close()

    def get_model_name(self, generation_id):
        return 'model' + str(generation_id) + '-' + str(self.train_id)

    def get_plot_name(self, generation_id):
        return 'plot' + str(generation_id) + '-' + str(self.train_id) + '.png'

    def play(self, generation):
        model = tf.keras.models.load_model(self.get_model_name(generation))

        while True:
            t = 0
            self.env.reset()
            observation, _, _, _ = self.env.step(self.env.action_space.sample())
            while True:
                q_values = model.predict(np.array([observation]))[0]
                action_id = get_max(q_values)

                observation, _, done, _ = self.env.step(action_id)
                self.env.render()
                if done:
                    break
                t += 1

            print('[' + str(self.train_id) + '] Simulation ended after ' + str(t) + ' timesteps')

    def train(self, generation, hyperparameters, rendering=False):
        if os.path.exists(self.get_model_name(generation)):
            model = tf.keras.models.load_model(self.get_model_name(generation))
        else:
            model = init_model(hyperparameters[0])

        total_rewards = []
        timesteps_count = []
        data = []

        for e in range(self.max_episodes):
            self.env.reset()
            if rendering:
                self.env.render()
            observation, _, _, _ = self.env.step(self.env.action_space.sample())

            total_reward = 0.

            t = 0
            while True:
                #print('State: ' + str(np.array([observation])))
                q_values = model.predict(np.array([observation]))[0]
                #print('Q-Values: ' + str(q_values))

                random_ratio = 0.1 # Tune
                if random.uniform(0., 1.) < random_ratio:
                    action_id = self.env.action_space.sample()
                else:
                    action_id = get_max(q_values)
                action_q_value = q_values[action_id]

                next_observation, reward, done, _ = self.env.step(action_id)
                if done:
                    reward = -100.
                else:
                    reward = 100.
                    reward -= (abs(observation[0]) / 0.24) * 30.
                    reward -= (abs(observation[2]) / 0.20) * 70.
                #print('reward: ' + str(reward))
                total_reward += reward
                data.append((observation, reward, q_values, action_id, action_q_value))

                if rendering:
                    self.env.render()

                if done:
                    print('[' + str(self.train_id) + '] Episode ' + str(e) + ' fails after ' + str(t) + ' timesteps')
                    break;

                observation = next_observation
                t += 1;

            total_rewards.append(total_reward)
            timesteps_count.append(t)

            if e % 10 == 0:
                self.replay(model, data, hyperparameters)
                data.clear()
                model.save(self.get_model_name(generation))

        self.env.close()

        plt.xlabel('Episode')
        plt.ylabel('Total reward/Timesteps count')
        plt.plot(total_rewards)
        plt.plot(timesteps_count)
        plt.savefig(self.get_plot_name(generation))
        plt.clf()

        return sum(timesteps_count) # TODO Use linear regression?

    def replay(self, model, data, hyperparameters):
        print('[' + str(self.train_id) + '] Training...')

        for d in data:
            observation = d[0]
            reward = d[1]
            q_values = d[2]
            action_id = d[3]
            action_q_value = d[4]

            #print('Next state: ' + str(np.array([observation])))
            next_q_values = model.predict(np.array([observation]))[0]
            #print('Next Q-Values: ' + str(q_values))
            next_action_id = get_max(next_q_values)
            next_action_q_value = next_q_values[next_action_id]

            new_q_value = (1. - hyperparameters[1]) * action_q_value + hyperparameters[1] * (reward + hyperparameters[2] * next_action_q_value)
            new_q_values = q_values
            new_q_values[action_id] = new_q_value
            #print('New Q-Values: ' + str(new_q_values))
            model.fit(np.array([observation]),
                    np.array([new_q_values]),
                    epochs=10,
                    verbose=0)

        print('[' + str(self.train_id) + '] Training ended')
