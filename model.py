import gym
import math
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

def write_csv(path, data):
    with open(path, 'a+') as f:
        for i in range(len(data)):
            for j in range(len(data[i])):
                f.write(str(data[i][j]))
                if j < len(data[i]) - 1:
                    f.write(',')
            f.write('\n')

def get_score(data):
    count = len(data)
    avg_x = math.floor(count * (count + 1) / 2) / count

    avg_y = 0.
    for d in data:
        avg_y += d
    avg_y /= count

    i = 0.
    j = 0.
    for x in range(count):
        i += (x - avg_x) * (data[x] - avg_y)
        j += (x - avg_x) * (x - avg_x)
    return i / j

class GymModel:
    def __init__(self, train_id, max_episodes):
        self.env = gym.make('CartPole-v0')
        self.train_id = train_id
        self.max_episodes = max_episodes

    def __fini__(self):
        self.env.close()

    def get_hyperparameters_name(self, generation_id):
        return 'hyperparameters' + str(generation_id) + '-' + str(self.train_id)

    def get_model_name(self, generation_id):
        return 'model' + str(generation_id) + '-' + str(self.train_id)

    def get_plot_name(self, generation_id):
        return 'plot' + str(generation_id) + '-' + str(self.train_id) + '.png'

    def get_data_name(self, generation_id):
        return 'data' + str(generation_id) + '-' + str(self.train_id)

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
        with open(self.get_hyperparameters_name(generation), 'w+') as f:
            f.write(str(hyperparameters) + '\n')
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
                q_values = model.predict(np.array([observation]))[0]

                random_ratio = hyperparameters[1]
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
                total_reward += reward
                data.append([observation, reward, q_values, action_id, action_q_value])

                if rendering:
                    self.env.render()

                if done:
                    print('[' + str(self.train_id) + '] Episode ' + str(e) + ' fails after ' + str(t) + ' timesteps')
                    break;

                observation = next_observation
                t += 1;

            total_rewards.append(total_reward)
            timesteps_count.append(t)

            if e != 0 and e % 10 == 0:
                self.replay(model, data, hyperparameters)
                model.save(self.get_model_name(generation))
                write_csv(self.get_data_name(generation), data)
                data.clear()

        self.env.close()

        # TODO Fix: cannot work outside of main thread
        #plt.xlabel('Episode')
        #plt.ylabel('Total reward/Timesteps count')
        #plt.plot(total_rewards)
        #plt.plot(timesteps_count)
        #plt.savefig(self.get_plot_name(generation))
        #plt.clf()

        return get_score(timesteps_count)

    def replay(self, model, data, hyperparameters):
        print('[' + str(self.train_id) + '] Training...')

        for d in data:
            observation = d[0]
            reward = d[1]
            q_values = d[2]
            action_id = d[3]
            action_q_value = d[4]

            next_q_values = model.predict(np.array([observation]))[0]
            next_action_id = get_max(next_q_values)
            next_action_q_value = next_q_values[next_action_id]

            new_q_value = (1. - hyperparameters[2]) * action_q_value + hyperparameters[2] * (reward + hyperparameters[3] * next_action_q_value)
            new_q_values = q_values
            new_q_values[action_id] = new_q_value
            model.fit(np.array([observation]),
                    np.array([new_q_values]),
                    epochs=10,
                    verbose=0)

        print('[' + str(self.train_id) + '] Training ended')
