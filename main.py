import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
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
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def __fini__(self):
        self.env.close()

    def play(self):
        model = tf.keras.models.load_model('model')

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

            print('Simulation ended after ' + str(t) + ' timesteps')

    def train(self, train_id, max_episodes, learning_rate, q_learning_rate, discount_factor, rendering=False):
        model_location = 'model' + str(train_id)
        if os.path.exists(model_location):
            model = tf.keras.models.load_model(model_location)
        else:
            model = init_model(learning_rate)

        total_rewards = []
        timesteps_count = []
        data = []

        for e in range(max_episodes):
            self.env.reset()
            if rendering:
                self.env.render()
            observation, _, _, _ = self.env.step(self.env.action_space.sample())

            total_reward = 0.

            t = 0
            while True:
                print('State: ' + str(np.array([observation])))
                q_values = model.predict(np.array([observation]))[0]
                print('Q-Values: ' + str(q_values))

                random_ratio = 0.1 # Tune
                if random.uniform(0., 1.) < random_ratio:
                    action_id = self.env.action_space.sample()
                else:
                    action_id = get_max(q_values)
                action_q_value = q_values[action_id]

                next_observation, reward, done, _ = self.env.step(action_id)
                reward = 100.
                reward -= (abs(observation[0]) / 0.24) * 30.
                reward -= (abs(observation[2]) / 0.20) * 70.
                if done:
                    reward = -100.
                print('reward: ' + str(reward))
                total_reward += reward
                data.append((observation, reward, q_values, action_id, action_q_value))

                if rendering:
                    self.env.render()

                if done:
                    print('Episode ' + str(e) + ' fails after ' + str(t) + ' timesteps')
                    break;

                observation = next_observation
                t += 1;

            total_rewards.append(total_reward)
            timesteps_count.append(t)

            if e % 10 == 0:
                self.replay(model, data, q_learning_rate, discount_factor)
                data.clear()
                model.save(model_location)

        self.env.close()

        plt.xlabel('Episode')
        plt.ylabel('Total reward/Timesteps count')
        plt.plot(total_rewards)
        plt.plot(timesteps_count)
        plt.savefig('plot' + str(train_id) + '.png')
        plt.clf()
        #plt.show()

    def replay(self, model, data, q_learning_rate, discount_factor):
        print('Training...')

        for d in data:
            observation = d[0]
            reward = d[1]
            q_values = d[2]
            action_id = d[3]
            action_q_value = d[4]

            print('Next state: ' + str(np.array([observation])))
            next_q_values = model.predict(np.array([observation]))[0]
            print('Next Q-Values: ' + str(q_values))
            next_action_id = get_max(next_q_values)
            next_action_q_value = next_q_values[next_action_id]

            new_q_value = (1. - q_learning_rate) * action_q_value + q_learning_rate * (reward + discount_factor * next_action_q_value)
            new_q_values = q_values
            new_q_values[action_id] = new_q_value
            print('New Q-Values: ' + str(new_q_values))
            model.fit(np.array([observation]),
                    np.array([new_q_values]),
                    epochs=10)

def train_():
    train_id=0

    for i in range(1, 6):
        for j in range(1, 10):
            for k in range(1, 10):
                learning_rate=10e-6 * i
                q_learning_rate=0.1 * j
                discount_factor=0.1 * k

                print('Starting training with learning_rate=' + str(learning_rate) + ', q_learning_rate=' + str(q_learning_rate) + ', discount_factor=' + str(discount_factor))
                model = GymModel()
                model.train(train_id, 100, learning_rate, q_learning_rate, discount_factor, rendering=False)
                train_id += 1

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        train_()
    else:
        print('Starting simulation from training data...')
        model = GymModel()
        model.play()

main()
