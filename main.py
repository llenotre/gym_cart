import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tensorflow as tf

MODEL_LOCATION = 'model'

def init_model():
    activation = tf.keras.layers.LeakyReLU()
    layers = []
    for i in range(5):
        layers.append(tf.keras.layers.Dense(4 << i, activation=activation))
    layers.append(tf.keras.layers.Dense(2, activation=activation))

    model = tf.keras.Sequential(layers)
    model.compile(optimizer='adam', loss='mse')
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
        model = tf.keras.models.load_model(MODEL_LOCATION)

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

    def train(self, max_episodes, rendering=False):
        if os.path.exists(MODEL_LOCATION):
            model = tf.keras.models.load_model(MODEL_LOCATION)
        else:
            model = init_model()

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
                reward *= 10.
                reward -= (abs(observation[0]) / 0.24 - 0.5) * 10.
                reward -= (abs(observation[2]) / 0.20 - 0.5) * 20.
                if done:
                    reward -= 100.
                print('reward: ' + str(reward))
                total_reward += reward
                data.append((observation, reward, q_values, action_q_value))

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
                print('Training...')

                for d in data:
                    observation = d[0]
                    reward = d[1]
                    q_values = d[2]
                    action_q_value = d[3]

                    print('Next state: ' + str(np.array([observation])))
                    next_q_values = model.predict(np.array([observation]))[0]
                    print('Next Q-Values: ' + str(q_values))
                    next_action_id = get_max(next_q_values)
                    next_action_q_value = next_q_values[next_action_id]

                    learning_rate = 0.7 # TODO Tune
                    discount_factor = 1.0 # TODO Tune
                    epochs_count = 10 # TODO Tune
                    new_q_value = (1. - learning_rate) * action_q_value + learning_rate * (reward + discount_factor * next_action_q_value)
                    new_q_values = q_values
                    new_q_values[action_id] = new_q_value
                    print('New Q-Values: ' + str(new_q_values))
                    model.fit(np.array([observation]),
                            np.array([new_q_values]),
                            epochs=epochs_count)

                data.clear()
                model.save(MODEL_LOCATION)

        self.env.close()

        plt.xlabel('Episode')
        plt.ylabel('Total reward/Timesteps count')
        plt.plot(total_rewards)
        plt.plot(timesteps_count)
        plt.show()

def main():
    model = GymModel()
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        print('Starting training...')
        model.train(100, rendering=True)
    else:
        print('Starting simulation from training data...')
        model.play()

main()
