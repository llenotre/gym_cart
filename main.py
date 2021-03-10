import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

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

	def train(self, max_episodes, rendering=False):
		layers = []
		layers.append(tf.keras.layers.Dense(4, activation='relu'))
		for i in range(20):
			layers.append(tf.keras.layers.Dense(20, activation='relu'))
		layers.append(tf.keras.layers.Dense(2, activation='relu'))

		model = tf.keras.Sequential(layers)
		model.compile(optimizer='adam', loss='mse')

		total_rewards = []
		timesteps_count = []

		for e in range(max_episodes):
			self.env.reset()
			if rendering:
				self.env.render()
			observation, _, _, _ = self.env.step(self.env.action_space.sample())

			total_reward = 0.

			t = 0
			while True:
				in_data = np.array([observation])
				print('State: ' + str(in_data))
				q_values = model.predict(in_data)[0]
				print('Q-Values: ' + str(q_values))

				random_rate = 0.3 # TODO Tune
				if random.uniform(0., 1.) < random_rate:
					action_id = self.env.action_space.sample()
				else:
					action_id = get_max(q_values)

				action_q_value = q_values[action_id]

				if rendering:
					self.env.render()
				next_observation, reward, done, _ = self.env.step(action_id)
				if done:
					reward = -100.
				elif abs(next_observation[2]) > observation[2]:
					reward = -1.
				print('reward: ' + str(reward))
				total_reward += reward

				in_data = np.array([observation])
				print('Next state: ' + str(in_data))
				next_q_values = model.predict(in_data)[0]
				print('Next Q-Values: ' + str(q_values))
				action_next_q_value = next_q_values[get_max(next_q_values)]

				learning_rate = 0.8 # TODO Tune
				discount_factor = 0.5 # TODO Tune
				epochs_count = 10 # TODO Tune
				new_q_value = (1. - learning_rate) * action_q_value + learning_rate * (reward + discount_factor * action_next_q_value)

				new_q_values = q_values
				new_q_values[action_id] = new_q_value
				print('New Q-Values: ' + str(new_q_values))
				x = np.array([observation])
				y = np.array([new_q_values])
				model.fit(x, y, epochs=epochs_count)

				if done:
					print('Episode ' + str(e) + ' fails after ' + str(t) + ' timesteps')
					break;

				prev_observation = observation
				observation = next_observation
				t += 1;

			total_rewards.append(total_reward)
			timesteps_count.append(t)

		self.env.close()

		plt.xlabel('Episode')
		plt.ylabel('Total reward/Timesteps count')
		plt.plot(total_rewards)
		plt.plot(timesteps_count)
		plt.show()

def main():
	model = GymModel()
	model.train(100, rendering=True)

main()
