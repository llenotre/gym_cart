import gym
import tensorflow as tf
import numpy as np

def get_max(arr):
	max_i = 0
	for i in range(1, len(arr)):
		if arr[i] > arr[max_i]:
			max_i = i
	return max_i

def train(env):
	layers = []
	layers.append(tf.keras.layers.Dense(4, activation='relu'))
	for i in range(20):
		layers.append(tf.keras.layers.Dense(20, activation='relu'))
	layers.append(tf.keras.layers.Dense(2, activation='relu'))

	model = tf.keras.Sequential(layers)
	model.compile(optimizer='adam', loss='mse', metrics=['mae'])

	e = 0
	while True:
		env.reset()
		env.render()
		observation, _, _, _ = env.step(env.action_space.sample())

		t = 0
		while True:
			in_data = np.array([observation])
			q_values = model.predict(in_data)[0]
			max_q_value_id = get_max(q_values)
			max_q_value = q_values[max_q_value_id]

			env.render()
			next_observation, reward, done, _ = env.step(max_q_value_id)
			if done:
				reward = -1000
			print('reward: ' + str(reward))

			learning_rate = 0.5 # TODO Tune
			discount_factor = 0.9 # TODO Tune
			epochs_count = 10 # TODO Tune
			print('old Q-Values: ' + str(q_values))
			new_q_value = (1. - learning_rate) * max_q_value + learning_rate * (reward + discount_factor * max_q_value) # TODO fix

			new_q_values = q_values
			new_q_values[max_q_value_id] = new_q_value
			print('new Q-Values: ' + str(new_q_values))
			x = np.array([observation])
			y = np.array([new_q_values])
			model.fit(x, y, epochs=epochs_count)

			if done:
				print('Episode' + str(e) + 'fails after' + str(t) + 'timesteps')
				break;

			observation = next_observation
			t += 1;
		e += 1

env = gym.make('CartPole-v0')
train(env)
env.close()
