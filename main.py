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
	layers.append(tf.keras.layers.Dense(4))
	for i in range(20):
		layers.append(tf.keras.layers.Dense(20, activation='relu'))
	layers.append(tf.keras.layers.Dense(2))

	model = tf.keras.Sequential(layers)
	model.compile(optimizer='adam',
		  loss=tf.keras.losses.MeanSquaredError(),
		  metrics=['accuracy'])

	e = 0
	while True:
		env.reset()

		t = 0
		while True:
			env.render()

			observation = env.observation_space
			q_values = model.predict(observation.high)[0]
			max_q_value_id = get_max(q_values)
			max_q_value = q_values[max_q_value_id]
			_, reward, done, _ = env.step(max_q_value_id)

			learning_rate = 0.5 # TODO Tune
			discount_factor = 0.9 # TODO Tune
			epochs_count = 10 # TODO Tune
			new_q_value = (1. - learning_rate) * max_q_value + learning_rate * (reward + discount_factor * max_q_value) # TODO fix

			new_q_values = q_values
			new_q_values[max_q_value_id] = new_q_value
			model.fit(np.transpose([observation.high]), np.transpose([new_q_values]), epochs=epochs_count)

			if done:
				print('Episode', e, 'fails after', t, 'timesteps')
				break;
			t += 1;
		e += 1

env = gym.make('CartPole-v0')
train(env)
env.close()
