import gym
import tensorflow as tf
import numpy as np

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
			q_values = model.predict(observation)
			max_q_value = numpy.amax(q_value)
			max_q_value_id = numpy.where(q_value == max_q_value)

			action = env.action_space[max_q_value_id]
			_, reward, done, _ = env.step(action)

			learning_rate = 0.5 # TODO Tune
			discount_factor = 0.9 # TODO Tune
			epochs_count = 10 # TODO Tune
			new_q_value = (1. - learning_rate) * max_q_value + learning_rate * (reward + discount_factor * max_q_value) # TODO fix

			new_q_values = q_values
			new_q_values[max_q_value_id] = new_q_value
			model.fit(observation, new_q_values, epochs=epochs_count)

			if done:
				print('Episode', e, 'fails after', t, 'timesteps')
				break;
			t += 1;
		e += 1

env = gym.make('CartPole-v0')
train(env)
env.close()
