import gym
import pickle
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

# problem choice
env_name = 'Taxi-v1'

# alterable parameters
no_of_episodes = 10000
no_of_moves = 100
learning_rate = 0.1
discount_rate = 0.99
epsilon = 0.1

# other global variables
no_of_successes = 0
no_of_fails = 0
ACTION_MAP = ['left', 'down', 'right', 'up']  # used for printing
total_moves = 0


#####################################################################
# IMPORTANT NOTE FOR FROZEN LAKE:									#
# OBSERVATION RANGES FROM 0-15, WHERE EACH NUMBER n DENOTE TO THE	#
# STATE WHERE n == ((4*i)+j) FOR AGENT POSITION (i, j) 				#
# ACTIONS RANGES FROM 0-3, WITH THE FOLLOWING MAPPING:				#
# 0->LEFT, 1->DOWN, 2->RIGHT, 3->UP									#
#####################################################################


def main():
	global no_of_successes, no_of_fails, total_moves

	# create environment
	env = gym.make(env_name)

	# start recording of environment for upload
	#recording_path = 'recordings/' + env_name + '/Taxi-v1-trial-' + str(len(listdir('recordings/' + env_name)))
	#env.monitor.start(recording_path)

	# initialize state-action value estimate
	# actions on x-axis, states on y-axis
	q_table = np.zeros((env.action_space.n, env.observation_space.n))

	# array to hold total reward for each episode
	total_rewards = np.array([0]*no_of_episodes)

	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()

		# print('\n\n*** NEW EPISODE STARTED ***')

		# choose initial action epsilon-greedily (with prob. 1-epsilon)
		if 1 - epsilon > np.random.random():
			action = np.argmax(q_table[:, observation])
		else:
			action = np.random.randint(env.action_space.n)

		# print('Initial action:', action)

		for m in range(no_of_moves):
			# show graphical depiction of current environment
			# print('SELECT ACTION FOR:')
			# env.render()

			# save current action before choosing a new one
			prev_action = action
			# save current observation before action is performed
			prev_observation = observation

			# perform action
			# return values are of type object, float, boolean, dict
			observation, reward, done, info = env.step(action)

			# update total reward for current episode
			total_rewards[i_episode] += reward

			# choose next action epsilon-greedily (with prob. 1-epsilon)
			if 1 - epsilon > np.random.random():
				action = np.argmax(q_table[:, observation])
			else:
				action = np.random.randint(env.action_space.n)

			# update state-action value estimate based on chosen action
			potential_future_reward = q_table[action, observation]
			q_table[prev_action, prev_observation] += learning_rate * (
				reward + (discount_rate * potential_future_reward) - q_table[prev_action, prev_observation])

			# if agent has reached a terminal state (either fail or success)
			if done:
				no_of_successes += 1
				total_moves += m + 1
				# print('Episode finished after {} moves'.format(m + 1))
				break

	env.monitor.close()

	# upload to OpenAI Gym (not bothering with this just yet)
	'''gym.upload(
		recording_path,
		writeup='https://gist.github.com/gdb/b6365e79be6052e7531e7ba6ea8caf23',
		api_key='mikalbj')'''

	# save_q_table(trial_no)

	print('')
	for thing in q_table:
		for entry in thing:
			print(entry, end=' ')
		print('')
	print('')

	print('Average number of moves before success: ', total_moves/no_of_successes)
	print('\n\nOut of {} episodes, {} ended in success and {} ended in failure'.format(no_of_episodes, no_of_successes,
																					   no_of_episodes - no_of_successes))




main()
