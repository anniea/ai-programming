import gym
import pickle
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

# problem choice
env_name = 'FrozenLake-v0'

# alterable parameters
no_of_episodes = 20000
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
	# recording_path = 'recordings/' + env_name + '/FrozenLake-v0-trial-' + str(len(listdir('recordings/' + env_name)))
	# env.monitor.start(recording_path)

	# initialize state-action value estimate
	# actions on x-axis, states on y-axis
	# q_table = np.random.rand(env.action_space.n, env.observation_space.n)
	q_table = np.zeros((env.action_space.n, env.observation_space.n))
	ex3_q_table = load_q_table(2)

	# array to hold total reward for each episode
	total_rewards = np.array([0.0]*no_of_episodes)

	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()

		# print('\n\n*** NEW EPISODE STARTED ***')

		for m in range(no_of_moves):
			# show graphical depiction of current environment
			# print('SELECT ACTION FOR:')
			# env.render()

			# choose action epsilon-greedily (with prob. 1-epsilon)
			if 1 - epsilon > np.random.random():
				action = np.argmax(q_table[:, observation])
			else:
				action = np.random.randint(env.action_space.n)

			# save current observation before action is performed
			prev_observation = observation

			# perform action
			# return values are of type object, float, boolean, dict
			observation, reward, done, info = env.step(action)

			# update total reward for current episode
			total_rewards[i_episode] += reward

			# update state-action value estimate based on recent experience
			potential_future_reward = np.amax(ex3_q_table[:, observation])
			q_table[action, prev_observation] += learning_rate * (
				reward + (discount_rate * potential_future_reward) - q_table[action, prev_observation])

			# some prints to validate beliefs
			# print('Old observation:', observation)
			# print('New observation:', observation)
			# print('Action taken:', action, '(' + ACTION_MAP[action] + ')')
			# print('Reward gained:', reward, '\n\n')

			# if agent has reached a terminal state (either fail or success)
			if done:
				# if not at goal, agent has failed
				if observation != 15:
					# print('Agent failed.')
					no_of_fails += 1
				else:
					no_of_successes += 1
				total_moves += m + 1
				# print('Episode finished after {} moves'.format(m + 1))
				break

	# env.monitor.close()

	# upload to OpenAI Gym (not bothering with this just yet)
	# gym.upload(
	# 	recording_path,
	# 	writeup='https://gist.github.com/gdb/b6365e79be6052e7531e7ba6ea8caf23',
	# 	api_key='YOUR_API_KEY')

	print('')
	for thing in q_table:
		for entry in thing:
			print(entry, end=' ')
		print('')
	print('')

	print('Average number of moves before termination: ', total_moves/no_of_episodes)
	print('\n\nOut of {} episodes, {} ended in success and {} ended in failure'.format(no_of_episodes, no_of_successes,
																					   no_of_fails))


# load q-table for use / demonstration
def load_q_table(table_nr):

	# load q-table with given table number
	file = open('ex3_tables/q_table_' + str(table_nr) + '.pkl', 'rb')
	q_table = pickle.load(file)
	file.close()

	# print relevant information about loaded network
	print('\nQ-table nr. ' + str(table_nr) + ' loaded')
	print('Dimensions are {}'.format(q_table.shape))

	return q_table


main()