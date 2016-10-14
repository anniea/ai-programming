import gym
import pickle
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

# problem choice
env_name = 'FrozenLake-v0'

# alterable parameters
no_of_episodes = 100000
no_of_moves = 100
learning_rate = 0.0625
discount_rate = 0.99
epsilon = 0.1

# other global variables
no_of_successes = 0
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
	global no_of_successes, total_moves
	
	# create environment
	env = gym.make(env_name)
	
	# start recording of environment for upload
	# recording_path = 'recordings/' + env_name + '/FrozenLake-v0-trial-' + str(len(listdir('recordings/' + env_name)))
	# env.monitor.start(recording_path)
	
	# initialize state-action value estimate
	# actions on x-axis, states on y-axis
	q_table = np.zeros((env.action_space.n, env.observation_space.n))
	# q_table = load_q_table(0)
	
	# array to hold total reward for each episode
	total_rewards = np.array([0.0]*no_of_episodes)
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()
		
		# print('\n\n*** NEW EPISODE STARTED ***')
		
		for m in range(no_of_moves):
			# show graphical depiction of current environment
			# print('\nSELECT ACTION FOR:')
			# env.render()
			
			# choose action epsilon-greedily (with prob. 1-epsilon)
			if 1 - epsilon > np.random.random():
				action = np.argmax(q_table[:, observation])
			else:
				# print('\nRandom action taken.')
				action = np.random.randint(env.action_space.n)
				
			# print('Action taken:', action, '(' + ACTION_MAP[action] + ')')
			
			# save current observation before action is performed
			prev_observation = observation
			
			# perform action
			# return values are of type object, float, boolean, dict
			observation, reward, done, info = env.step(action)
			
			# update total reward for current episode
			total_rewards[i_episode] += reward
			
			# update state-action value estimate based on recent experience
			potential_future_reward = np.amax(q_table[:, observation])
			q_table[action, prev_observation] += learning_rate * (
				reward + (discount_rate * potential_future_reward) - q_table[action, prev_observation])
		
			# if agent has reached a terminal state (either fail or success)
			if done:
				# if reward is 1, agent has reached goal
				if reward == 1:
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
	
	trial_no = len(listdir('ex3_plots'))
	
	plot_episode_rewards(total_rewards, trial_no)

	save_q_table(q_table, trial_no)
	
	print('\nQ-table:')
	for action_list in q_table:
		for value in action_list:
			print('{0:.5f}'.format(value), end=' ')
		print('')
	
	print('\n\nAverage number of moves before termination: ', total_moves/no_of_episodes)
	print('Out of {} episodes, {} ended in success and {} ended in failure'.format(
		no_of_episodes, no_of_successes, no_of_episodes - no_of_successes))


def plot_episode_rewards(total_rewards, figure_nr):
	# add random value to total rewards to better visualize data in plot
	total_rewards += (np.random.randint(-300, 301, len(total_rewards)) / 1000)
	plt.figure(figsize=(20, 10))
	plt.plot(total_rewards, 'g.', ms=5.0)
	plt.title('Total reward per episode')
	plt.xlabel('Episode #')
	plt.ylabel('Total reward')
	plt.ylim(-0.5, 1.5)
	plt.yticks([0, 1])
	plt.savefig('ex3_plots/trial_' + str(figure_nr) + '_total_rewards.png')
	plt.clf()


# save q-table for later use
def save_q_table(q_table, table_nr):
	
	# save q-table to given folder with unused name
	file = open('ex3_tables/q_table_' + str(table_nr) + '.pkl', 'wb')
	pickle.dump(q_table, file, 2)
	file.close()

	# print relevant information about saved q-table
	print('\nQ-table saved as q_table_' + str(table_nr) + '.pkl')


# load q-table for use
def load_q_table(table_nr):
	
	# load q-table with given table number
	file = open('ex3_tables/q_table_' + str(table_nr) + '.pkl', 'rb')
	q_table = pickle.load(file)
	file.close()
	
	# print relevant information about loaded q-table
	print('\nQ-table nr. ' + str(table_nr) + ' loaded')
	print('Dimensions are {}'.format(q_table.shape))
	
	return q_table


main()
