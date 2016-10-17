import gym
import numpy as np
from os import listdir
from helpers import plot_frozen_lake_rewards, save_q_table, load_q_table, print_q_table, choose_action_eps_greedy

# problem choice
env_name = 'FrozenLake-v0'

# alterable parameters
no_of_episodes = 25000
no_of_moves = 100
learning_rate = 0.1
discount_rate = 0.99
epsilon = 0.1
start_eps = epsilon

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


def rele(lr, e):
	global no_of_successes, epsilon, total_moves, learning_rate, epsilon, start_eps
	
	no_of_successes = 0
	total_moves = 0
	learning_rate = lr
	epsilon = e
	start_eps = e
	
	# create environment
	env = gym.make(env_name)
	
	# start recording of environment for upload
	# recording_path = 'recordings/' + env_name + '/FrozenLake-v0-trial-' + str(len(listdir('recordings/' + env_name)))
	# env.monitor.start(recording_path)
	
	# initialize state-action value estimate
	# actions on x-axis, states on y-axis
	q_table = np.zeros((env.action_space.n, env.observation_space.n))
	# q_table = load_q_table('ex3_plots', 0)
	
	# array to hold total reward for each episode
	total_rewards = np.array([0.0] * no_of_episodes)
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()
		
		# old_q_table = [action_values[:] for action_values in q_table]
		
		# print('\n\n*** NEW EPISODE STARTED ***')
		
		for m in range(no_of_moves):
			# show graphical depiction of current environment
			# print('\nSELECT ACTION FOR:')
			# env.render()
			
			# choose action epsilon-greedily (with prob. 1-epsilon)
			action = choose_action_eps_greedy(q_table, observation, epsilon, env.action_space.n)
			
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
					epsilon *= 0.99
				total_moves += m + 1
				# print('Episode finished after {} moves'.format(m + 1))
				break
		
		# if (i_episode + 1) % 100 == 0:
		# 	epsilon -= (start_eps - 0.01) / (no_of_episodes / 100)
		# print(epsilon)
		
		# stable = True
		# old_q_table -= q_table
		# for action_values in np.absolute(old_q_table):
		# 	for value in action_values:
		# 		if value > 0.01:
		# 			stable = False
		# 			break
		#
		# if stable:
		# 	print('Q-table has converged in episode', i_episode)
		# 	print_q_table(q_table)
	
	# env.monitor.close()
	
	# upload to OpenAI Gym (not bothering with this just yet)
	# gym.upload(
	# 	recording_path,
	# 	writeup='https://gist.github.com/gdb/b6365e79be6052e7531e7ba6ea8caf23',
	# 	api_key='YOUR_API_KEY')
	
	trial_no = len(listdir('ex3_plots'))
	
	# plot_frozen_lake_rewards(total_rewards, 'ex3_plots', trial_no)
	#
	# save_q_table(q_table, 'ex3_plots', trial_no)
	
	# print_q_table(q_table)
	
	# print('\n\nAverage number of moves before termination: ', total_moves / no_of_episodes)
	# print('Out of {} episodes, {} ended in success and {} ended in failure'.format(
	# 	no_of_episodes, no_of_successes, no_of_episodes - no_of_successes))

	return no_of_successes


def main():
	
	top_100 = []
	
	lr = 0.1
	
	while lr < 1.01:
		
		e = 0.1
		while e < 1.01:
			avg_success = 0
			for i in range(5):
				avg_success += rele(lr, e)
			top_100.append((lr, e, avg_success/5))
			e += 0.1
		
		lr += 0.1

	top_100 = sorted(top_100, key=lambda res: res[2], reverse=True)
	
	for i in range(100):
		print('Ranking', i+1, '-', top_100[i])
		
	print('Done.')

main()
