import gym
import numpy as np
from os import listdir
from helpers import plot_frozen_lake_rewards, plot_frozen_lake_avg_rewards, save_q_table, print_q_table, choose_action_eps_greedy

# problem choice
env_name = 'FrozenLake-v0'

# alterable parameters
no_of_episodes = 50000
no_of_moves = 100
learning_rate = 0.3
discount_rate = 0.99
epsilon = 0.5

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
	global no_of_successes, epsilon, total_moves
	
	# create environment
	env = gym.make(env_name)
	
	# initialize state-action value estimate
	# actions on x-axis, states on y-axis
	q_table = np.zeros((env.action_space.n, env.observation_space.n))
	
	# array to hold total reward for every episode
	total_rewards = np.array([0.0]*no_of_episodes)
	
	# array to hold average reward across every finished episode
	average_rewards = np.array([0.0]*no_of_episodes)
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()
		
		# print('\n\n*** NEW EPISODE STARTED ***')
		for m in range(no_of_moves):
			# show graphical depiction of current environment
			# print('\nSELECT ACTION FOR:')
			# env.render()
			
			# choose action epsilon-greedily (greedy with prob. 1-epsilon)
			action = choose_action_eps_greedy(q_table, observation, epsilon, env.action_space.n)
				
			# print('Action taken:', action, '(' + ACTION_MAP[action] + ')')
			
			# save current observation before action is performed
			prev_observation = observation
			
			# perform action
			# return values are of type object, float, boolean, dict
			observation, reward, done, info = env.step(action)
			
			# update state-action value estimate based on recent experience
			potential_future_reward = np.amax(q_table[:, observation])
			q_table[action, prev_observation] += learning_rate * (
				reward + (discount_rate * potential_future_reward) - q_table[action, prev_observation])
		
			# if agent has reached a terminal state (either fail or success)
			if done:
				# if reward is 1, agent has reached goal
				if reward == 1:
					no_of_successes += 1
					# reduce epsilon to reduce exploration
					epsilon *= 0.99
				total_moves += m + 1
				break
		
		# update total reward and average reward with reward from current episode
		total_rewards[i_episode] = reward
		average_rewards[i_episode] = np.sum(total_rewards[:i_episode+1]) / (i_episode + 1.0)
		
	trial_no = len(listdir('ex3_tables'))
	
	plot_frozen_lake_rewards(total_rewards, 'ex3_plots', trial_no)
	
	plot_frozen_lake_avg_rewards(average_rewards, 'ex3_plots', trial_no)

	save_q_table(q_table, 'ex3_tables', trial_no)
	
	print_q_table(q_table)
	
	print('\n\nAverage number of moves before termination: ', total_moves/no_of_episodes)
	print('Out of {} episodes, {} ended in success and {} ended in failure'.format(
		no_of_episodes, no_of_successes, no_of_episodes - no_of_successes))

main()
