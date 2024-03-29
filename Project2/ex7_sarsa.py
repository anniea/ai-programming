import gym
import numpy as np
from os import listdir
from helpers import choose_action_eps_greedy, plot_taxi_rewards, print_q_table

# problem choice
env_name = 'Taxi-v1'

# alterable parameters
no_of_episodes = 2000
no_of_moves = 100
learning_rate = 0.4
discount_rate = 0.99
epsilon = 0.1

# other global variables
no_of_successes = 0
total_moves = 0


def main():
	global no_of_successes, total_moves

	# create environment
	env = gym.make(env_name)

	# initialize state-action value estimate
	# actions on x-axis, states on y-axis
	q_table = np.zeros((env.action_space.n, env.observation_space.n))

	# array to hold total reward for each episode
	total_rewards = np.array([0]*no_of_episodes)

	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()

		# print('\n\n*** NEW EPISODE STARTED ***')

		# choose initial action epsilon-greedily (greedy with prob. 1-epsilon)
		action = choose_action_eps_greedy(q_table, observation, epsilon, env.action_space.n)

		for m in range(no_of_moves):
			# show graphical depiction of current environment
			# print('\nSELECT ACTION FOR:')
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

			# choose next action epsilon-greedily (greedy with prob. 1-epsilon)
			action = choose_action_eps_greedy(q_table, observation, epsilon, env.action_space.n)

			# update state-action value estimate based on chosen action
			potential_future_reward = q_table[action, observation]
			q_table[prev_action, prev_observation] += learning_rate * (
				reward + (discount_rate * potential_future_reward) - q_table[prev_action, prev_observation])

			# if agent has reached a terminal state (must be success)
			if done:
				no_of_successes += 1
				total_moves += m + 1
				break

	trial_no = len(listdir('ex7_sarsa_plots'))
	plot_taxi_rewards(total_rewards, 'ex7_sarsa_plots', trial_no)

	print_q_table(q_table)
	
	print('\n\nAverage number of moves before termination: ', total_moves / no_of_episodes)
	print('Out of {} episodes, {} ended in success and {} ended in failure'.format(
		no_of_episodes, no_of_successes, no_of_episodes - no_of_successes))

main()
