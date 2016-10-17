import gym
import numpy as np
from helpers import choose_action_eps_greedy

# problem choice
env_name = 'FrozenLake-v0'

# alterable parameters
no_of_episodes = 25000
no_of_moves = 100
discount_rate = 0.99


def ex3_modified(learning_rate, epsilon):

	no_of_successes = 0
	
	# create environment
	env = gym.make(env_name)
	
	# initialize state-action value estimate
	# actions on x-axis, states on y-axis
	q_table = np.zeros((env.action_space.n, env.observation_space.n))
	
	# array to hold total reward for each episode
	total_rewards = np.array([0.0] * no_of_episodes)
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()
		
		for m in range(no_of_moves):
			
			# choose action epsilon-greedily (with prob. 1-epsilon)
			action = choose_action_eps_greedy(q_table, observation, epsilon, env.action_space.n)
			
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
				break
		
	return no_of_successes


def main():
	
	every_avg_success = []
	
	lr = 0.1
	
	while lr < 1.01:
		
		e = 0.1
		while e < 1.01:
			avg_success = 0
			for i in range(5):
				avg_success += ex3_modified(lr, e)
			every_avg_success.append((lr, e, avg_success/5))
			e += 0.1
		
		lr += 0.1

	every_avg_success = sorted(every_avg_success, key=lambda res: res[2], reverse=True)
	
	for i in range(100):
		print('Ranking', i+1, '-', every_avg_success[i])
		
	print('Done.')

main()
