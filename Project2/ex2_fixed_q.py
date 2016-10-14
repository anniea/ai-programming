import gym
import numpy as np
from os import listdir
from helpers import choose_action_eps_greedy

# problem choice
env_name = 'FrozenLake-v0'

# alterable parameters
no_of_episodes = 100
no_of_moves = 100
epsilon = 0.4

# other global variables
no_of_successes = 0
no_of_fails = 0
ACTION_MAP = ['left', 'down', 'right', 'up']  # used for printing


#####################################################################
# IMPORTANT NOTE FOR FROZEN LAKE:									#
# OBSERVATION RANGES FROM 0-15, WHERE EACH NUMBER n DENOTE TO THE	#
# STATE WHERE n == ((4*i)+j) FOR AGENT POSITION (i, j) 				#
# ACTIONS RANGES FROM 0-3, WITH THE FOLLOWING MAPPING:				#
# 0->LEFT, 1->DOWN, 2->RIGHT, 3->UP									#
#####################################################################


def main():
	global no_of_successes, no_of_fails
	
	# create environment
	env = gym.make(env_name)
	
	# start recording of environment for upload
	# recording_path = 'recordings/' + env_name + '/FrozenLake-v0-trial-' + str(len(listdir('recordings/' + env_name)))
	# env.monitor.start(recording_path)
	
	# initialize state-action value estimate
	# actions on x-axis, states on y-axis
	q_table = np.full((env.action_space.n, env.observation_space.n), 0.5)
	
	# make value for southbound choices 1 (action 1 is down)
	q_table[1] = 1
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()
		
		# print('\n\n*** NEW EPISODE STARTED ***')
		
		for m in range(no_of_moves):
			# show graphical depiction of current environment
			# print('\nSELECT ACTION FOR:')
			# env.render()
			
			# choose action greedily (2b)
			# action = np.argmax(q_table[:, observation])
			
			# choose action epsilon-greedily (with prob. 1-epsilon) (2c)
			action = choose_action_eps_greedy(q_table, observation, epsilon, env.action_space.n)

			# print('Action taken:', action, '(' + ACTION_MAP[action] + ')')
			
			# perform action
			# return values are of type object, float, boolean, dict
			observation, reward, done, info = env.step(action)
			
			# task 2 wants fixed q-values, so we do not update q_table
			
			# if agent has reached a terminal state (either fail or success)
			if done:
				# if reward is 1, agent has reached goal
				if reward == 1:
					no_of_successes += 1
				# print('Episode finished after {} moves'.format(m + 1))
				break
			
	# env.monitor.close()
	
	# upload to OpenAI Gym (not bothering with this just yet)
	# gym.upload(
	# 	recording_path,
	# 	writeup='https://gist.github.com/gdb/b6365e79be6052e7531e7ba6ea8caf23',
	# 	api_key='YOUR_API_KEY')
	
	print('Out of {} episodes, {} ended in success and {} ended in failure'.format(
		no_of_episodes, no_of_successes, no_of_episodes - no_of_successes))


main()
