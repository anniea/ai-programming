import gym
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

# problem choice
env_name = 'FrozenLake-v0'

# alterable parameters
no_of_episodes = 100
no_of_moves = 100
my = 0.1
gamma = 0.5
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
	q_table = np.random.rand(env.action_space.n, env.observation_space.n)
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()
		
		for m in range(no_of_moves):
			# show graphical depiction of current environment
			print('SELECT ACTION FOR:')
			env.render()
			
			# choose action greedily (2b)
			action = np.argmax(q_table[:, observation])
			
			# choose action epsilon-greedily (with prob. 1-epsilon) (2c)
			# if 1 - epsilon > np.random.random():
			# 	action = np.argmax(q_table[:, observation])
			# else:
			# 	action = np.random.randint(env.action_space.n)
			
			# save current observation before action is performed
			prev_observation = observation
			
			# perform action
			# return values are of type object, float, boolean, dict
			observation, reward, done, info = env.step(action)
			
			# if agent has reached a terminal state (either fail or success)
			if done:
				# if not at goal, agent has failed
				if observation != 15:
					print('Agent failed.')
					no_of_fails += 1
				else:
					no_of_successes += 1
				print('Episode finished after {} moves'.format(m + 1))
				break
			
			# some prints to validate beliefs
			print('Old observation:', observation)
			print('New observation:', observation)
			print('Action taken:', action, '(' + ACTION_MAP[action] + ')')
			print('Reward gained:', reward, '\n\n')
			
			# update state-action value estimate based on recent experience
			potential_future_reward = np.amax(q_table[:, observation])
			q_table[action, prev_observation] += my * (
				reward + (gamma * potential_future_reward) - q_table[action, prev_observation])
	
	# env.monitor.close()
	
	# upload to OpenAI Gym (not bothering with this just yet)
	# gym.upload(
	# 	recording_path,
	# 	writeup='https://gist.github.com/gdb/b6365e79be6052e7531e7ba6ea8caf23',
	# 	api_key='YOUR_API_KEY')
	
	print('\n\nOut of {} episodes, {} ended in success and {} ended in failure'.format(no_of_episodes, no_of_successes,
																					   no_of_fails))

main()
