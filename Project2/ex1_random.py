import gym

# problem choice
env_name = 'FrozenLake-v0'
# env_name = 'Taxi-v1'

# alterable parameters
no_of_episodes = 100
no_of_moves = 100

# other global variables
no_of_successes = 0


def main():
	global no_of_successes
	
	# create environment
	env = gym.make(env_name)
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		env.reset()
		
		print('\n\n*** NEW EPISODE STARTED ***')
		
		for m in range(no_of_moves):
			# show graphical depiction of current environment
			env.render()
			
			# choose random action
			action = env.action_space.sample()
			
			# perform action
			# return values are of type object, float, boolean, dict
			observation, reward, done, info = env.step(action)
			
			# if agent has reached a terminal state (either fail or success)
			if done:
				# if taxi is done, it has succeeded
				# if frozen lake is done, reward must be 1 for success
				if env_name == 'Taxi-v1' or reward == 1:
					no_of_successes += 1
				break
	
	print('Out of {} episodes, {} ended in success and {} ended in failure'.format(
		no_of_episodes, no_of_successes, no_of_episodes - no_of_successes))
	
main()
