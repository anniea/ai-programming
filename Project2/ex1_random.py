import gym
from os import listdir

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
	
	# start recording of environment for upload
	# recording_path = 'recordings/' + env_name + '/FrozenLake-v0-trial-' + str(len(listdir('recordings/' + env_name)))
	# env.monitor.start(recording_path)
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()
		
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
				# print('Episode finished after {} moves'.format(m + 1))
				break
	
	# env.monitor.close()
	
	# gym.upload(
	# 	recording_path,
	# 	writeup='https://gist.github.com/gdb/b6365e79be6052e7531e7ba6ea8caf23',
	# 	api_key='YOUR_API_KEY')
	
	print('Out of {} episodes, {} ended in success and {} ended in failure'.format(
		no_of_episodes, no_of_successes, no_of_episodes - no_of_successes))
	
main()

