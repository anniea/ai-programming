import gym
from os import listdir

# problem choice
env_name = 'FrozenLake-v0'
# env_name = 'Taxi-v1'

# alterable parameters
no_of_episodes = 20
no_of_trials = 100


def main():
	# create environment
	env = gym.make(env_name)
	
	# start recording of environment for upload
	# recording_path = 'recordings/' + env_name + '/FrozenLake-v0-trial-' + str(len(listdir('recordings/' + env_name)))
	# env.monitor.start(recording_path)
	
	for i_episode in range(no_of_episodes):
		# set environment initial state
		observation = env.reset()
		
		for t in range(no_of_trials):
			# show graphical depiction of current environment
			env.render()
			
			# print state identifier
			print(observation, '\n')
			
			# choose random action
			action = env.action_space.sample()
			
			# perform action
			# return values are of type object, float, boolean, dict
			observation, reward, done, info = env.step(action)
			
			# if agent has reached a terminal state (either fail or success)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
	
	# env.monitor.close()
	
	# gym.upload(
	# 	recording_path,
	# 	writeup='https://gist.github.com/gdb/b6365e79be6052e7531e7ba6ea8caf23',
	# 	api_key='YOUR_API_KEY')
	
main()

