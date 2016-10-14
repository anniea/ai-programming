import pickle
import numpy as np
import matplotlib.pyplot as plt


# choose action epsilon-greedily (with prob. 1-epsilon)
def choose_action_eps_greedy(q_table, observation, epsilon, no_of_actions):
	if 1 - epsilon > np.random.random():
		return np.argmax(q_table[:, observation])
	else:
		# print('\nRandom action taken.')
		return np.random.randint(no_of_actions)
		
	
# plot totals rewards per episode for FrozenLake-v0
def plot_frozen_lake_rewards(total_rewards, plot_path, figure_nr):
	# add random value to total rewards to better visualize data in plot
	total_rewards += (np.random.randint(-300, 301, len(total_rewards)) / 1000)
	plt.figure(figsize=(20, 10))
	plt.plot(total_rewards, 'g.', ms=5.0)
	plt.title('Total reward per Episode for Frozen Lake Environment')
	plt.xlabel('Episode #')
	plt.ylabel('Total reward')
	plt.ylim(-0.5, 1.5)
	plt.yticks([0, 1])
	plt.savefig(plot_path + '/trial_' + str(figure_nr) + '_total_rewards.png')
	plt.clf()


# plot totals rewards per episode for Taxi-v1
def plot_taxi_rewards(total_rewards, plot_path, figure_nr):
	plt.figure(figsize=(20, 10))
	plt.plot(total_rewards, 'g.', ms=5.0)
	plt.title('Total Reward per Episode for Taxi Environment')
	plt.xlabel('Episode #')
	plt.ylabel('Total Reward')
	plt.savefig(plot_path + '/trial_' + str(figure_nr) + '_total_rewards.png')
	plt.clf()


# save q-table for later use
def save_q_table(q_table, table_path, table_nr):
	
	# save q-table to given path with unused name
	file = open(table_path + '/q_table_' + str(table_nr) + '.pkl', 'wb')
	pickle.dump(q_table, file, 2)
	file.close()

	# print relevant information about saved q-table
	print('\nQ-table saved as q_table_' + str(table_nr) + '.pkl')


# load q-table for use
def load_q_table(table_path, table_nr):
	
	# load q-table with given path and table number
	file = open(table_path + '/q_table_' + str(table_nr) + '.pkl', 'rb')
	q_table = pickle.load(file)
	file.close()
	
	# print relevant information about loaded q-table
	print('\nQ-table nr. ' + str(table_nr) + ' loaded')
	print('Dimensions are {}'.format(q_table.shape))
	
	return q_table


# print q-table in a readable way
def print_q_table(q_table):
	print('\nQ-table:')
	for action_list in q_table:
		for value in action_list:
			print('{0:.5f}'.format(value), end=' ')
		print('')