import pickle
import numpy as np
import matplotlib.pyplot as plt


# choose action epsilon-greedily (greedy with prob. 1-epsilon)
def choose_action_eps_greedy(q_table, observation, epsilon, no_of_actions):
	if 1 - epsilon > np.random.random():
		return np.argmax(q_table[:, observation])
	else:
		return np.random.randint(no_of_actions)
	
	
# plot average rewards per episode for FrozenLake-v0
def plot_frozen_lake_rewards(total_rewards, plot_path, figure_nr):
	plt.figure(figsize=(20, 10))
	plt.plot(total_rewards, 'b-')
	plt.title('Average Reward per Episode for Frozen Lake Environment')
	plt.xlabel('Episode #')
	plt.ylabel('Average Reward')
	plt.savefig(plot_path + '/flv0_' + str(figure_nr) + '_avg_rewards.png')
	plt.clf()


# plot totals rewards per episode for Taxi-v1
def plot_taxi_rewards(total_rewards, plot_path, figure_nr):
	plt.figure(figsize=(20, 10))
	plt.plot(total_rewards, 'g.', ms=5.0)
	plt.title('Total Reward per Episode for Taxi Environment')
	plt.xlabel('Episode #')
	plt.ylabel('Total Reward')
	plt.yticks([-500, 50])
	plt.savefig(plot_path + '/taxiv1_' + str(figure_nr) + '_total_rewards.png')
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
	for action_values in q_table:
		for value in action_values:
			print('{0:.5f}'.format(value), end=' ')
		print('')
