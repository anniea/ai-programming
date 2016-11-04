import numpy as np
import matplotlib.pyplot as plt
from os import listdir


def read_tsp_data(filename):
	# get file
	file = open('data/' + filename + '.txt', 'r')
	
	# skip over non-important lines
	line = file.readline()
	while 'NODE_COORD_SECTION' not in line:
		line = file.readline()
	
	# read and process lines with coordinates
	data_points = []
	line = file.readline()
	while 'EOF' not in line:
		coordinates = line.split(' ')
		data_points.append((float(coordinates[1]), float(coordinates[2].strip())))
		line = file.readline()
	
	# make into numpy array to access columns only
	data_points = np.array(data_points)
	
	# close file
	file.close()
	
	# return scaled and unscaled data
	return data_points/np.amax(data_points), data_points


def plot_intermediate_tsp(neuron_points, city_points, num, current_distance):
	plt.figure(figsize=(20, 10))
	# plot tsp cities positions
	plt.plot(city_points[:, 0], city_points[:, 1], 'ro', label='City Coordinates')

	# set limits to be fixed on city points
	plt.xlim(plt.xlim())
	plt.ylim(plt.ylim())

	# use append to create circular plot
	# plot neuron positions and connections
	plt.plot(np.append(neuron_points[:, 0], neuron_points[0, 0]),
			 np.append(neuron_points[:, 1], neuron_points[0, 1]),
			 'bo-', label='Neuron Scaffold')
	
	plt.legend(loc='upper right')
	plt.title('Self Organizing Map #%d\nLength of current solution: %.2f' % (num, current_distance))
	plt.savefig('intermediate_plots/trial_%d_#%d.png' % (len(listdir('solutions')), num))
	# plt.show()
	plt.clf()
	plt.close()


def plot_solution_tsp(solution_points, solution_distance, tsp_name):
	plt.figure(figsize=(20, 10))
	# use append to create circular plot
	# plot tsp solution
	plt.plot(np.append(solution_points[:, 0], solution_points[0, 0]),
			 np.append(solution_points[:, 1], solution_points[0, 1]),
			 'bo-', label='Solution Traversal')
	
	plt.legend(loc='upper right')
	plt.title('Self Organizing Map Solution for %s\nLength of solution: %.2f' % (tsp_name, solution_distance))
	plt.savefig('solutions/trial_%d_solution.png' % len(listdir('solutions')))
	# plt.show()
	plt.clf()
	plt.close()
	
	print('Solution saved as trial_%d_solution.png' % (len(listdir('solutions'))-1))


def plot_decay_reductions(iteration_limit, decay_interval, linear_constant, radius, tsp_name):

	# plot learning rate decay per tenth iteration
	linear_lr_decay = np.array([linear_constant*(iteration_limit-i) for i in range(0, iteration_limit, decay_interval)])
	exp_lr_values = np.array([np.power(np.e, -0.001*i) for i in range(0, iteration_limit, decay_interval)])
	plt.plot(linear_lr_decay, 'r-', label='Linear')
	plt.plot([exp_lr_values[i-1]-exp_lr_values[i] for i in range(1, 250)], 'g-', label='Exp.')
	plt.legend(loc='upper right')
	plt.title('Learning Rate Decay Per Tenth Iteration for ' + tsp_name)
	plt.savefig('decay_plots/' + tsp_name + '_lr_decay.png')
	plt.clf()
	
	# plot radius decay per tenth iteration
	linear_radius_decay = radius * linear_lr_decay
	exp_radius_values = radius * exp_lr_values
	plt.plot(linear_radius_decay, 'r-', label='Linear')
	plt.plot([exp_radius_values[i-1]-exp_radius_values[i] for i in range(1, int(iteration_limit/decay_interval))], 'g-', label='Exp.')
	plt.legend(loc='upper right')
	plt.title('Radius Decay Per Tenth Iteration for ' + tsp_name)
	plt.savefig('decay_plots/' + tsp_name + '_radius_decay.png')
	plt.clf()
	plt.close()
