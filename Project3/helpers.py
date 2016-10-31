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
	
	# scale coordinates to real numbers between 0 and 1
	data_points /= np.amax(data_points)
	
	# close file
	file.close()

	# print('All data points:')
	# print(data_points)
	# print("All X's:")
	# print(data_points[:, 0])
	# print("All Y's:")
	# print(data_points[:, 1])
		
	return data_points/np.amax(data_points), data_points


def plot_tsp_data(neuron_points, city_points, num):
	plt.figure(figsize=(20, 10))
	# use append to create circular plot
	# plot tsp cities positions
	plt.plot(np.append(city_points[:, 0], city_points[0, 0]),
			 np.append(city_points[:, 1], city_points[0, 1]),
			 'ro', label='City Coordinates')
	
	# plot neuron positions
	plt.plot(np.append(neuron_points[:, 0], neuron_points[0, 0]),
			 np.append(neuron_points[:, 1], neuron_points[0, 1]),
			 'bo-', label='Neuron Coordinates')
	
	plt.legend(loc='upper right')
	plt.title('Self Organizing Map #' + str(num))
	plt.savefig('intermediate_plots/trial_%d_#%d.png' % (len(listdir('solutions')), num))
	# plt.show()
	plt.clf()
