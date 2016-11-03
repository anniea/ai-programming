from helpers import read_tsp_data, plot_intermediate_tsp, plot_solution_tsp
import numpy as np
import math

# global variables
# choose tsp-problem by name (exists in data-folder)
# tsp = 'wi29'
# tsp = 'dj38'
# tsp = 'qa194'
tsp = 'uy734'

# alterable parameters
initial_learning_rate = 1
initial_radius = None
no_of_neurons = None
base_iteration_limit = 2500
iteration_multiple = 1
iteration_limit = base_iteration_limit * iteration_multiple
linear_constant = 7.25/(1000 * iteration_multiple * iteration_limit)
decay_interval = 10
plotting = True

# choose decay types for learning rate (lr) and radius
# decay_type = 'static'
# decay_type = 'linear'
decay_type = 'exponential'
	
	
def main():
	# find_tsp_solution(tsp)
	auto_runner()
	

def find_tsp_solution(tsp_name):
	global no_of_neurons, initial_radius

	# read data from TSP data set
	cities_scaled, cities = read_tsp_data(tsp_name)
	
	# set global parameters based on the data
	no_of_neurons = len(cities) * 2
	initial_radius = math.ceil(no_of_neurons/10)
	
	# create an initial random self-organizing-map (SOM)
	initial_som = np.random.rand(no_of_neurons, 2)
	
	# use the SOM to create a scaffold
	scaffold = create_scaffold(initial_som, cities_scaled, initial_learning_rate, initial_radius)

	# read solution from scaffold
	solution = read_solution(scaffold, cities_scaled)
	
	# print decay types
	print('\nDecay type:', decay_type)

	# print traversal distance of initial and final solution
	solution_distance = get_total_distance(cities, solution)
	print('Initial distance:', get_total_distance(cities, [x for x in range(len(cities))]))
	print('Solution distance:', solution_distance)

	# plot solution traversal with length
	plot_solution_tsp(np.array([cities[i] for i in solution]), solution_distance, tsp_name)
	
	return solution_distance


def create_scaffold(som, cities_scaled, learning_rate, radius):

	if plotting:
		# plot initial (random) scaffold
		plot_intermediate_tsp(som, cities_scaled, 0)

	for i in range(iteration_limit):
		# select random city
		city = cities_scaled[np.random.randint(0, len(cities_scaled))]

		# find the best matching unit (BMU)
		bmu = get_bmu(som, city)

		# update the bmu
		som[bmu] += learning_rate * (city - som[bmu])

		# update neighbors of the bmu according to the neighborhood radius
		for n in range(1, math.ceil(radius+1)):
			next_neighbor = bmu + n
			prev_neighbor = bmu - n

			# ensure index is wrapped around list if out of bounds
			if next_neighbor >= no_of_neurons:
				next_neighbor %= no_of_neurons
			elif prev_neighbor < 0:
				prev_neighbor %= no_of_neurons

			spatial_decay = 1 - n/math.ceil(radius+1)
			som[next_neighbor] += spatial_decay * learning_rate * (city - som[next_neighbor])
			som[prev_neighbor] += spatial_decay * learning_rate * (city - som[prev_neighbor])

		if plotting and not (i+1) % int(iteration_limit/10):
			plot_intermediate_tsp(som, cities_scaled, i+1)
		
		# decay learning rate and radius at certain intervals
		if not (i+1) % decay_interval:
			
			# static formula is x = x - k, where k is a constant
			# linear formula is x = x - (k * (t_max - t)), where k is a constant and  t is iteration number
			# exp. formula is x = init_x * (e^(-kt)), where k is constant and t is iteration number
			
			# perform learning rate decay
			if decay_type == 'static':
				learning_rate -= decay_interval/iteration_limit
				radius -= initial_radius * decay_interval/iteration_limit
			elif decay_type == 'linear':
				learning_rate -= linear_constant * (iteration_limit-i-1)
				radius -= initial_radius * linear_constant * (iteration_limit-i-1)
			elif decay_type == 'exponential':
				learning_rate = np.power(np.e, -0.001*(i+1))
				radius = initial_radius * np.power(np.e, -0.001*(i+1))
			else:
				print('\nNo match for decay type - no decay performed')
			
	return som


def read_solution(scaffold, cities_scaled):
	scaffold_traversal = [[] for _ in range(no_of_neurons)]
	for i in range(len(cities_scaled)):
		# add city index to match list of nearest neuron
		scaffold_traversal[get_bmu(scaffold, cities_scaled[i])].append(i)

	# create solution by adding cities in consecutive order of appearance in scaffold_traversal
	solution = []
	for neuron_matches in scaffold_traversal:
		for city in neuron_matches:
			solution.append(city)
	return solution


# find the best matching unit (BMU) which is the closest neuron to a city
def get_bmu(som, city):
	return np.argmin([get_distance(neuron, city) for neuron in som])


# find the euclidean distance between two points in a 2D space
def get_distance(point1, point2):
	return math.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))


# calculate the total distance of a solution
def get_total_distance(cities, solution):
	distance = get_distance(cities[solution[0]], cities[solution[-1]])
	for i in range(1, len(cities)):
		distance += get_distance(cities[solution[i-1]], cities[solution[i]])
	return distance


def auto_runner():
	global decay_type
	
	decay_types = ['static', 'linear', 'exponential']
	tsps = ['wi29', 'dj38', 'qa194', 'uy734']
	for decay in decay_types:
		decay_type = decay
		for tsp in tsps:
			distance = 0
			for _ in range(10):
				distance += find_tsp_solution(tsp)
			
			print('\n\nAverage distance of ' + tsp + ' using ' + decay + ' decay is ' + str(distance/10))

main()
