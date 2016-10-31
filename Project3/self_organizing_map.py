from helpers import read_tsp_data, plot_intermediate_tsp, plot_solution_tsp
import numpy as np
import math  # TODO: maybe replace all math-functions with numpy-functions?

# global variables
tsp = 'wi29'
learning_rate = 1
iteration_limit = 2500
no_of_neurons = None
plotting = True


def main():
	global no_of_neurons

	# read data from TSP data set
	cities_scaled, cities = read_tsp_data(tsp)
	
	# set parameters based on the data
	no_of_neurons = len(cities) * 2
	# neighborhood_radius = int(no_of_neurons/10)
	neighborhood_radius = int(no_of_neurons/5)  # test of different value

	# create an initial random self-organizing-map (SOM)
	initial_som = np.random.rand(no_of_neurons, 2)
	# use the SOM to create a scaffold
	scaffold = create_scaffold(initial_som, cities_scaled, neighborhood_radius)

	# read solution from scaffold
	solution = read_solution(scaffold, cities_scaled)
	plot_solution_tsp(np.array([cities[i] for i in solution]))

	print('Solution:', solution)

	print('\nInitial distance:', get_total_distance(cities, [x for x in range(len(cities))]))
	print('Computed distance:', get_total_distance(cities, solution))


def create_scaffold(som, cities_scaled, radius):
	global learning_rate

	if plotting:
		plot_intermediate_tsp(som, cities_scaled, 0)

	for i in range(iteration_limit):
		city = cities_scaled[np.random.randint(0, len(cities_scaled))]

		# find the best matching unit (BMU)
		bmu = get_bmu(som, city)

		# update the bmu
		som[bmu] += learning_rate * (city - som[bmu])

		# update neighbors of the bmu according to the neighborhood radius
		for n in range(1, math.ceil(radius+1)):
			next_neighbor = bmu + n
			prev_neighbor = bmu - n

			if next_neighbor >= no_of_neurons:
				next_neighbor %= no_of_neurons
			elif prev_neighbor < 0:
				prev_neighbor %= no_of_neurons

			spatial_decay = 1 - n/math.ceil(radius+1)
			som[next_neighbor] += spatial_decay * learning_rate * (city - som[next_neighbor])
			som[prev_neighbor] += spatial_decay * learning_rate * (city - som[prev_neighbor])

		if plotting and not (i+1) % 100:
			plot_intermediate_tsp(som, cities_scaled, i+1)

		# decrease learning_rate linearly
		# decrease neighborhood linearly
		# test reduction
		if not (i+1) % 10:
			learning_rate *= 0.99
			radius *= 0.99
			# print('Radius as float:', radius, 'and as int', math.ceil(radius))

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
	return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


# calculate the total distance of a solution
def get_total_distance(cities, solution):
	distance = get_distance(cities[solution[0]], cities[solution[-1]])
	for i in range(1, len(cities)):
		distance += get_distance(cities[solution[i-1]], cities[solution[i]])
	return distance


def print_som(som):
	for i in range(len(som)):
		print(i+1, som[i, :])


main()
