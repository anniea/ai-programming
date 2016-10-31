from helpers import read_tsp_data, plot_intermediate_tsp
import numpy as np
import math

# global variables
tsp = 'wi29'
learning_rate = 1
iteration_limit = 1000
no_of_neurons = None
plotting = True


def main():
	global no_of_neurons

	# read data from TSP data set
	cities_scaled, cities = read_tsp_data(tsp)

	# set parameters based on the data
	no_of_neurons = len(cities) * 2
	neighborhood_radius = int(no_of_neurons/10)

	# create an initial random self-organizing-map (SOM)
	initial_som = np.random.rand(no_of_neurons, 2)
	# use the SOM to create a scaffold
	scaffold = create_scaffold(cities_scaled, initial_som, neighborhood_radius)

	# read solution from scaffold
	solution = read_solution(cities, scaffold)
	#TODO: plot solution

	total_distance = get_total_distance(cities, solution)
	print('Total distance:', total_distance)
	print('Initial distance:', get_total_distance(cities, [x for x in range(len(cities))]))


def create_scaffold(cities, som, radius):

	if plotting:
		plot_intermediate_tsp(som, cities, 0)

	for i in range(iteration_limit):
		city = cities[np.random.randint(0, len(cities))]

		# find the best matching unit (BMU)
		bmu = get_bmu(som, city)

		# update the bmu
		som[bmu] += learning_rate * (city - som[bmu])

		# update neighbors of the bmu according to the neighborhood radius
		for n in range(1, radius+1):
			next_neighbor = bmu + n
			prev_neighbor = bmu - n

			if next_neighbor >= no_of_neurons:
				next_neighbor %= no_of_neurons
			elif prev_neighbor < 0:
				prev_neighbor %= no_of_neurons

			som[next_neighbor] += (1 - n/(radius+1)) * learning_rate * (city - som[next_neighbor])
			som[prev_neighbor] += (1 - n/(radius+1)) * learning_rate * (city - som[prev_neighbor])

		if plotting and not (i+1) % 100:
			plot_intermediate_tsp(som, cities, i+1)

		# decrease learning_rate linearly
		# decrease neighborhood linearly

	return som


def read_solution(cities, scaffold):
	temp = [[] for _ in range(no_of_neurons)]
	for i in range(len(cities)):
		temp[get_bmu(scaffold, cities[i])].append(i)

	solution = []
	for t in temp:
		for city in t:
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
