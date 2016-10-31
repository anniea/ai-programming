from data_reader import read_tsp_data
import numpy as np


def main():
	data = read_tsp_data('wi29')
	number_of_neurons = len(data)
	neighborhood_radius = int(number_of_neurons/10)
	learning_rate = 1

	som = np.random.rand(number_of_neurons, 2)

	print_som(som)


def traveling_salesman():
	pass


def print_som(som):
	for i in range(len(som)):
		print(i+1, som[i, :])

main()
