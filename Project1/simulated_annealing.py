# coding=utf-8
import random
import math
from time import time

# global variables
start_time = 0
printing = False
length = 0
solutions = []
duplicate_count = 0
trials = 0
solution_interval = 0


def main():
	global length, printing

	length = int(input('Choose a length for the n-queens problem: '))
	input_positions = input('Enter positions separated by spaces, or press enter to generate a "1 2 3 ..." input: ')
	printing = input('Print step by step? [y/n]: ')

	if printing == 'y':
		printing = True
	else:
		printing = False

	# mark start time
	start_time = time()

	if input_positions == '':
		positions = list(range(1, length+1))
	else:
		positions = [int(pos) for pos in input_positions.split(' ')]

	# check if input is valid, and terminate program if not
	if not valid_input(positions):
		print('A solution is impossible because the specified input is illegal.')
		return

	print('Given input: {} for n = {}.'.format(' '.join(str(i) for i in positions), length))

	# process input to meet requirements of implementation
	positions = process_input(positions)

	temperature = 10
	dt = 0.01

	# will run the algorithm until there are relatively few
	# new solutions found or until the time limit is reached
	while solution_interval < math.floor(200/length) and (time() - start_time) * 1000 < 300000:
		simulated_annealing(positions, temperature, dt)

	# print solutions and other data
	print('\n\nFinished algorithm.')
	if len(solutions) == 0:
		print('- Found no solutions for the given input.')
	else:
		print('- Found {} solution(s) for the given input for {} trial(s) and {} duplicate(s).'.format(len(solutions), trials, duplicate_count))

	print('\n\nTime: {:.2f} ms'.format(((time() - start_time) * 1000)))


def simulated_annealing(initial_positions, tmax, dt):
	global solutions, duplicate_count, solution_interval

	# initialize the temperature
	temperature = tmax

	# set initial threat count
	solution_interval = 0
	positions = initial_positions[:]
	current_threat_count = count_threats(positions)

	# run simulated annealing until temperature reaches zero
	while temperature > 0:

		if printing:
			print('Current threat count is {} at temperature {:.2f}'.format(current_threat_count, temperature))

		# generate neighborhood and find the best neighbor
		neighbors, best_neighbor, best_threat_count = generate_neighbors(positions)

		# calculate variables for acceptance function
		q = length - (best_threat_count - current_threat_count)
		p = min(1.0, math.exp(-((q/length)**2)/temperature))
		r = random.uniform(0, 1)

		# neighbor is better than current / exploitation,
		# choose best neighbor
		if q >= length or p < r:
			if printing:
				print('The algorithm chose the best neighbor: {}'.format(' '.join(str(i) for i in best_neighbor)))
			positions = best_neighbor
			current_threat_count = best_threat_count
		# exploration, choose a random neighbor
		else:
			rand = random.randint(0, len(neighbors)-1)
			positions = neighbors[rand]
			current_threat_count = count_threats(positions)
			if printing:
				print('The algorithm chose a random neighbor: {}'.format(' '.join(str(i) for i in positions)))

		# check if current is a solution, and add if unique
		if current_threat_count == 0:
			if positions not in solutions:
				solutions.append(positions)
				if printing:
					print('\nFound a solution: {} which gives a total of {} solutions\n\n'.format(' '.join(str(i) for i in positions), len(solutions)))
				else:
					print('Found a new solution: {}'.format(' '.join(str(i) for i in positions)))
				return
			else:
				duplicate_count += 1
				solution_interval += 1
				if printing:
					print('\nFound a duplicate solution.\n\n')

		# decrease temperature after each iteration
		temperature -= dt

	if printing:
		print('\nFound no solution before the temperature reached zero.\n\n')


def generate_neighbors(positions):
	global trials

	neighbors = [None for _ in range(length)]
	best_neighbor = None
	best_threat_count = float('inf')

	for i in range(length):

		trials += 1
		# swap two random neighbors
		rand = random.sample(range(length), 2)
		neighbor_positions = positions[:]
		neighbor_positions[rand[0]], neighbor_positions[rand[1]] = neighbor_positions[rand[1]], neighbor_positions[rand[0]]
		neighbors[i] = neighbor_positions

		neighbor_threat_count = count_threats(neighbor_positions)

		# find the best neighbor
		if neighbor_threat_count < best_threat_count:
			best_threat_count = neighbor_threat_count
			best_neighbor = neighbor_positions

	return neighbors, best_neighbor, best_threat_count


# count the amount of queens that are threatened by another, i.e. how many queens are placed illegally
def count_threats(positions):

	# for each diagonal, we use mapping to check if there are any queen conflicts
	# if two positions are mapped to the same integer, the queens at these positions are threatening each other
	# if two or more queens threaten each other, the first queen is said to be legal, while the rest are not
	# mapping is different for rising and falling diagonals, and thus we need one set for each

	# create sets to contain unique integers for queens along the rising and falling diagonals
	falling_diag_uniques = set()
	rising_diag_uniques = set()

	for i in range(length):
		# map queen positions to integers and add the unique integers to their respective set
		falling_diag_uniques.add(length - positions[i] - i)
		rising_diag_uniques.add(length - positions[i] + i)

	# compute the size difference from a solution set for both sets (solution set has no threats)
	# corresponds to the number of illegally placed queens, or amount of threats
	return 2 * length - len(falling_diag_uniques) - len(rising_diag_uniques)


# used to check if positions in input are valid
def valid_input(positions):
	# check if input length matches input positions given
	if len(positions) != length:
		return False

	# check if all queen placements are within the board
	for pos in positions:
		if pos < 1 or pos > length:
			return False

	return True


# replaces duplicate rows in positions with available rows
def process_input(positions):
	row_count = [0] * length
	duplicate_positions = []
	changed = False

	# count all occurrences of each row, and collect duplicate positions
	for i in range(length):
		if row_count[positions[i] - 1] > 0:
			duplicate_positions.append(i)
			changed = True
		row_count[positions[i] - 1] += 1

	# for all duplicate positions, find replace with available row
	for pos in duplicate_positions:
		available_row = row_count.index(0)
		positions[pos] = available_row + 1
		row_count[available_row] = 1

	# if positions were changed, let me know
	if changed:
		print('Positions changed to: {}\n'.format(' '.join(str(i) for i in positions)))

	return positions


main()
