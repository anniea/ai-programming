from time import time
import numpy.random as npr

from ga_functions import choose_parents, create_child

# global variables
start_time = 0
printing = False
length = 0
solutions = []

# can be altered to change how program runs
time_limit = 300000  # alter to change how long program runs
pop_size = 15000  # choose population size
no_of_parents = int(pop_size / 5)  # number of parents set to be 20% of population
mutation_chance = 10  # choose chance for newly created child to mutate
random_swap_amount = 0  # updated after length is set in main()


def main():
	global solutions, length, printing, random_swap_amount, start_time
	
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
	
	# set the amount of random swaps to be performed to create new individuals for initial population
	random_swap_amount = int(length / 2)
	
	# process input to meet requirements of implementation
	positions = process_input(positions)
	
	# find legal queen placements
	find_placements(positions)
	
	# print solutions
	if len(solutions) == 0:
		print('- There are no solutions for the given input.')
	else:
		print('\n- Found {} solutions for the given input.'.format(len(solutions)))
		if printing:
			print('These are:')
			solutions.sort()
			for sol in solutions:
				print(' '.join(str(i) for i in sol))
	
	print('\n\nTime: {:.2f} ms'.format(((time() - start_time) * 1000)))


# iteratively choose parents and create children to find legal queen placements
def find_placements(positions):
	global solutions
	
	# generate initial population
	population = generate_population(positions)
	
	# length of crossover swath in PMX
	co_length = int(length / 2)
	
	# generation_no = 0
	while (time() - start_time) * 1000 < time_limit:
		# generation_no += 1
		
		# choose parents used for breeding
		parents = choose_parents(population, pop_size, no_of_parents, length, solutions, printing)
		
		# initialize new population with parents, and add children until intended population size is reached
		population = parents[:]
		while len(population) < pop_size:
			# create child with crossover (co) and mutation
			population.append(create_child(parents, co_length, no_of_parents, mutation_chance, length))
		
		# print amount of solutions for every set interval of generations passed
		# if generation_no % 100 == 0:
		# 	print('%d solutions found in generation %d' % (len(solutions), generation_no))


# generate population with initial positions as basis
def generate_population(initial_positions):
	population = []
	
	# create as many new positions as intended population size
	for _ in range(pop_size):
		positions = initial_positions[:]
		
		# perform a predetermined amount of random swaps to initial positions to diversify population
		for __ in range(random_swap_amount):
			rand_1, rand_2 = two_distinct_random(0, length)
			positions[rand_1], positions[rand_2] = positions[rand_2], positions[rand_1]
		
		population.append(positions)
	
	return population


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


# find two distinct random numbers withing given interval
def two_distinct_random(minimum, maximum):
	random_1 = random_2 = 0
	while random_1 == random_2:
		random_1 = npr.randint(minimum, maximum)
		random_2 = npr.randint(minimum, maximum)
	
	return random_1, random_2


main()
