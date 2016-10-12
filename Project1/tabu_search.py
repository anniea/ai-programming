from time import time
from random import randint
from math import floor

# global variables
start_time = 0
printing = False
length = 0
solutions = []
trials = 0
duplicate_count = 0
solution_interval = 0

# alter to change how long program runs
time_limit = 300000


def main():
	global solutions, length, printing, start_time
	
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

	# set tabu tenure to be half of all possible swaps
	tabu_tenure = int(((length * (length + 1) / 2) - 1) / 2)
	print('\nTabu tenure calculated to be %d' % tabu_tenure)
	
	# process input to meet requirements of implementation
	positions = process_input(positions)
	
	# find legal queen placements
	find_placements(positions, tabu_tenure)
	
	# print solutions and other data
	print('\n\n- The algorithm tried a total of {} boards.'.format(trials))
	if len(solutions) == 0:
		print('- There are no solutions for the given input.')
	else:
		print('\n- During execution, %d duplicate solutions were found' % duplicate_count)
		print('\n- Found {} solutions for the given input.'.format(len(solutions)))
		if printing:
			print('These are:')
			solutions.sort()
			for sol in solutions:
				print(' '.join(str(i) for i in sol))
	
	print('\n\nTime: {:.2f} ms'.format(((time() - start_time) * 1000)))


# iteratively test different positions through swaps to find legal queen placements
def find_placements(initial_positions, tabu_tenure):
	global solutions, trials, duplicate_count, solution_interval
	
	# create a matrix to represent the tabu list
	# entry (i, j) how long a swap using i and j is tabu
	tabu_list = [[0 for _ in range(length)] for _ in range(length)]
	
	# initialize to start
	restart = True
	
	# swap_round = 0
	# will run the algorithm until there are relatively few
	# new solutions found or until the time limit is reached
	while solution_interval < floor(200/length) and (time() - start_time) * 1000 < time_limit:
		# swap_round += 1

		# for every restart, revert to initial positions
		if restart:
			positions = initial_positions[:]
			current_threat_count = count_threats(positions)
			restart = False
		
		# initialize best threat count to current
		best_threat_count = current_threat_count
		
		# for every possible swap
		for i in range(length - 1):
			for j in range(i + 1, length):
				
				# decrement tabu time if swap is tabu, and skip
				if tabu_list[i][j] > 0:
					tabu_list[i][j] -= 1
					continue
				
				# create new positions from swap and compute threat count
				new_positions = positions[:]
				new_positions[i], new_positions[j] = new_positions[j], new_positions[i]
				new_threat_count = count_threats(new_positions)
				
				# check if best swap so far, and remember if so
				if new_threat_count < best_threat_count:
					best_swap = (i, j)
					best_threat_count = new_threat_count
					
				trials += 1
		
		# if no better positions found, find random swap to perform
		if best_threat_count == current_threat_count:
			random_1, random_2 = two_distinct_random(0, length - 1)
			best_swap = (random_1, random_2)
			
		# perform best swap, and find the current threat count for the new positions
		positions[best_swap[0]], positions[best_swap[1]] = positions[best_swap[1]], positions[best_swap[0]]
		current_threat_count = count_threats(positions)
		
		# update tabu time to tabu tenure for current swap
		tabu_list[best_swap[0]][best_swap[1]] = tabu_tenure
		
		# if current positions make a solution, add if unique and then restart
		if current_threat_count == 0:
			if positions not in solutions:
				solutions.append(positions)
				if printing:
					print('\nFound a solution: {} which gives a total of {} solutions\n\n'.format(' '.join(str(i) for i in positions), len(solutions)))
				else:
					print('Found a new solution: {}'.format(' '.join(str(i) for i in positions)))
				solution_interval = 0
			else:
				duplicate_count += 1
				solution_interval += 1
				if printing:
					print('\nFound a duplicate solution.\n\n')
			restart = True
		
		if printing:
			print('Current positions: ', ' '.join(str(i) for i in positions))
		
		# print amount of solutions for every set interval of swaps performed
		# if swap_round % 1000 == 0:
		# 	print('%d solutions found in swapping round %d' % (len(solutions), swap_round))

	if printing:
		print('\nTabu list on exit:')
		for tabu in tabu_list:
			print(tabu)


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


# find two distinct random numbers withing given interval
def two_distinct_random(minimum, maximum):
	random_1 = random_2 = 0
	while random_1 == random_2:
		random_1 = randint(minimum, maximum)
		random_2 = randint(minimum, maximum)
	
	return random_1, random_2


main()
