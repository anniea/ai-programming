from math import ceil
from time import time

# global variables
start_time = 0
printing = False
board_length = 0
solutions = []


def main():
	global board_length, printing, start_time, solutions

	board_length = int(input('Choose a length for the n-queens problem: '))
	input_positions = input('Enter positions for a chosen first number of queens separated by spaces, '
							'or press enter to start with empty positions: ')
	printing = input('Print step by step? [y/n]: ')

	if printing == 'y':
		printing = True
	else:
		printing = False

	# mark start time
	start_time = time()

	if input_positions == '':
		positions = []
	else:
		positions = [int(i) for i in input_positions.split(' ')]

	amount_placed = len(positions)

	print('Given input: {} for n = {}.'.format(' '.join(str(i) for i in positions), board_length))

	if not is_legal(positions):
		print('A solution is impossible because the specified input is illegal.')
		return

	if printing:
		print('\nLooking for solutions starting queen number {}...'.format(amount_placed + 1))

	# find the initial possible rows to place the remaining queens
	free_rows = []
	for row in range(1, board_length+1):
		if row not in positions:
			free_rows.append(row)

	# run backtracking algorithm
	find_placements(positions, amount_placed + 1, free_rows)

	# print solutions and other data
	print('\n\nFinished recursion.')
	if len(solutions) == 0:
		print('- There are no solutions for the given input.')
	else:
		print('- Found {} solutions for the given input.'.format(len(solutions)))
		if printing:
			print('These are, in increasing order:')
			solutions.sort()
			for sol in solutions:
				print(' '.join(str(i) for i in sol))

	print('\n\nTime: {:.2f} ms'.format(((time() - start_time) * 1000)))


def find_placements(positions, queen_number, free_rows):
	global solutions
	
	# end of board reached
	if queen_number > board_length:
		solutions.append(positions)
		if printing:
			print('\n\nFound a solution!\n {} is added to the list of solutions.'.format(' '.join(str(i) for i in positions)))
		return

	# try new legal placements
	for row in find_legal_rows(positions, free_rows):

		# copy lists
		new_positions = positions[:] + [row]
		new_free_rows = free_rows[:]
		new_free_rows.remove(row)

		if printing:
			print('\n\nTrying {}\n'.format(' '.join(str(i) for i in positions)))

		# recursion
		find_placements(new_positions, queen_number + 1, new_free_rows)

	# no available spots, backtracking step
	if printing:
		print('\nNo possible solutions for queen {}, backtracking...\n'.format(queen_number))
	return


def is_legal(positions):

	# check that there are no threats on rows
	pos_length = len(positions)
	if pos_length != len(set(positions)):
		return False

	# check that there are no threats on diagonals
	falling_diag_uniques = set()
	rising_diag_uniques = set()
	
	for i in range(pos_length):
		falling_diag_uniques.add(board_length - positions[i] - i)
		rising_diag_uniques.add(board_length - positions[i] + i)
		
	if pos_length != len(falling_diag_uniques) or pos_length != len(rising_diag_uniques):
		return False

	return True


def find_legal_rows(positions, free_rows):
	
	legal_rows = free_rows[:]

	pos_length = len(positions)
	for i in range(pos_length):

		# remove all illegal placements
		falling = positions[i] - pos_length + i
		if falling in legal_rows:
			legal_rows.remove(falling)

		rising = positions[i] + pos_length - i
		if rising in legal_rows:
			legal_rows.remove(rising)
		
	return legal_rows


main()
