from time import time

# global variables
start_time = 0
printing = False
length = 8
trials = 0


def main():
	global printing, start_time

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
		positions = [0] * length
	else:
		positions = [int(i) for i in input_positions.split(' ')]
		positions.extend([0] * (length - len(positions)))

	# initialize the board with the given queens of the input
	board, amount_placed = init_board(positions)

	print('Given input: {} for n = {}.'.format(' '.join(str(i) for i in positions), length))

	if board is None:
		print('A solution is impossible because the specified input is illegal.')
		return

	print('\nFor boards, . marks a threatened position and + marks a safe position.')

	print('\nThe initial board for the given input:\n')
	print_board(board)

	solution = find_placements(board, positions, amount_placed + 1)

	# print solutions and other data
	print('\nFinished recursion. The algorithm tried a total of {} boards.'.format(trials))
	if len(solution) == 0:
		print('\n\nThere is no solution for the given input.')
	else:
		print('\n\nA solution for the given input is:\n{}\n'.format(' '.join(str(i) for i in solution)))
		final_board, _ = init_board(solution)
		print_board(final_board)

	print('\n\nTime: {:.2f} ms'.format(((time() - start_time) * 1000)))


def find_placements(board, positions, queen_number):
	global trials

	# end of board reached
	if queen_number > length:
		return positions

	if printing:
		print('\nLooking for solutions for queen in column {}...'.format(convert_column(queen_number)))

	# heuristic: try queens which makes the least amount of new threats
	new_queens = []
	for i in range(length):
		if board[i][queen_number - 1] == 1:
			new_queens.append((i, queen_number - 1, count_new_threats(board, i, queen_number - 1)))

	new_queens = sorted(new_queens, key=lambda queen: queen[2])

	if printing:
		print('\nTry placement positions in the following order:')
		for new_queen in new_queens:

			print('({}, {}) - which threatens {} new position(s).'.format(new_queen[0], new_queen[1], new_queen[2]))

	# try the next queen
	for new_queen in new_queens:

		new_board = [row[:] for row in board]
		new_positions = positions[:]
		insert_queen(new_board, new_queen[0], new_queen[1], new_positions)

		if printing:
			print('\n\nTrying board:\n')
			print_board(new_board)
		
		trials += 1

		# recursion
		result_positions = find_placements(new_board, new_positions, queen_number + 1)
		if len(result_positions) > 0:
			return result_positions

	# no available spots, backtracking step
	if printing:
		print('\n\nNo possible solutions for queen {}, backtracking...'.format(convert_column(queen_number)))
	return []


def init_board(positions):
	board = [[1 for _ in range(length)] for _ in range(length)]

	# convert positions to coordinates in the matrix
	queen_coordinates = create_matrix_coordinates(positions)

	# insert each queen into the board
	for coordinate in queen_coordinates:
		success = insert_queen(board, coordinate[0], coordinate[1], positions)

		# return if queen was not inserted correctly
		if not success:
			return None, 0

	return board, len(queen_coordinates)


def mark_threats(board, x_pos, y_pos):

	for i in range(1, length + 1):

		# mark horizontal
		if i - 1 != y_pos:
			board[x_pos][i - 1] = 0

		# mark vertical
		if i - 1 != x_pos:
			board[i - 1][y_pos] = 0

		# mark diagonals
		if (x_pos + i) <= length - 1 and (y_pos + i) <= length - 1:
			board[x_pos + i][y_pos + i] = 0
		if (x_pos + i) <= length - 1 and (y_pos - i) >= 0:
			board[x_pos + i][y_pos - i] = 0
		if (x_pos - i) >= 0 and (y_pos + i) <= length - 1:
			board[x_pos - i][y_pos + i] = 0
		if (x_pos - i) >= 0 and (y_pos - i) >= 0:
			board[x_pos - i][y_pos - i] = 0


def count_new_threats(board, x_pos, y_pos):
	
	new_threats = 0

	for i in range(1, length + 1):

		# mark horizontal
		if i - 1 != y_pos:
			new_threats += board[x_pos][i - 1]

		# mark vertical
		if i - 1 != x_pos:
			new_threats += board[i - 1][y_pos]

		# mark diagonals
		if (x_pos + i) <= length - 1 and (y_pos + i) <= length - 1:
			new_threats += board[x_pos + i][y_pos + i]
		if (x_pos + i) <= length - 1 and (y_pos - i) >= 0:
			new_threats += board[x_pos + i][y_pos - i]
		if (x_pos - i) >= 0 and (y_pos + i) <= length - 1:
			new_threats += board[x_pos - i][y_pos + i]
		if (x_pos - i) >= 0 and (y_pos - i) >= 0:
			new_threats += board[x_pos - i][y_pos - i]

	return new_threats


def create_matrix_coordinates(positions):
	coordinates = []

	for i in range(length):
		if positions[i] != 0:
			row = length - positions[i]
			col = i
			
			coordinates.append((row, col))

	return coordinates


def insert_queen(board, x_pos, y_pos, positions):
	
	if board[x_pos][y_pos] == 0:
		return False

	# insert queen into board and update positions
	board[x_pos][y_pos] = 'Q'
	positions[y_pos] = length - x_pos
	# mark all threats from the inserted queen
	mark_threats(board, x_pos, y_pos)

	return True


def print_board(board):
	column_names = '  abcdefgh'
	print('  '.join(column_names))
	print('   ----------------------------')
	for i in range(length):
		print(length - i, end='  |  ')
		for j in range(length):
			if (board[i][j]) == 1:
				print('+', end='  ')
			elif board[i][j] == 0:
				print('·', end='  ')
			else:
				print(board[i][j], end='  ')
		print('|')
	print('   ----------------------------')


def convert_column(col):
	column_names = 'abcdefgh'
	return column_names[col - 1]

main()
