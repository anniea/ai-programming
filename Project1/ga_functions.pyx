import numpy.random as npr

# this file requires numpy, cython and a c-compiler
# it is compiled to .pyd and .c using setup.py (more instructions included in there)


# choose the best parents from the population based on their fitness
def choose_parents(list population, int pop_size, int no_of_parents, int length, list solutions, printing):
	
	cdef list parents = []
	
	# evaluate fitness of whole population
	cdef int fitness, i
	for i in range(pop_size):
		
		fitness = count_threats(population[i], length)
		
		population[i] = (population[i], fitness)
		
		# if fitness = 0, individual has no threats and is a solution
		if fitness == 0:
			# do not include solutions we have found already
			if population[i] not in solutions:
				solutions.append(population[i])
				print(population[i][0])
	
	# sort population based on fitness, with the best at the foremost indices
	population = sorted(population, key=lambda individual: individual[1])
	
	if printing:
		print('Best of current generation:', population[0][0])
	
	# select a predetermined number of the first (and best) individuals as parents
	cdef int j
	for j in range(no_of_parents):
		parents.append(population[j][0])
	
	return parents


# create children using PMX as crossover and random chance mutation
def create_child(list parents, int co_length, int no_of_parents, int mutation_chance, int length):
	
	cdef int father_index, mother_index, co_start, co_stop, replace_row, replace_index, mut_point_1, mut_point_2, \
		new_row, best_fitness, current_fitness, i
	cdef list father, mother, father_genes, mother_genes, child, best_child

	# initialize best fitness to be guaranteed beatable
	best_fitness = length

	# choose two random, but different parents
	father_index, mother_index = two_distinct_random(0, no_of_parents)
	father = parents[father_index]
	mother = parents[mother_index]

	# for all possible swaths
	for i in range(co_length+1):
		
		# set crossover swath start and ending point
		co_start = i
		co_stop = co_start + co_length
		
		# find the genes in the two swaths
		father_genes = father[co_start:co_stop]
		mother_genes = mother[co_start:co_stop]
		
		# perform PMX-crossover
		child = mother[:co_start] + father_genes + mother[co_stop:]
		# for every value in mother swath
		for new_row in mother_genes:
			# if value is lost value during copying of father swath
			if new_row not in father_genes:
				
				# find duplicate row to replace
				replace_row = father[mother.index(new_row)]
				replace_index = mother.index(replace_row)
				while co_start <= replace_index < co_stop:
					replace_row = father[replace_index]
					replace_index = mother.index(replace_row)
				
				# replace duplicate with lost value
				child[replace_index] = new_row
				
		# if mutation activates, swap the row placement of two random columns
		if mutation_chance > npr.randint(0, 100):
			mut_point_1, mut_point_2 = two_distinct_random(0, length)
			child[mut_point_1], child[mut_point_2] = child[mut_point_2], child[mut_point_1]
		
		# compute fitness of child, and remember it if it is the best so far
		current_fitness = count_threats(child, length)
		if current_fitness < best_fitness:
			best_child = child
			best_fitness = current_fitness
	
	# return only the best child
	return best_child
	
# count the amount of queens that are threatened by another, i.e. how many queens are placed illegally
cdef int count_threats(list positions, int length):
	
	# for each diagonal, we use mapping to check if there are any queen conflicts
	# if two positions are mapped to the same integer, the queens at these positions are threatening each other
	# if two or more queens threaten each other, the first queen is said to be legal, while the rest are not
	# mapping is different for rising and falling diagonals, and thus we need one set for each

	cdef int threat_count, i
	cdef set falling_diag_uniques, rising_diag_uniques
	
	# create sets to contain unique integers for queens along the rising and falling diagonals
	falling_diag_uniques = set()
	rising_diag_uniques = set()
	
	for i in range(length):
		# map queen positions to integers and add the unique integers to their respective set
		falling_diag_uniques.add(length - positions[i] - i)
		rising_diag_uniques.add(length - positions[i] + i)
	
	# compute the size difference from a solution set for both sets (solution set has no threats)
	# corresponds to the number of illegally placed queens, or amount of threats
	return 2*length - len(falling_diag_uniques) - len(rising_diag_uniques)


# find two distinct random numbers withing given interval
def two_distinct_random(minimum, maximum):
	random_1 = random_2 = 0
	while random_1 == random_2:
		random_1 = npr.randint(minimum, maximum)
		random_2 = npr.randint(minimum, maximum)
	
	return random_1, random_2


		################################################################################################
		# FUNCTION BELOW ARE NOT USED IN CURRENT SOLUTION, BUT ARE INCLUDED TO SHOW ALTERNATE ATTEMPTS #
		################################################################################################


# choose the parents from the population stochastically based on their fitness
def choose_parents_roulette(list population, int pop_size, int no_of_parents, int length, list solutions):
	
	# define maximum amount of fitness
	# multiply with a constant to create a wider gap between a bad and a good solution
	cdef int max_fitness = length*3
	
	cdef list roulette_wheel = []
	cdef list parents = []
	
	# evaluate fitness of whole population
	cdef int fitness, i
	for i in range(pop_size):
		
		# to give better individuals a higher fitness instead of lower, use difference between length and amount of threats
		# use constant to widen gap
		fitness = (length - count_threats(population[i], length))*3
		
		# add the individual to the roulette wheel as many times as it fitness score
		# higher score (for better solutions) gives an increased chance of of being chosen
		roulette_wheel.extend([population[i]]*fitness)

		# if fitness is maximum, individual has no threats and is a solution
		if fitness == max_fitness:
			# do not include solutions we have found already
			if population[i] not in solutions:
				solutions.append(population[i])
				print(population[i][0])
		
	# repeatedly choose a random individual to be a parent until we have enough parents
	cdef int roulette_max = len(roulette_wheel)
	cdef int j
	for j in range(no_of_parents):
		parents.append(roulette_wheel[npr.randint(0, roulette_max)])
	
	return parents


# creates a single pmx child with a randomly chosen swath
def create_child_pmx(list parents, int co_length, int no_of_parents, int mutation_chance, int length):
			
	cdef int father_index, mother_index, co_start, co_stop, replace_row, replace_index, mut_point_1, mut_point_2, \
		new_row
	cdef list father, mother, father_genes, mother_genes, child
	
	# choose two random, but different parents
	father_index, mother_index = two_distinct_random(0, no_of_parents)
	father = parents[father_index]
	mother = parents[mother_index]
	
	# set crossover swath start (chosen randomly) and ending point
	co_start = npr.randint(0, length - co_length + 1)
	co_stop = co_start + co_length
		
	# find the genes in the two swaths
	father_genes = father[co_start:co_stop]
	mother_genes = mother[co_start:co_stop]
	
	# perform PMX-crossover
	child = mother[:co_start] + father_genes + mother[co_stop:]
	# for every value in mother swath
	for new_row in mother_genes:
		# if value is lost value during copying of father swath
		if new_row not in father_genes:
			
			# find duplicate row to replace
			replace_row = father[mother.index(new_row)]
			replace_index = mother.index(replace_row)
			while co_start <= replace_index < co_stop:
				replace_row = father[replace_index]
				replace_index = mother.index(replace_row)
			
			# replace duplicate with lost value
			child[replace_index] = new_row
	
	# if mutation activates, swap the row placement of two random columns
	if mutation_chance > npr.randint(0, 100):
		mut_point_1, mut_point_2 = two_distinct_random(0, length)
		child[mut_point_1], child[mut_point_2] = child[mut_point_2], child[mut_point_1]
		
	return child


# create a child where the values that are similar in the mother and father are kept, and the rest are randomly chosen
def create_child_keep_similar(list parents, int co_length, int no_of_parents, int mutation_chance, int length, dict threat_dict):
	
	cdef int father_index, mother_index, mut_point_1, mut_point_2, new_row, i
	cdef list father, mother, child

	# choose two random, but different parents
	father_index, mother_index = two_distinct_random(0, no_of_parents)
	father = parents[father_index]
	mother = parents[mother_index]
	
	# copy all similar values to child
	child = [-1]*length
	for i in range(length):
		if father[i] == mother[i]:
			child[i] = father[i]
			
	# while child is still missing values, try a random value and add if not duplicate
	while -1 in child:
		new_row = npr.randint(1, length+1)
		if new_row not in child:
			child[child.index(-1)] = new_row
	
	# if mutation activates, swap the row placement of two random columns
	if mutation_chance > npr.randint(0, 100):
		mut_point_1, mut_point_2 = two_distinct_random(0, length)
		child[mut_point_1], child[mut_point_2] = child[mut_point_2], child[mut_point_1]
	
	return child
