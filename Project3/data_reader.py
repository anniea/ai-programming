
def read_tsp_data(filename):
	file = open('data/' + filename + '.txt', 'r')
	
	data_points = []
	
	line = file.readline()
	while 'NODE_COORD_SECTION' not in line:
		line = file.readline()
	
	line = file.readline()
	while 'EOF' not in line:
		coordinates = line.split(' ')
		print(coordinates)
		data_points.append((float(coordinates[1]), float(coordinates[2].strip())))
		line = file.readline()
		
	return data_points
