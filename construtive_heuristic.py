"""



"""
import re
import os
import numpy as np

def read_instances(file_name):

	with open("Entrada/EUC_2D/" + file_name, 'r') as instances:

		instances = instances.read()

		instances = instances.split('\n')

		edge_type = instances[4].split(':')[1].strip()

		instances = instances[6: ]

		instances = list(map(lambda x: re.findall(r'\d+', x), instances))

		instances = list(filter(lambda x: len(x) > 2, instances))

		instances = np.array(list(map(lambda line: np.array([int(line[0]), float(line[1]), float(line[2])]), instances)))

	return instances, edge_type


def measure_distance(input_matrix, node_a, node_b, edge_type):
	"""

		Calcula a distância entre dois nós

	"""

	xd = input_matrix[node_a][1] - input_matrix[node_b][1]

	yd = input_matrix[node_a][2] - input_matrix[node_b][2]

	dij = (xd * xd + yd * yd)

	if edge_type == 'EUC_2D':

		return int(np.sqrt(dij))

	# a distância é ATT
	rij = np.sqrt(dij/10)
	tij = int(rij)

	if tij < rij:

		return tij + 1

	return tij 


def generate_distance_matrix(input_matrix, edge_type):


	## cria uma matriz preenchida com zeros do tamanho da quantidade de nós
	## complexidade de memória O(n^2) sendo n a quantidade de localidades
	distance_matrix = np.zeros((len(input_matrix), len(input_matrix)))

	for node_a, _ in enumerate(input_matrix):

		for node_b, _ in enumerate(input_matrix):

			if node_a == node_b:

				distance_matrix[node_a][node_b] = 0

			else:

				distance_matrix[node_a][node_b] = measure_distance(input_matrix, node_a, node_b, edge_type)

	return distance_matrix


def greedy_constructive_heuristic(input_matrix, distance_matrix, first_place='random'):
	"""
		
		A seleção do primeiro lugar pode impactar muito na escolha
		Então temos dois tipos de escolha:
		- Aleatória
		- Ponto Central

	"""

	if first_place == 'random':

		start_place = np.random.randint(0, len(input_matrix))

	else:

		start_place = np.argmin(np.sum(distance_matrix, axis=0))

	route = np.array([], dtype=int)

	route = np.append(route, start_place)

	all_places = np.array(range(0, len(input_matrix)))

	total_costs = 0

	while len(route) != len(input_matrix):

		places_available = np.setdiff1d(all_places, route)

		places_available = places_available[places_available != start_place]

		# qual é o local de menor distância que não esta a rota ainda?
		index_place = np.argmin(distance_matrix[start_place][places_available])

		total_costs += distance_matrix[start_place][places_available[index_place]]

		start_place = places_available[index_place]

		route = np.append(route, start_place)


	## calculando a qualidade da solução

	return total_costs


def plot_places(input_matrix):

	import matplotlib.pyplot as plt

	plt.figure(figsize=(10, 10))

	for index, _ in enumerate(input_matrix):

		plt.scatter(input_matrix[index][1], input_matrix[index][2])

	plt.show()


if __name__ == '__main__':


	for file_instance in os.listdir("Entrada/EUC_2D/"):

		print(file_instance)

		input_matrix, edge_type = read_instances(file_instance)

		#plot_places(input_matrix)

		distance_matrix = generate_distance_matrix(input_matrix, edge_type)

		## dado um local aleatório inicial, vá sempre para o local mais próximo a ele
		central_result = greedy_constructive_heuristic(input_matrix, distance_matrix, 'central')
		
		#random_result = greedy_constructive_heuristic(input_matrix, distance_matrix, 'random')

