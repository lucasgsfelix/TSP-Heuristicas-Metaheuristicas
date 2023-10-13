"""



"""

import common_operations

import os
import numpy as np


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

	return route + 1, total_costs


if __name__ == '__main__':


	for file_instance in os.listdir("Entrada/EUC_2D/"):

		sprint(file_instance)

		input_matrix, edge_type = common_operations.read_instances(file_instance)

		#plot_places(input_matrix)

		distance_matrix = common_operations.generate_distance_matrix(input_matrix, edge_type)

		## dado um local aleatório inicial, vá sempre para o local mais próximo a ele
		central_result = greedy_constructive_heuristic(input_matrix, distance_matrix, 'central')

