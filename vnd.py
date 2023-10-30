import common_operations

import os
import numpy as np



def measure_fitness(distance_matrix, individual):

	fitness = 0

	for place_a, place_b in zip(individual[:-1], individual[1:]):

		fitness += distance_matrix[place_a][place_b]

	return fitness


def generate_solution(input_matrix_size):

	all_places = np.array(range(0, input_matrix_size))

	np.random.shuffle(all_places)

	return all_places


def solution_neighborhood_2opt(solution, distance_matrix):

	solution_improving = True

	fitness_sol = measure_fitness(distance_matrix, solution)

	for index_a in range(1, len(solution) - 2):

		for index_b in range(index_a + 1, len(solution)):

			# caso onde index_a e index_b são adjascentes
			if index_b - index_a == 1:

				continue

			new_solution = solution.copy()

			## swap
			new_solution[index_a], new_solution[index_b] = solution[index_b], solution[index_a]

			fitness_nova_sol = measure_fitness(distance_matrix, new_solution)

			if fitness_sol > fitness_nova_sol:

				return new_solution, fitness_nova_sol, True

	return solution, fitness_sol, False



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

	return route, total_costs


def variable_neighborhood_descent(input_matrix, distance_matrix):

	#solution = generate_solution(len(input_matrix))

	solution, _ = greedy_constructive_heuristic(input_matrix, distance_matrix, 'central')

	count = 0

	for _ in range(0, 1000):

		solution, fitness, improved = solution_neighborhood_2opt(solution, distance_matrix)

		if not improved:

			count += 1

		else:

			count = 0

		if count >= 100:

			break

	return solution, fitness





if __name__ == '__main__':


	for file_instance in os.listdir("Entrada/EUC_2D/"):

		input_matrix, edge_type = common_operations.read_instances(file_instance)

		distance_matrix = common_operations.generate_distance_matrix(input_matrix, edge_type)
		
		solution, fitness = variable_neighborhood_descent(input_matrix, distance_matrix)

		common_operations.write_results(file_instance, fitness, "results.txt")
