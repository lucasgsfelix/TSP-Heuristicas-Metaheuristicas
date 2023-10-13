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

	while solution_improving:

		solution_improving = False

		for index_a in range(1, len(solution) - 2):

			for index_b in range(index_a + 1, len(solution)):

				# caso onde index_a e index_b são adjascentes
				if index_b - index_a == 1:

					continue

				new_solution = solution.copy()

				## swap
				new_solution[index_a], new_solution[index_b] = solution[index_b], solution[index_a]

				if measure_fitness(distance_matrix, solution) > measure_fitness(distance_matrix, new_solution):

					solution = new_solution

					solution_improving = True

	return solution

def variable_neighborhood_descent(input_matrix, distance_matrix):

	solution = generate_solution(len(input_matrix))

	print(measure_fitness(distance_matrix, solution))

	for _ in range(0, 100):

		solution = solution_neighborhood_2opt(solution, distance_matrix)

		print(measure_fitness(distance_matrix, solution))

	return start_solution


if __name__ == '__main__':


	for file_instance in os.listdir("Entrada/EUC_2D/"):

		print(file_instance)

		input_matrix, edge_type = common_operations.read_instances('st70.tsp')

		distance_matrix = common_operations.generate_distance_matrix(input_matrix, edge_type)
		
		solution = variable_neighborhood_descent(input_matrix, distance_matrix)

		break