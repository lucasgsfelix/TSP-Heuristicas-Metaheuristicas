### GRASP Algorithm

import os
import numpy as np
import common_operations



def measure_fitness(distance_matrix, individual):

	fitness = 0

	for place_a, place_b in zip(individual[:-1], individual[1:]):

		fitness += distance_matrix[place_a][place_b]

	return fitness


def build_rcl(route, input_matrix, distance_matrix, alpha=0.5):


	# beta = cmin + alpha * (cmax - cmin)

	non_visited_places = np.setdiff1d(input_matrix, route)


	edge_cost = distance_matrix[route[-1]][non_visited_places]
	# cmin
	min_gain = np.min(edge_cost)

	# cmax
	beta = min_gain + alpha * (np.max(edge_cost) - min_gain)

	# locais que são candidatos
	candidates = non_visited_places[edge_cost <= beta]

	selected_place = np.random.choice(candidates, 1)[0]

	return selected_place


def random_greedy(input_matrix, distance_matrix, alpha=0.5):

	route = np.array([np.random.choice(input_matrix, 1)[0]])

	while len(route) != len(input_matrix):

		## atualização automática do alpha
		new_place = build_rcl(route, input_matrix, distance_matrix, alpha)

		route = np.append(route, new_place)

	return route


def local_search_2opt(solution, distance_matrix):
	
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


def select_alpha(fitness, alphas):
	"""
		
		Implementação de acordo com o livro do Talbi

		## z* == fitness
		## Ai == np.mean(alphas[alpha]) --> media do alpha atual
		## qj = z*/Ai
		## qi/ sum Ai
		pi = qi/sum qj

	"""


	alpha_mean = np.array([np.mean(alphas[alpha]['results']) for alpha in alphas.keys()])

	mean_ai = fitness/alpha_mean

	# novas probabilidades
	probability_alpha = [fitness/alpha_mean[alpha_index] for alpha_index in range(0, len(alphas))]/np.sum(mean_ai)


	return np.random.choice(list(alphas.keys()), 1, p=probability_alpha)[0]


def local_search(solution, distance_matrix):

	count = 0 

	for _ in range(0, 100):

		## similar ao vnd implementado
		solution, fitness, improved = local_search_2opt(solution, distance_matrix)

		if not improved:

			count += 1

		else:

			count = 0

		if count >= 10:

			break

	return solution, fitness


def grasp_heuristic(input_matrix, distance_matrix):

	iteration, max_iterations, best_fitness, best_solution = 0, 50, 0, None

	alphas = {alpha/10: {'results': np.array([])} for alpha in range(3, 8)}

	alpha_index, count_alpha = 0, 0

	while iteration < max_iterations:

		if count_alpha < len(alphas):

			alpha = list(alphas.keys())[alpha_index]

			alpha_index += 1

			count_alpha += 1

		solution = random_greedy(input_matrix, distance_matrix, alpha)

		solution, fitness = local_search(solution, distance_matrix)

		alphas[alpha]['results'] = np.append(alphas[alpha]['results'], fitness)

		if count_alpha >= len(alphas):
			## durante as 5 primeiras rodadas utilizamos um alpha de cada
			## para poder ter insumo e fazer o calculo
			alpha = select_alpha(fitness, alphas)

		# selectiona o alpha com base na probabilidade

		if iteration == 0 or fitness < best_fitness:

			best_fitness = fitness

			best_solution = solution

		print(iteration, best_fitness)

		iteration += 1

	return best_solution, best_fitness


if __name__ == '__main__':


	for file_instance in os.listdir("Entrada/EUC_2D/"):

		input_matrix, edge_type = common_operations.read_instances(file_instance)

		distance_matrix = common_operations.generate_distance_matrix(input_matrix, edge_type)
		
		solution, fitness = grasp_heuristic(np.array(range(0, len(input_matrix))), distance_matrix)

		common_operations.write_results(file_instance, fitness, "results.txt")
