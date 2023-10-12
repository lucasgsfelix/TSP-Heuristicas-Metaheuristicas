"""



"""

import common_operations

import os
import numpy as np


from itertools import groupby
from functools import partial

def generate_solution(input_matrix_size):

	all_places = np.array(range(0, input_matrix_size))

	np.random.shuffle(all_places)

	return all_places

def generate_initial_population(input_matrix_size, population_size):

	return np.array([generate_solution(input_matrix_size) for _ in range(0, population_size)])


def measure_fitness(distance_matrix, individual):

	fitness = 0

	for place_a, place_b in zip(individual[:-1], individual[1:]):

		fitness += distance_matrix[place_a][place_b]

	return fitness


def tournament_selection(fitness):

	selected = np.array([], dtype=int)

	# fitness tem o tamanho da população
	for _ in range(0, len(fitness)):

		individual_one = np.random.randint(0, len(fitness))

		individual_two = np.random.randint(0, len(fitness))

		selected_individual = np.argmin([fitness[individual_one], fitness[individual_two]])

		selected = np.append(selected, [individual_one, individual_two][selected_individual])

	return selected


def generate_ox_individual(individual_one, individual_two, cut_min, cut_max):

	individual = np.append(individual_one[cut_max: ],
						   individual_two[cut_min: cut_max])

	individual = np.append(individual,
						   	individual_one[0: cut_min])

	individual = individual.astype(int)

	# remove duplicatas da rota
	individual = np.array(list(dict.fromkeys(individual)))

	## agora temos que garantir que todos os locais estão na rota gerada

	## verifica o tamanho do individuo
	if len(individual) >= len(individual_one):

		## nesse caso temos mais locais do que deveria

		individual = individual[0: len(individual_one)]

	else:

		## nesse caso há menos, então tá faltando locais na rota
		## vamos adicionar esses locais no final

		not_in_route = np.setdiff1d(np.array(range(0, len(individual_one))), individual)

		np.random.shuffle(not_in_route)

		individual = np.append(individual, not_in_route)

	return individual


def ox_crossover(individual_one, individual_two):


	first_cut = np.random.randint(1, len(individual_one) - 1)

	second_cut = np.random.randint(1, len(individual_one) - 1)

	cut_min = np.min([first_cut, second_cut])

	cut_max = np.max([first_cut, second_cut])

	new_one = generate_ox_individual(individual_one, individual_two, cut_min, cut_max)

	new_two = generate_ox_individual(individual_two, individual_one, cut_min, cut_max)

	return new_one, new_two


def crossover_operator(population, fitness):

	### os individuos aqui devem chegar ordenados pelo fitness

	## salvando o melhor individual
	#best_individual = population[0]


	for i in range(1, len(population), 2):

		individual_one = population[i - 1]

		individual_two = population[i]

		individual_one, individual_two = ox_crossover(individual_one, individual_two)

		population[i - 1] = individual_one

		population[i] = individual_two

	#population = np.concatenate((population, [best_individual]))

	return population


def mutation_operator(population, mutation_rate):

	for index, individual in enumerate(population):

		if np.random.random(1)[0] <= mutation_rate:

			## mutando diversos genes
			amount_muted_genes = np.random.randint(0, len(individual))

			for _ in range(0, amount_muted_genes):

				gene_index_a = np.random.randint(0, len(individual))

				gene_index_b = np.random.randint(0, len(individual))

				gene_a, gene_b = individual[gene_index_a], individual[gene_index_b]

				population[index][gene_index_a] = gene_a

				population[index][gene_index_b] = gene_b

	return population


def genetic_algorithm(input_matrix, distance_matrix):


	params = {
		"generations": 100,
		"population": 100,
		"mutation_rate": 0.05
	}

	population_params = {
		"best_solution": None,
		"best_fitness": None
	}

	generation_equal = 0

	population = generate_initial_population(len(input_matrix), params['population'])

	# import construtive_heuristic

	# const_solution, _ = construtive_heuristic.greedy_constructive_heuristic(input_matrix, distance_matrix, 'central')

	# const_solution -= 1

	# population = np.concatenate((population, [const_solution]))

	#population = np.concatenate((population, [const_solution]))

	#print(population)

	#exit()

	measure_indivdual_fitness = partial(measure_fitness, distance_matrix)

	for generation in range(0, params['generations']):

		fitness = np.array(list(map(measure_indivdual_fitness, population)))

		best_index = np.argmin(fitness)

		if generation == 0 or population_params['best_fitness'] > fitness[best_index]:

			population_params['best_fitness'] = fitness[best_index]

			population_params['best_solution'] = population[best_index]

			generation_equal = 0

		else:

			generation_equal += 1

		## agora começam as operações genéticas
		print(population_params['best_fitness'])

		##selection - tem uma parte que é por elitismo
		selected_individuals = tournament_selection(fitness)

		## crossover - os individuos mais aptos cruzam com os mais aptos
		sort_index = np.argsort(fitness[selected_individuals])

		population = crossover_operator(population[selected_individuals][sort_index],
										fitness[selected_individuals][sort_index])


		if generation_equal % 5 == 0 and params['mutation_rate'] <= 0.20:

			params['mutation_rate'] += 0.02


		population = mutation_operator(population, params['mutation_rate'])



if __name__ == '__main__':


	for file_instance in os.listdir("Entrada/EUC_2D/"):

		print(file_instance)

		input_matrix, edge_type = common_operations.read_instances(file_instance)

		#plot_places(input_matrix)

		distance_matrix = common_operations.generate_distance_matrix(input_matrix, edge_type)

		
		genetic_algorithm(input_matrix, distance_matrix)		

		break