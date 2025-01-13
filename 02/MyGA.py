# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""

import os

import numpy as np

# GA params
MU = 50  # Population size
PC = 0.8  # Probability of crossover
PM = 0.1  # Probability of mutation
MAX_EVALS = 10 ** 5


class MyGA:
    def __init__(self, n, decode_fct, select_fct, fitness_fct, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.n = n
        self.distance_matrix = None

        self.decode_fct = decode_fct
        self.fitness_fct = fitness_fct
        self.select_fct = select_fct

        self.history = []
        self.fmin = np.inf
        self.xmin = None

        self.calculate_distance_matrix()

    def run(self):
        genome = np.array([np.random.permutation(self.n) for _ in range(MU)])
        phenotype = np.array([self.decode_fct(_genome) for _genome in genome])
        fitness_pop = np.array([self.fitness_fct(_phenotype, self.distance_matrix) for _phenotype in phenotype])
        eval_cntr = MU
        self.fmin = np.min(fitness_pop)
        self.xmin = phenotype[np.argmin(fitness_pop)]
        self.history.append(self.fmin)

        while eval_cntr < MAX_EVALS:
            parents_indices = self.select_fct(fitness_pop, MU)
            parents = genome[parents_indices]

            # Generate offspring through crossover
            offspring = []
            for i in range(0, MU, 2):
                p1, p2 = parents[i], parents[(i + 1) % MU]
                if np.random.rand() < PC:
                    # Single-point crossover
                    point = np.random.randint(1, self.n - 1)
                    child1 = np.concatenate([p1[:point], p2[point:]])
                    child2 = np.concatenate([p2[:point], p1[point:]])
                else:
                    child1, child2 = p1, p2
                offspring.extend([child1, child2])

            # Mutation
            for child in offspring:
                if np.random.rand() < PM:
                    # Swap two random positions
                    i, j = np.random.choice(self.n, size=2, replace=False)
                    child[i], child[j] = child[j], child[i]

            # Replace the old population with the new offspring
            genome = np.array(offspring)

            # Decode and evaluate new population
            phenotype = np.array([self.decode_fct(_genome) for _genome in genome])
            fitness_pop = np.array([self.fitness_fct(_phenotype, self.distance_matrix) for _phenotype in phenotype])
            eval_cntr += MU

            # Track the best solution in the current generation
            gen_min_fitness = np.min(fitness_pop)
            if gen_min_fitness < self.fmin:
                self.fmin = gen_min_fitness
                self.xmin = phenotype[np.argmin(fitness_pop)]

            # Save the best fitness value for this generation
            self.history.append(self.fmin)

    def calculate_distance_matrix(self):
        """
        @author: ofersh@telhai.ac.il
        """
        dirname = ""
        fname = os.path.join(dirname, "tokyo.dat")
        data = []
        NTrials = 10 ** 5
        with open(fname) as f:
            for line in f:
                data.append(line.split())
        n = len(data)
        self.distance_matrix = np.empty([n, n])
        for i in range(n):
            for j in range(i, n):
                self.distance_matrix[i, j] = np.linalg.norm(
                    np.array([float(data[i][1]), float(data[i][2])]) - np.array([float(data[j][1]), float(data[j][2])]))
                self.distance_matrix[j, i] = self.distance_matrix[i, j]
