# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""

import numpy as np


def GA(n, max_evals, decodefct, selectfct, fitnessfct, seed=None):
    if seed is not None:
        np.random.seed(seed)

    eval_cntr = 0
    history = []
    fmin = np.inf
    xmin = None  # Will hold the best phenotype

    # GA params
    mu = 50  # Population size
    pc = 0.8  # Probability of crossover
    pm = 0.1  # Probability of mutation

    # Initialize population (Genome)
    # Binary genome: random initialization of mu individuals with n genes each
    Genome = np.random.randint(2, size=(mu, n))

    # Decode genome to phenotype
    Phenotype = np.array([decodefct(genome) for genome in Genome])

    # Evaluate fitness of the initial population
    fitnessPop = np.array([fitnessfct(phenotype) for phenotype in Phenotype])
    eval_cntr += mu  # Each individual was evaluated once

    # Track the best solution
    fmin = np.min(fitnessPop)
    xmin = Phenotype[np.argmin(fitnessPop)]
    history.append(fmin)

    while eval_cntr < max_evals:
        # Selection (e.g., tournament or roulette wheel selection)
        parents_indices = selectfct(fitnessPop, mu)
        parents = Genome[parents_indices]

        # Generate offspring through crossover
        offspring = []
        for i in range(0, mu, 2):
            p1, p2 = parents[i], parents[(i + 1) % mu]
            if np.random.rand() < pc:
                # Single-point crossover
                point = np.random.randint(1, n - 1)
                child1 = np.concatenate([p1[:point], p2[point:]])
                child2 = np.concatenate([p2[:point], p1[point:]])
            else:
                child1, child2 = p1, p2
            offspring.extend([child1, child2])

        # Mutation
        for child in offspring:
            if np.random.rand() < pm:
                mutation_point = np.random.randint(n)
                child[mutation_point] = 1 - child[mutation_point]  # Flip the bit

        # Replace the old population with the new offspring
        Genome = np.array(offspring)

        # Decode and evaluate new population
        Phenotype = np.array([decodefct(genome) for genome in Genome])
        fitnessPop = np.array([fitnessfct(phenotype) for phenotype in Phenotype])
        eval_cntr += mu

        # Track the best solution in the current generation
        gen_min_fitness = np.min(fitnessPop)
        if gen_min_fitness < fmin:
            fmin = gen_min_fitness
            xmin = Phenotype[np.argmin(fitnessPop)]

        # Save the best fitness value for this generation
        history.append(fmin)

    return xmin, fmin, history
