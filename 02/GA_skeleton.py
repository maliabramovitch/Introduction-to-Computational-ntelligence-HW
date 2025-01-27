# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""


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


# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def computeTourLength(perm, Graph):
    tlen = 0.0
    for i in range(len(perm)):
        tlen += Graph[perm[i], perm[np.mod(i + 1, len(perm))]]
    return tlen


if __name__ == "__main__":
    dirname = ""
    fname = os.path.join(dirname, "tokyo.dat")
    data = []
    NTrials = 10 ** 5
    with open(fname) as f:
        for line in f:
            data.append(line.split())
    n = len(data)
    G = np.empty([n, n])
    for i in range(n):
        for j in range(i, n):
            G[i, j] = np.linalg.norm(
                np.array([float(data[i][1]), float(data[i][2])]) - np.array([float(data[j][1]), float(data[j][2])]))
            G[j, i] = G[i, j]
    #
    tourStat = []
    for k in range(NTrials):
        tourStat.append(computeTourLength(np.random.permutation(n), G))
    plt.hist(tourStat, bins=100)
    plt.show()
