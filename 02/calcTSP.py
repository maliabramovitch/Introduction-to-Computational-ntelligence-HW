# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""

import matplotlib.pyplot as plt
import numpy as np

from MyGA import MyGA

NTrials = 10**3


def computeTourLength(perm, Graph):
    tlen = 0.0
    for i in range(len(perm)):
        tlen += Graph[perm[i], perm[((i + 1) % len(perm))]]
    return tlen


def roulette_wheel_selection(fitnessPop, mu):
    # Convert fitness values to probabilities (lower is better)
    total_fitness = np.sum(1 / fitnessPop)
    probabilities = (1 / fitnessPop) / total_fitness
    # Select indices with replacement based on probabilities
    return np.random.choice(len(fitnessPop), size=mu, p=probabilities)


if __name__ == "__main__":
    myga = MyGA(150, (lambda x: x), roulette_wheel_selection, computeTourLength)
    tourStat = []
    for k in range(NTrials):
        myga.run()
        tourStat.append(myga.fmin)
    plt.hist(tourStat, bins=100)
    plt.show()
