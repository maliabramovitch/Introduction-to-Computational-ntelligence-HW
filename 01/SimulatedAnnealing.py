# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
Simulated Annealing on a continuous domain bounded within [lb,ub]**n
"""
import matplotlib.pyplot as plt
import numpy as np

import objFunctions as fct


def SimulatedAnnealing(n, lb, ub, max_evals, variation=lambda x: x + 2.0 * np.random.normal(size=len(x)),
                       func=lambda x: x.dot(x), seed=None):
    T_init = 6.0
    T_min = 1e-4
    alpha = 0.99
    f_lower_bound = 0
    eps_satisfactory = 1e-5
    max_internal_runs = 1000
    local_state = np.random.RandomState(seed)
    history = []
    xbest = xmin = local_state.uniform(size=n) * (ub - lb) + lb
    fbest = fmin = func(xmin)
    eval_cntr = 1
    T = T_init
    history.append(fmin)
    while ((T > T_min) and eval_cntr < max_evals):
        for _ in range(max_internal_runs):
            x = variation(xmin)
            f_x = func(x)
            eval_cntr += 1
            dE = f_x - fmin
            if dE <= 0 or local_state.uniform(size=1) < np.exp(-dE / T):
                xmin = x
                fmin = f_x
            if fmin < fbest:
                fbest = f_x
                xbest = x
                if fbest < f_lower_bound + eps_satisfactory:
                    T = T_min
                    break
            history.append(fmin)
            if np.mod(eval_cntr, int(max_evals / 10)) == 0:
                print(eval_cntr, " evals: fmin=", fmin)

        T *= alpha
    return xbest, fbest, history


#
if __name__ == "__main__":
    lb, ub = -5, 5
    n = 10
    evals = 10 ** 6
    Nruns = 2
    fbest = []
    xbest = []
    for i in range(Nruns):
        xmin, fmin, history = SimulatedAnnealing(n, lb, ub, evals, lambda x: x + 0.75 * np.random.normal(size=len(x)),
                                                 fct.WildZumba, i + 17)
        plt.semilogy(history)
        plt.show()
        print(i, ": minimal Zumba found is ", fmin, " at location ", xmin)
        fbest.append(fmin)
        xbest.append(xmin)
    print("====\n Best ever: ", min(fbest), "x*=", xbest[fbest.index(min(fbest))])
