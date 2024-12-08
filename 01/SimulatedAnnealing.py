# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
def SimulatedAnnealing(n=100, max_evals=1000, variation=lambda x: x + 2.0 * np.random.normal(size=len(x)),
                       func=lambda x: x.dot(x), seed=None):
    T_init = 6.0
    T_min = 1e-4
    alpha = 0.99
    f_lower_bound = 0
    eps_satisfactory = 1e-5
    max_internal_runs = 1000
    local_state = np.random.RandomState(seed)
    history = []
    xbest = xmin = np.random.choice([1, -1], size=n)
    fbest = fmin = func(xmin)
    eval_cntr = 1
    T = T_init
    history.append(fmin)
    while (T > T_min) and eval_cntr < max_evals:
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
            # if np.mod(eval_cntr, int(max_evals / 10)) == 0:
            #     print(eval_cntr, " evals: fmin=", fmin)

        T *= alpha
    return xbest, fbest, history
