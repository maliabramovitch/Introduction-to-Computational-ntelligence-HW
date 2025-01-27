# -*- coding: utf-8 -*-
"""
Enhanced (1+1)-Evolution Strategy for Mixed-Integer Optimization.
Includes:
1. Separate mutation strategies for continuous and integer variables.
2. Dynamic mutation step-size adjustment (adaptive mutation).
3. Local search for integer variables.
4. Randomized rounding for integer variables.
5. Improved handling of both variable types.

@author: YourName
"""

import numpy as np
from MixedVariableObjectiveFunctions import setC
import MixedVariableObjectiveFunctions as f_mixed
import ellipsoidFunctions as Efunc

def mixed_mutation(x, sigma_c, sigma_z, lb, ub, int_indices):
    """
    Mixed mutation operator: applies Gaussian noise to continuous variables
    and discrete steps to integer variables.
    """
    x_new = np.copy(x)
    for i in range(len(x)):
        if i in int_indices:  # Integer variable
            step = np.round(sigma_z * np.random.normal())
            x_new[i] = np.clip(x[i] + step, lb, ub)  # Ensure bounds
        else:  # Continuous variable
            x_new[i] = np.clip(x[i] + sigma_c * np.random.normal(), lb, ub)
    return x_new

def randomized_rounding(x, int_indices):
    """
    Applies randomized rounding to integer variables in the solution vector.
    """
    for i in int_indices:
        frac_part = x[i] - np.floor(x[i])
        x[i] = np.floor(x[i]) + (np.random.rand() < frac_part)
    return x

def local_search(x, func, int_indices, lb, ub):
    """
    Perform local search for integer variables around the current solution.
    """
    x_best = np.copy(x)
    f_best = func(x_best.reshape(1, -1))
    for i in int_indices:
        for delta in [-1, 1]:
            x_new = np.copy(x)
            x_new[i] = np.clip(x_new[i] + delta, lb, ub)
            f_new = func(x_new.reshape(1, -1))
            if f_new < f_best:
                x_best, f_best = x_new, f_new
    return x_best

def OnePlusOneEvolutionStrategy(n, lb, ub, maxEvals, func=lambda x: x.dot(x), fstop=0, seed=None):
    local_state = np.random.RandomState(seed)
    fhistory, shistory = [], []
    xmin = local_state.uniform(size=n) * (ub - lb) + lb
    fmin = func(xmin.reshape(1, -1))  # Initial evaluation
    fhistory.append(fmin)
    sigma_c = (ub - lb) / 6.0  # Step size for continuous variables
    sigma_z = 1.0  # Initial step size for integer variables
    shistory.append(sigma_c)
    evalcount, osuccess = 0, 0
    tol = 1e-6
    epoch = 50
    k_sigma = 0.827
    int_indices = range(n // 2, n)  # Integer variables are in the second half

    while evalcount < maxEvals and fmin > fstop + tol:
        x = mixed_mutation(xmin, sigma_c, sigma_z, lb, ub, int_indices)
        x = randomized_rounding(x, int_indices)  # Apply randomized rounding
        x = local_search(x, func, int_indices, lb, ub)  # Local search for integer variables
        f_x = func(x.reshape(1, -1))
        evalcount += 1

        if f_x < fmin:
            xmin = np.copy(x)
            fmin = f_x
            osuccess += 1

        if evalcount % epoch == 0:  # Adapt step sizes using 1/5th success rule
            ps = osuccess / epoch
            if ps < 0.2:
                sigma_c *= k_sigma
                sigma_z *= k_sigma
            elif ps > 0.2:
                sigma_c /= k_sigma
                sigma_z /= k_sigma
            osuccess = 0

        fhistory.append(fmin)
        shistory.append(sigma_c)

    return xmin, fmin, fhistory, shistory

if __name__ == "__main__":
    lb, ub = -100, 100
    dim = 64
    N = dim // 2
    setC(N)
    c = 100
    budget = 1e6
    NRUNS = 30
    X = np.full((3 * NRUNS, dim + 2), np.nan)
    objFunc = "MixedVarsEllipsoid"

    for index, funcName in enumerate(['genHcigar', 'genRotatedHellipse', 'genHadamardHellipse']):
        H = eval(f'Efunc.{funcName}')(dim, c)
        f = eval(f'f_mixed.{objFunc}')(d=dim, bid=0, ind=N, H=H, c=c, max_eval=budget)

        for k in range(NRUNS):
            xmin, fmin, fhistory, shistory = OnePlusOneEvolutionStrategy(dim, lb, ub, budget, func=f)
            xx = np.array([xmin[index] if index < N else np.round(xmin[index]) for index in range(len(xmin))])
            print(f"{funcName}: minimal objective function value found is {fmin}\nat location {xx}\nusing {len(fhistory)} evaluations")
            X[NRUNS * index + k, :] = np.hstack(([k], fmin, xx))

    print(X)  # Save to file or process further as needed
