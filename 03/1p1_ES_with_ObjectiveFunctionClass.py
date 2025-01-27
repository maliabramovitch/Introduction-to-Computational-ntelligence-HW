# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
(1+1)-Evolution Strategy with the 1/5th success-rule initialized within [lb,ub]**n
The objective function evaluation calls are adjusted to the ObjectiveFunctoin interface.
"""
import numpy as np

from MixedVariableObjectiveFunctions import setC


def OnePlusOneEvolutionStrategy(n, lb, ub, maxEvals, func=lambda x: x.dot(x), fstop=0, seed=None):
    local_state = np.random.RandomState(seed)
    fhistory, shistory = [], []
    xmin = local_state.uniform(size=n) * (ub - lb) + lb
    fmin = func(
        xmin.reshape(1, -1))  # reshape since it is a singleton and func receives a population in a 2D numpy array
    fhistory.append(fmin)
    sigma = (ub - lb) / 6.0
    shistory.append(sigma)
    evalcount, osuccess = 0, 0
    tol = 1e-6
    epoch = 50
    k_sigma = 0.827
    while (evalcount < maxEvals and fmin > fstop + tol):
        x = xmin + sigma * local_state.normal(size=n)
        f_x = func(
            x.reshape(1, -1))  # reshape since it is a singleton and func receives a population in a 2D numpy array
        evalcount += 1
        if f_x < fmin:
            xmin = np.copy(x)
            fmin = f_x
            osuccess += 1
        if (np.mod(evalcount, epoch) == 0):  # 1/5th success-rule every epoch
            ps = osuccess / epoch
            if (ps < 0.2):
                sigma *= k_sigma
            elif (ps > 0.2):
                sigma /= k_sigma
            osuccess = 0;
        #
        fhistory.append(fmin)
        shistory.append(sigma)
    return xmin, fmin, fhistory, shistory


#
"""
The following __main__ function applies the (1+1)-ES to 3 instances of the mixed-integer quadratic function "MixedVarsEllipsoid", 
whose Hessian matrices are generated via "ellipsoidFunctions": the 'Cigar', 'RotateEllipse', and the 'HadamardEllipse' instances.
The ES does not handle the integer constraint in a particular manner, but lets the objective function evaluation round the values 
to the nearest integer. The experimental setup runs the ES NRUNS=30 times on each of the 3 problem instances.
"""
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
            print(funcName, ": minimal objective function value found is ", fmin, " at location ", xx, " using ",
                  len(fhistory), " evaluations")
            X[NRUNS * index + k, :] = np.hstack(([k], fmin, xx))
    # print(X) # or preferably save to a file and post-process elsewhere
    # //// EOF ////
