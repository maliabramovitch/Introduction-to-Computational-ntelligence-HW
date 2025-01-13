# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""

import numpy as np


def EA(n, max_evals, mutatefct, selectfct, fitnessfct, seed=None):
    # EA population: TODOs
    mu = ...
    lmbda = ...
    evalcount = 0
    histf = []
    Pop = np.array([n, mu])
    fpop = np.array([1, mu])
