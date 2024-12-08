# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
import math

import matplotlib.pyplot as plt
import numpy as np


def WhosYourDaddy(b):
    """ The counting-ones function, assumes a binary numpy vector as input """
    return (int)(np.sum(b))


def SwedishPump(b):
    """ The correlation function, assumes a numpy vector {-1,+1} as input """
    n = np.size(b)
    E = []
    for k in range(1, n):
        E.append((b[:n - k].dot(b[k:])) ** 2)
    return (n ** 2) / (2 * sum(E))


def WildZumba(x, c1=20, c2=0.2, c3=2 * np.pi):
    """ A separable R**n==>R function, assumes a real-valued numpy vector as input """
    return -c1 * np.exp(-c2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(c3 * x))) + c1 + np.exp(1)


if __name__ == "__main__":
    samples, N = 50000, 100
    numBins = N + 1
    bMat = np.random.randint(2, size=(samples, N))
    fb = []
    for i in range(samples):
        fb.append(WhosYourDaddy(bMat[i]))
    bins = np.linspace(math.ceil(min(fb)), math.floor(max(fb)), numBins)
    plt.hist(fb, bins=bins)
    plt.show()

    oMat = 2 * np.random.randint(2, size=(samples, N)) - 1
    fo = []
    for i in range(samples):
        fo.append(SwedishPump(oMat[i]))
    bins = np.linspace(math.ceil(min(fo)), math.floor(max(fo)), numBins)
    plt.hist(fo)
    plt.show()

    xMat = np.random.normal(size=(samples, N))
    fx = []
    for i in range(samples):
        fx.append(WildZumba(xMat[i]))

    plt.hist(fx)
    plt.show()
