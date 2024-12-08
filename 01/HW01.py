import numpy as np

import objFunctions as fct
from SimulatedAnnealing import SimulatedAnnealing


# --- Ofer Shir ---#
def monte_carlo(n=100, evals=1000, func=fct.SwedishPump):
    X = []
    FX = []
    fmin = 0
    xmin = 0
    for i in range(evals):
        x = np.random.choice([1, -1], size=n)
        f_x = func(x)
        X.append(x)
        FX.append(f_x)
        if i == 0:
            fmin = f_x
            xmin = x
        else:
            if fmin > f_x:
                fmin = f_x
                xmin = x
    return fmin, xmin


# --- GPT functions ---#
def Swap_Two_Elements(x):
    i, j = np.random.choice(len(x), size=2, replace=False)
    x[i], x[j] = x[j], x[i]
    return x


def Reverse_Subsequence(x):
    x = x.copy()
    i, j = sorted(np.random.choice(len(x), size=2, replace=False))
    x[i:j + 1] = x[i:j + 1][::-1]
    return x


def Insert_Element(x):
    i, j = sorted(np.random.choice(len(x), size=2, replace=False))
    x[i:j + 1] = x[i:j + 1][::-1]
    return x


def Bit_Flip(x):
    i = np.random.randint(len(x))
    x[i] = 1 - x[i]  # Flip the binary bit
    return x


variation_funcs = [Swap_Two_Elements, Reverse_Subsequence, Insert_Element, Bit_Flip]

if __name__ == "__main__":
    lb, ub = -5, 5
    n = 100
    evals = 1000
    alpha = 0.99
    Nruns = 2
    func_res = {}

    for _ in range(50):
        for var_func in variation_funcs:
            fbest = []
            xbest = []
            # print(f"Varietion function: {var_func.__name__}")
            for i in range(Nruns):
                xmin, fmin, history = SimulatedAnnealing(n, evals, var_func, fct.SwedishPump, i + 17)
                fbest.append(fmin)
                xbest.append(xmin)
            func_res[var_func.__name__] = min(fbest)
            print()
            if func_res.get(var_func) is None or func_res[var_func.__name__] > min(fbest):
                func_res[var_func.__name__] = min(fbest)

    for fr in func_res:
        print(f"{fr}: best f(x)={func_res[fr]}")

    print(f"best variation function is: {min(func_res)}")

    fmin, xmin = monte_carlo(n, evals, fct.SwedishPump)
    print(f"best f(x) found from Monte Carlo is: {fmin} ", fmin)
