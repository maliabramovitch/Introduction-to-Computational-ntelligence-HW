import matplotlib.pyplot as plt
import numpy as np
import objFunctions as fct
import SimulatedAnnealing


def Swap_Two_Elements(x):
    i, j = np.random.choice(len(x), size=2, replace=False)
    x[i], x[j] = x[j], x[i]
    return x


def Reverse_Subsequence(x):
    i, j = np.random.choice(len(x), size=2, replace=False)
    elem = x.pop(i)
    x.insert(j, elem)
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
    fbest = []
    xbest = []
    func_res = {}

    for var_func in variation_funcs:
        print(f"Varietion function: {var_func.__name__}")
        for i in range(Nruns):
            xmin, fmin, history = SimulatedAnnealing(n, lb, ub, evals, var_func,
                                                     fct.SwedishPump, i + 17)
            plt.semilogy(history)
            plt.show()
            print(i, ": Swedish Pump found is ", fmin, " at location ", xmin)
            fbest.append(fmin)
            xbest.append(xmin)
        print("====\n Best ever: ", min(fbest), "x*=", xbest[fbest.index(min(fbest))])
        func_res[var_func.__name__] = [f"Best f(x)={fbest}", f"best xx={xbest[fbest.index(min(fbest))]}"]
        print()
    for fr in func_res:
        print(f"{fr}: {func_res[fr]}")
