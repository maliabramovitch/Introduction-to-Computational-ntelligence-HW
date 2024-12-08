import numpy as np
import objFunctions as fct
from SimulatedAnnealing import SimulatedAnnealing
import matplotlib.pyplot as plt

# --- Ofer Shir ---#
def monte_carlo(lb=-5, ub=5, n=100, evals=1000, func=fct.SwedishPump):
    X = []
    FX = []
    fmin = 0
    xmin = 0
    f_history = []
    for i in range(evals):
        x = np.random.uniform(size=n) * (ub - lb) + lb  # alternatively: normal
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
        f_history.append(fmin)  # Track the best f(x) at each iteration
    return fmin, xmin, f_history

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

n = 100
evals = 1000
alpha = 0.99
Nruns = 2
func_res = {}
history_dict = {}

for _ in range(50):
    for var_func in variation_funcs:
        fbest = []
        xbest = []
        history = []
        for i in range(Nruns):
            xmin, fmin, h = SimulatedAnnealing(n, evals, var_func, fct.SwedishPump, i + 17)
            fbest.append(fmin)
            xbest.append(xmin)
            history.append(h)  # Store the history of function values

        func_res[var_func.__name__] = min(fbest)
        history_dict[var_func.__name__] = history  # Store the history for plotting

# Print results
for fr in func_res:
    print(f"{fr}: best f(x)={func_res[fr]}")

best_func = min(func_res, key=func_res.get)
print(f"Best variation function is: {best_func}")

fmin_mc, xmin_mc, f_history_mc = monte_carlo(n, evals, fct.SwedishPump)
print(f"Best f(x) found from Monte Carlo is: {fmin_mc}")

# Plotting the function histories for each variation function
plt.figure(figsize=(10, 6))
for var_func in variation_funcs:
    plt.plot(np.mean(history_dict[var_func.__name__], axis=0), label=var_func.__name__)

# Running Monte Carlo and plotting the result
plt.plot(f_history_mc, label='Monte Carlo', linestyle='--', color='black')

plt.title('Objective Function Value History for Each Variation Function and Monte Carlo')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value (f(x))')
plt.legend()
plt.grid(True)
plt.show()