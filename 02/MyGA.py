import os
import numpy as np

# GA params (Genetic Algorithm parameters)
MU = 50  # Population size
PC = 0.8  # Probability of crossover
PM = 0.1  # Probability of mutation
MAX_EVALS = 10 ** 2  # Maximum number of evaluations
MIN_DISTANCE = 6528


class MyGA:
    def __init__(self, n, decode_fct, select_fct, fitness_fct, seed=None):
        """
        Initialize the Genetic Algorithm.

        Args:
            n (int): Number of elements in a genome (problem size).
            decode_fct (function): Function to decode a genome to its phenotype.
            select_fct (function): Function to select parents based on fitness.
            fitness_fct (function): Function to calculate fitness of a phenotype.
            seed (int, optional): Seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility

        self.n = n
        self.distance_matrix = None  # Matrix to hold distances between elements (if required)

        self.decode_fct = decode_fct  # Decoding function
        self.fitness_fct = fitness_fct  # Fitness evaluation function
        self.select_fct = select_fct  # Parent selection function

        # Historical tracking and best solution storage
        self.history = []  # Store best fitness over generations
        self.fmin = None  # Best fitness found so far
        self.xmin = None  # Phenotype corresponding to the best fitness
        self.genome = None  # Current population of genomes
        self.phenotype = None  # Decoded phenotypes of the current population
        self.fitness_pop = None  # Fitness values of the current population

        # Precompute the distance matrix if applicable
        self.calculate_distance_matrix()

    def run(self):
        """
        Execute the Genetic Algorithm to optimize the problem.
        """
        # Initialize the population if not already done
        if self.genome is None:
            # Generate initial population with random permutations
            self.genome = np.array([np.random.permutation(self.n) for _ in range(MU)])
            # Decode the population
            self.phenotype = np.array([self.decode_fct(_genome) for _genome in self.genome])
            # Evaluate fitness for the population
            self.fitness_pop = np.array(
                [self.fitness_fct(_phenotype, self.distance_matrix) for _phenotype in self.phenotype])
            # Track the best solution in the initial population
            self.fmin = np.min(self.fitness_pop)
            self.xmin = self.phenotype[np.argmin(self.fitness_pop)]
            # Save the best fitness value for the first generation
            self.history.append(self.fmin)

        # Counter for number of fitness evaluations
        eval_cntr = MU

        # Main loop of the algorithm
        while eval_cntr < MAX_EVALS and self.fmin != MIN_DISTANCE:
            # Select parents based on fitness
            parents_indices = self.select_fct(self.fitness_pop, MU)
            parents = self.genome[parents_indices]

            # Generate offspring through crossover
            offspring = []
            for i in range(0, MU, 2):
                p1, p2 = parents[i], parents[(i + 1) % MU]  # Pair parents
                if np.random.rand() < PC:  # Perform crossover with probability PC
                    # Single-point crossover
                    point = np.random.randint(1, self.n - 1)
                    child1 = np.concatenate([p1[:point], p2[point:]])
                    child2 = np.concatenate([p2[:point], p1[point:]])
                else:  # No crossover, copy parents
                    child1, child2 = p1, p2
                offspring.extend([child1, child2])

            # Apply mutation to offspring
            for child in offspring:
                if np.random.rand() < PM:  # Perform mutation with probability PM
                    # Swap two random positions
                    i, j = np.random.choice(self.n, size=2, replace=False)
                    child[i], child[j] = child[j], child[i]

            # Replace the old population with the new offspring
            self.genome = np.array(offspring)

            # Decode and evaluate new population
            self.phenotype = np.array([self.decode_fct(_genome) for _genome in self.genome])
            self.fitness_pop = np.array(
                [self.fitness_fct(_phenotype, self.distance_matrix) for _phenotype in self.phenotype])
            eval_cntr += MU  # Update evaluation counter

            # Track the best solution in the current generation
            gen_min_fitness = np.min(self.fitness_pop)
            if gen_min_fitness < self.fmin:  # Check for improvement
                self.fmin = gen_min_fitness
                self.xmin = self.phenotype[np.argmin(self.fitness_pop)]

            # Save the best fitness value for this generation
            self.history.append(self.fmin)

    def calculate_distance_matrix(self):
        """
        Precompute the distance matrix for use in fitness calculations.
        Reads data from a file named "tokyo.dat".
        """
        dirname = ""  # Directory of the file
        fname = os.path.join(dirname, "tokyo.dat")  # Path to data file
        data = []  # List to store data points
        NTrials = 10 ** 5  # Not used in current implementation
        with open(fname) as f:
            for line in f:
                data.append(line.split())  # Parse each line into a list
        n = len(data)  # Number of data points
        self.distance_matrix = np.empty([n, n])  # Initialize distance matrix
        for i in range(n):
            for j in range(i, n):
                # Calculate Euclidean distance between points
                self.distance_matrix[i, j] = np.linalg.norm(
                    np.array([float(data[i][1]), float(data[i][2])]) - np.array([float(data[j][1]), float(data[j][2])]))
                self.distance_matrix[j, i] = self.distance_matrix[i, j]  # Symmetric matrix
