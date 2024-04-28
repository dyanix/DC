import random
from deap import base, creator, tools, algorithms
import numpy as np

# Fitness function (maximize sum of elements)
def eval_func(individual):
    return sum(individual),  # maximize the sum

# Create fitness and individual types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox initialization
toolbox = base.Toolbox()

# Attribute creation operator (binary values)
toolbox.register("attr_bool", random.randint, 0, 1)

# Individual creation operator (list of 10 booleans)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=10)

# Population creation operator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function registration
toolbox.register("evaluate", eval_func)

# Mating operator (two-point crossover)
toolbox.register("mate", tools.cxTwoPoint)

# Mutation operator (flip bit with 5% probability)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# Selection operator (tournament selection, size 3)
toolbox.register("select", tools.selTournament, tournsize=3)
# Create initial population
population = toolbox.population(n=300)

# Start of evolutionary algorithm
print("Start of evolution")

# Hall of Fame (stores best individual)
hof = tools.HallOfFame(1)

# Define statistics to track
stats = tools.Statistics()
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
# Run genetic algorithm
population, log = algorithms.eaSimple(
    population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True
)

# Print best individual
print("Best individual:", hof[0])