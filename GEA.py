import random
import math

# PARAMETERS
POP_SIZE = 6
CHROM_LENGTH = 7         # length of genetic sequence
FUNCTIONS = ['+', '-', '*']
TERMINALS = [str(i) for i in range(10)]  # constants 0–9
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
GENERATIONS = 20

# HELPER FUNCTIONS
def random_gene():
    """Return a random gene (either function or terminal)."""
    if random.random() < 0.4:
        return random.choice(FUNCTIONS)
    return random.choice(TERMINALS)

def create_individual():
    """Generate a random chromosome (sequence)."""
    return [random_gene() for _ in range(CHROM_LENGTH)]

def decode_expression(chromosome):
    """Convert chromosome into a valid arithmetic expression."""
    expr = ""
    for gene in chromosome:
        expr += gene
    return expr

def evaluate(chromosome):
    """Evaluate chromosome by expressing it as integer x, then f(x)=x^2."""
    expr = decode_expression(chromosome)
    try:
        # Evaluate safely
        x_val = int(eval(expr))  
    except Exception:
        return 0  # invalid expression
    if x_val < 0 or x_val > 31:  # constrain to problem domain
        return 0
    return x_val**2

def roulette_wheel_selection(pop, fitnesses):
    """Select one individual using roulette wheel."""
    total_fit = sum(fitnesses)
    if total_fit == 0:
        return random.choice(pop)
    pick = random.uniform(0, total_fit)
    current = 0
    for i, f in enumerate(fitnesses):
        current += f
        if current > pick:
            return pop[i]

def crossover(parent1, parent2):
    """Single point crossover."""
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]
    point = random.randint(1, CHROM_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    """Mutate chromosome by flipping a gene."""
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = random_gene()
    return chromosome

# MAIN LOOP
population = [create_individual() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    fitnesses = [evaluate(ind) for ind in population]
    best_index = fitnesses.index(max(fitnesses))
    best = population[best_index]
    print(f"Gen {gen+1}: Best = {decode_expression(best)}, f(x) = {max(fitnesses)}")

    # New population
    new_population = []
    while len(new_population) < POP_SIZE:
        p1 = roulette_wheel_selection(population, fitnesses)
        p2 = roulette_wheel_selection(population, fitnesses)
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        new_population.extend([c1, c2])
    population = new_population[:POP_SIZE]

# Final result
fitnesses = [evaluate(ind) for ind in population]
best_index = fitnesses.index(max(fitnesses))
best = population[best_index]
print("\nFinal Best Solution:")
print("Chromosome:", decode_expression(best))
print("Fitness:", max(fitnesses))
