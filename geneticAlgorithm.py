import random

# Fitness function
def fitness(x):
    return x**2   # maximize x^2

# Convert integer to 5-bit binary string
def int_to_bin(x):
    return format(x, '05b')

# Convert 5-bit binary string to integer
def bin_to_int(b):
    return int(b, 2)

# Initialize population
def init_population(size):
    return [random.randint(0, 31) for _ in range(size)]

# Selection: Tournament Selection
def tournament_selection(pop, k=3):
    selected = random.sample(pop, k)
    selected.sort(key=lambda x: fitness(x), reverse=True)
    return selected[0]

# Crossover: Single point
def crossover(p1, p2):
    b1, b2 = int_to_bin(p1), int_to_bin(p2)
    point = random.randint(1, 4)  # crossover between bits
    child1 = bin_to_int(b1[:point] + b2[point:])
    child2 = bin_to_int(b2[:point] + b1[point:])
    return child1, child2

# Mutation: Flip one random bit
def mutate(x, mutation_rate=0.1):
    if random.random() < mutation_rate:
        b = list(int_to_bin(x))
        pos = random.randint(0, 4)
        b[pos] = '1' if b[pos] == '0' else '0'
        return bin_to_int("".join(b))
    return x

# Main GA
def genetic_algorithm(pop_size=6, generations=20, crossover_rate=0.8, mutation_rate=0.1):
    population = init_population(pop_size)

    for gen in range(generations):
        # Sort by fitness
        population.sort(key=lambda x: fitness(x), reverse=True)
        best = population[0]
        print(f"Gen {gen}: Best x={best}, f(x)={fitness(best)}")

        new_pop = [best]  # elitism: carry best forward

        while len(new_pop) < pop_size:
            # Parent selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_pop.extend([child1, child2])

        population = new_pop[:pop_size]

    # Final result
    population.sort(key=lambda x: fitness(x), reverse=True)
    best = population[0]
    print(f"\nBest Solution: x={best}, f(x)={fitness(best)}")

# Run the GA
genetic_algorithm()
