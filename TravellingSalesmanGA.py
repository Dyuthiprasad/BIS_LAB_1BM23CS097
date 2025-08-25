import random
import math
from typing import List, Tuple

City = Tuple[float, float]
Route = List[int]

def euclidean(a: City, b: City) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def route_length(route: Route, cities: List[City]) -> float:
    # complete tour (wrap back to start)
    dist = 0.0
    for i in range(len(route)):
        a = cities[route[i]]
        b = cities[route[(i+1) % len(route)]]
        dist += euclidean(a, b)
    return dist

def fitness(route: Route, cities: List[City]) -> float:
    # higher is better
    return 1.0 / (1e-9 + route_length(route, cities))

def tournament_selection(pop: List[Route], cities: List[City], k: int = 3) -> Route:
    contestants = random.sample(pop, k)
    return max(contestants, key=lambda r: fitness(r, cities))

def ordered_crossover(parent1: Route, parent2: Route) -> Tuple[Route, Route]:
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    # child1: take slice from p1, fill remainder in p2 order
    slice1 = parent1[a:b+1]
    child1 = [None]*n
    child1[a:b+1] = slice1
    fill = [g for g in parent2 if g not in slice1]
    idx = 0
    for i in range(n):
        if child1[i] is None:
            child1[i] = fill[idx]
            idx += 1
    # child2 symmetric
    slice2 = parent2[a:b+1]
    child2 = [None]*n
    child2[a:b+1] = slice2
    fill = [g for g in parent1 if g not in slice2]
    idx = 0
    for i in range(n):
        if child2[i] is None:
            child2[i] = fill[idx]
            idx += 1
    return child1, child2

def swap_mutation(route: Route, mutation_rate: float = 0.2) -> Route:
    r = route[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(r)), 2)
        r[i], r[j] = r[j], r[i]
    return r

# GA loop
def genetic_algorithm_tsp(
    cities: List[City],
    pop_size: int = 150,
    generations: int = 500,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    tournament_k: int = 3,
    elitism: int = 1,
    seed: int | None = None
):
    if seed is not None:
        random.seed(seed)

    n = len(cities)
    # initial population: random permutations
    population: List[Route] = [random.sample(range(n), n) for _ in range(pop_size)]

    best_route = min(population, key=lambda r: route_length(r, cities))
    best_dist = route_length(best_route, cities)
    history = [best_dist]

    for gen in range(generations):
        # sort by fitness descending
        population.sort(key=lambda r: fitness(r, cities), reverse=True)
        # elitism
        new_pop: List[Route] = population[:elitism]

        # produce children
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, cities, k=tournament_k)
            p2 = tournament_selection(population, cities, k=tournament_k)
            if random.random() < crossover_rate:
                c1, c2 = ordered_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            c1 = swap_mutation(c1, mutation_rate)
            c2 = swap_mutation(c2, mutation_rate)
            new_pop.extend([c1, c2])

        population = new_pop[:pop_size]

        # track best
        candidate = min(population, key=lambda r: route_length(r, cities))
        cand_dist = route_length(candidate, cities)
        if cand_dist < best_dist:
            best_dist = cand_dist
            best_route = candidate
        history.append(best_dist)

        # (optional) print progress sparsely
        if (gen+1) % max(1, generations//10) == 0:
            print(f"Gen {gen+1:4d} | best distance = {best_dist:.4f}")

    return best_route, best_dist, history

if __name__ == "__main__":
    # Example coordinates (you can replace with your own)
    cities = [
        (0,0), (1,5), (5,2), (6,6), (8,3),
        (2,1), (7,7), (3,8), (9,1), (4,4),
        (6,1), (2,7), (8,8), (1,9), (9,9)
    ]

    best_route, best_dist, hist = genetic_algorithm_tsp(
        cities,
        pop_size=200,
        generations=600,
        crossover_rate=0.95,
        mutation_rate=0.25,
        tournament_k=4,
        elitism=2,
        seed=42
    )

    print("\nBest tour length:", round(best_dist, 4))
    print("Best route (city indices):", best_route)
