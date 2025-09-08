import numpy as np
import random

# Step 1: Define the problem (cities and coordinates)
cities = {
    0: (0, 0),
    1: (1, 5),
    2: (5, 2),
    3: (6, 6),
    4: (8, 3),
    5: (7, 9)
}

num_cities = len(cities)

# Distance matrix
dist_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            dist_matrix[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
        else:
            dist_matrix[i][j] = np.inf  # no self loops

# Step 2: Initialize Parameters
num_ants = 10
alpha = 1.0     # pheromone importance
beta = 5.0      # distance heuristic importance
rho = 0.5       # evaporation rate
Q = 100         # pheromone deposit factor
iterations = 100

pheromone = np.ones((num_cities, num_cities))  # initial pheromone levels

# Step 3: Solution construction
def construct_solution(start_city):
    tour = [start_city]
    unvisited = set(range(num_cities)) - {start_city}
    
    while unvisited:
        current_city = tour[-1]
        probabilities = []
        for next_city in unvisited:
            pher = pheromone[current_city][next_city] ** alpha
            heuristic = (1.0 / dist_matrix[current_city][next_city]) ** beta
            probabilities.append(pher * heuristic)
        
        probabilities = probabilities / np.sum(probabilities)
        next_city = random.choices(list(unvisited), weights=probabilities, k=1)[0]
        tour.append(next_city)
        unvisited.remove(next_city)
    
    return tour

# Calculate tour length
def tour_length(tour):
    length = 0
    for i in range(len(tour) - 1):
        length += dist_matrix[tour[i]][tour[i+1]]
    length += dist_matrix[tour[-1]][tour[0]]  # return to start
    return length

# Step 4: Update pheromones
def update_pheromones(all_tours):
    global pheromone
    pheromone *= (1 - rho)  # evaporation
    for tour in all_tours:
        length = tour_length(tour)
        for i in range(len(tour) - 1):
            pheromone[tour[i]][tour[i+1]] += Q / length
        pheromone[tour[-1]][tour[0]] += Q / length  # close the tour

# Step 5: Iterate
best_tour = None
best_length = float("inf")

for it in range(iterations):
    all_tours = []
    for ant in range(num_ants):
        start_city = random.randint(0, num_cities-1)
        tour = construct_solution(start_city)
        all_tours.append(tour)
        length = tour_length(tour)
        if length < best_length:
            best_length = length
            best_tour = tour
    
    update_pheromones(all_tours)
    print(f"Iteration {it+1}: Best length = {best_length:.2f}")

# Step 6: Output Best Solution
print("\nBest tour found:", best_tour)
print("Tour length:", best_length)
