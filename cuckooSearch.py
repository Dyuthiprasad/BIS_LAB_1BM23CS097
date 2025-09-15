import numpy as np
import math

# Objective function example: Sphere function
def objective(x):
    return np.sum(x**2)

# Generate a random solution within bounds
def random_solution(bounds, dim):
    return np.random.uniform(bounds[0], bounds[1], dim)

# Ensure solution stays within search space
def ensure_bounds(x, bounds):
    return np.clip(x, bounds[0], bounds[1])

# Lévy flight using Mantegna's algorithm
def levy_flight(dim, alpha=0.01, beta=1.5):
    sigma_u = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
              (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v)**(1/beta))
    return alpha * step

# Cuckoo Search main function
def cuckoo_search(objective, dim, bounds=(-5,5), n_nests=25, pa=0.25,
                  alpha=0.01, max_iter=500):

    # 1. initialize nests
    nests = [random_solution(bounds, dim) for _ in range(n_nests)]
    fitness = np.array([objective(x) for x in nests])
    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()

    for _ in range(max_iter):
        # 2. generate new solutions with levy flights
        for i in range(n_nests):
            new = nests[i] + levy_flight(dim, alpha) * (nests[i] - best)
            new = ensure_bounds(new, bounds)
            f_new = objective(new)

            # 3. greedy selection
            if f_new < fitness[i]:
                nests[i], fitness[i] = new, f_new
                if f_new < fitness[best_idx]:
                    best_idx = i
                    best = new.copy()

        # 4. abandon a fraction of worst nests
        K = int(pa * n_nests)
        worst_indices = np.argsort(fitness)[-K:]
        for idx in worst_indices:
            nests[idx] = random_solution(bounds, dim)
            fitness[idx] = objective(nests[idx])
            if fitness[idx] < fitness[best_idx]:
                best_idx = idx
                best = nests[idx].copy()

    return best, fitness[best_idx]


# ================== RUN TEST ==================
if __name__ == "__main__":
    dim = 5
    best_sol, best_fit = cuckoo_search(objective, dim, bounds=(-5,5),
                                       n_nests=30, max_iter=500)
    print("Best solution:", best_sol)
    print("Best fitness:", best_fit)
