import numpy as np
import matplotlib.pyplot as plt

# --- Customer locations (x, y coordinates) ---
customers = np.array([
    [2, 3], [5, 8], [1, 9], [8, 2], [7, 6],
    [3, 7], [6, 1], [9, 9], [4, 5], [10, 3]
])

# --- Fitness function ---
def logistics_cost(warehouse):
    # warehouse = [x, y]
    dists = np.linalg.norm(customers - warehouse, axis=1)  # distances to all customers
    transport_cost = np.sum(dists)  # total transport cost
    penalty = np.max(dists) * 2     # penalty for farthest customer
    return transport_cost + penalty

# --- PSO for Warehouse Placement ---
class PSO:
    def __init__(self, func, dim=2, bounds=(0, 12),
                 num_particles=30, iterations=100,
                 w=0.7, c1=1.5, c2=1.5, seed=None):
        np.random.seed(seed)
        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.w, self.c1, self.c2 = w, c1, c2

        # init positions & velocities
        self.pos = np.random.uniform(self.lb, self.ub, (num_particles, dim))
        self.vel = np.random.uniform(-1, 1, (num_particles, dim))

        # personal best
        self.pbest = self.pos.copy()
        self.pbest_val = np.array([func(p) for p in self.pos])

        # global best
        best_idx = np.argmin(self.pbest_val)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_val = self.pbest_val[best_idx]

    def optimize(self, verbose=False):
        history = []
        for it in range(self.iterations):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive = self.c1 * r1 * (self.pbest - self.pos)
            social = self.c2 * r2 * (self.gbest - self.pos)

            self.vel = self.w * self.vel + cognitive + social
            self.pos += self.vel
            self.pos = np.clip(self.pos, self.lb, self.ub)

            vals = np.array([self.func(p) for p in self.pos])
            improved = vals < self.pbest_val
            self.pbest[improved] = self.pos[improved]
            self.pbest_val[improved] = vals[improved]

            best_idx = np.argmin(self.pbest_val)
            if self.pbest_val[best_idx] < self.gbest_val:
                self.gbest = self.pbest[best_idx].copy()
                self.gbest_val = self.pbest_val[best_idx]

            history.append(self.gbest_val)
            if verbose and (it+1) % 10 == 0:
                print(f"Iter {it+1}: Best Cost = {self.gbest_val:.2f}")

        return self.gbest, self.gbest_val, history

# --- Run PSO ---
pso = PSO(func=logistics_cost, iterations=100, num_particles=40, seed=42)
best_loc, best_cost, hist = pso.optimize(verbose=True)

print("\nOptimal Warehouse Location:", best_loc)
print("Minimum Logistics Cost:", best_cost)

# --- Visualization ---
plt.figure(figsize=(6,6))
plt.scatter(customers[:,0], customers[:,1], c="blue", label="Customers")
plt.scatter(best_loc[0], best_loc[1], c="red", marker="*", s=200, label="Optimal Warehouse")
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.grid(True)
plt.legend()
plt.title("Supply Chain Optimization with PSO")
plt.show()

# --- Convergence Plot ---
plt.plot(hist)
plt.xlabel("Iteration")
plt.ylabel("Best Cost")
plt.title("PSO Convergence - Logistics Cost")
plt.grid(True)
plt.show()
