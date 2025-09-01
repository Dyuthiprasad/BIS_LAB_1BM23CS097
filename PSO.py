"""
Simple PSO (Particle Swarm Optimization) implementation in Python (minimization).
Depends only on numpy.

Core parameters:
- w  : inertia weight (exploration vs exploitation)
- c1 : cognitive coefficient (particle's own experience)
- c2 : social coefficient (swarm's experience)
"""

import numpy as np

class PSO:
    def __init__(self, func, dim, bounds, swarmsize=30, iterations=100,
                 w=0.7, c1=1.5, c2=1.5, vel_clamp=None, seed=None):
        """
        func       : callable(x) -> scalar (objective to minimize). x is 1D numpy array length dim.
        dim        : dimensionality of the search space (int)
        bounds     : tuple (lower, upper) or array-like shape (dim,2)
        swarmsize  : number of particles
        iterations : number of iterations (generations)
        w, c1, c2  : PSO hyperparameters
        vel_clamp  : None or (vmin, vmax) to clamp velocities
        seed       : random seed for reproducibility
        """
        self.func = func
        self.dim = dim
        self.swarmsize = swarmsize
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vel_clamp = vel_clamp
        if seed is not None:
            np.random.seed(seed)

        # handle bounds
        if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0]):
            lo = np.full(dim, bounds[0], dtype=float)
            hi = np.full(dim, bounds[1], dtype=float)
        else:
            arr = np.array(bounds, dtype=float)
            if arr.shape == (dim, 2):
                lo, hi = arr[:, 0], arr[:, 1]
            else:
                raise ValueError("bounds must be (low, high) scalars or array-like shape (dim,2)")
        self.lb = lo
        self.ub = hi

        # initialize particles
        self.pos = np.random.uniform(self.lb, self.ub, (swarmsize, dim))
        # small random initial velocities
        self.vel = np.random.uniform(-np.abs(self.ub - self.lb), np.abs(self.ub - self.lb), (swarmsize, dim)) * 0.1

        # personal best positions and values
        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.array([self.func(p) for p in self.pos])

        # global best
        best_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_val = self.pbest_val[best_idx]

        # history (optional)
        self.history = []

    def step(self):
        r1 = np.random.rand(self.swarmsize, self.dim)
        r2 = np.random.rand(self.swarmsize, self.dim)

        cognitive = self.c1 * r1 * (self.pbest_pos - self.pos)
        social = self.c2 * r2 * (self.gbest_pos - self.pos)

        # velocity update
        self.vel = self.w * self.vel + cognitive + social

        # velocity clamping
        if self.vel_clamp is not None:
            vmin, vmax = self.vel_clamp
            self.vel = np.clip(self.vel, vmin, vmax)

        # position update
        self.pos = self.pos + self.vel

        # boundary handling (reflective)
        out_low = self.pos < self.lb
        out_high = self.pos > self.ub
        if np.any(out_low) or np.any(out_high):
            # reflect position inside and invert velocity component
            self.pos = np.where(out_low, self.lb + (self.lb - self.pos), self.pos)
            self.pos = np.where(out_high, self.ub - (self.pos - self.ub), self.pos)
            self.vel = np.where(out_low | out_high, -self.vel, self.vel)

        # evaluate
        vals = np.array([self.func(p) for p in self.pos])

        # update personal bests
        improved = vals < self.pbest_val
        if np.any(improved):
            self.pbest_pos[improved] = self.pos[improved]
            self.pbest_val[improved] = vals[improved]

        # update global best
        best_idx = np.argmin(self.pbest_val)
        if self.pbest_val[best_idx] < self.gbest_val:
            self.gbest_val = self.pbest_val[best_idx]
            self.gbest_pos = self.pbest_pos[best_idx].copy()

        return self.gbest_pos, self.gbest_val

    def optimize(self, verbose=False):
        for it in range(self.iterations):
            gpos, gval = self.step()
            self.history.append(gval)
            if verbose and ((it + 1) % max(1, self.iterations // 10) == 0 or it == 0):
                print(f"Iter {it+1:4d}/{self.iterations}  best = {gval:.6g}")
        return self.gbest_pos, self.gbest_val

# ---------- Example objective functions ----------
def sphere(x):
    # simple convex function, global min at 0
    return np.sum(x**2)

def rastrigin(x):
    # multimodal benchmark (min at 0)
    A = 10.0
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# ---------- Example usage ----------
if __name__ == "__main__":
    dim = 5
    bounds = (-5.12, 5.12)   # common Rastrigin bounds

    # choose the objective:
    objective = rastrigin   # or sphere

    pso = PSO(func=objective,
              dim=dim,
              bounds=bounds,
              swarmsize=50,
              iterations=300,
              w=0.729,    # often used value (constriction factor style)
              c1=1.49445, # common PSO tuning constants
              c2=1.49445,
              vel_clamp=(-2.5, 2.5),
              seed=42)

    best_pos, best_val = pso.optimize(verbose=True)

    print("\nSolution found:")
    print("best position:", best_pos)
    print("best value   :", best_val)

    # optional: inspect history (list of best values per iteration)
    try:
        import matplotlib.pyplot as plt
        plt.plot(pso.history)
        plt.xlabel("Iteration")
        plt.ylabel("Best objective value")
        plt.title("PSO Convergence")
        plt.grid(True)
        plt.show()
    except Exception:
        # matplotlib may not be available — that's fine.
        pass
