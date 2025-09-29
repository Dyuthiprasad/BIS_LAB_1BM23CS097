import numpy as np

# Objective function (Sphere)
def objective_function(x):
    return np.sum(x**2)

def grey_wolf_optimizer(obj_func, dim, bounds, n_wolves=20, max_iter=100):
    # Initialize wolf positions
    wolves = np.random.uniform(bounds[0], bounds[1], (n_wolves, dim))
    
    # Initialize alpha, beta, delta
    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")
    
    for t in range(max_iter):
        for i in range(n_wolves):
            # Ensure within bounds
            wolves[i] = np.clip(wolves[i], bounds[0], bounds[1])
            
            # Calculate fitness
            fitness = obj_func(wolves[i])
            
            # Update alpha, beta, delta
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, wolves[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, wolves[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, wolves[i].copy()
        
        # Update wolves
        a = 2 - 2 * (t / max_iter)  # Decreases from 2 to 0
        for i in range(n_wolves):
            for leader_pos in [alpha_pos, beta_pos, delta_pos]:
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = abs(C * leader_pos - wolves[i])
                X = leader_pos - A * D
                wolves[i] = (wolves[i] + X) / 2  # average move towards leaders
    
    return alpha_pos, alpha_score

# Test the GWO
best_pos, best_score = grey_wolf_optimizer(objective_function, dim=5, bounds=(-10, 10), n_wolves=20, max_iter=50)

print("Best solution found:", best_pos)
print("Best fitness value:", best_score)
