import numpy as np

np.random.seed(42)
X = np.random.rand(100, 2)
true_weights = np.array([3, 5])
y = X @ true_weights + 2 + np.random.randn(100) * 0.1  


num_ants = 30
num_iterations = 50
num_features = X.shape[1] + 1  
pheromone_init = 1.0
evaporation_rate = 0.3
alpha = 1
beta = 2
search_range = (-10, 10)  


num_values = 50  
pheromones = np.ones((num_features, num_values)) * pheromone_init
weight_values = np.linspace(search_range[0], search_range[1], num_values)

def evaluate_solution(weights):
    
    y_pred = X @ weights[:-1] + weights[-1]
    mse = np.mean((y - y_pred) ** 2)
    return mse

def choose_weight(feature_idx):
    
    probs = pheromones[feature_idx] ** alpha
    probs /= probs.sum()
    value = np.random.choice(weight_values, p=probs)
    idx = np.where(weight_values == value)[0][0]
    return value, idx

best_weights = None
best_error = float('inf')


for iteration in range(num_iterations):
    all_solutions = []
    all_errors = []
    all_indices = []

    
    for ant in range(num_ants):
        weights = []
        indices = []
        for f in range(num_features):
            value, idx = choose_weight(f)
            weights.append(value)
            indices.append(idx)
        mse = evaluate_solution(np.array(weights))
        all_solutions.append(weights)
        all_errors.append(mse)
        all_indices.append(indices)

    
    pheromones *= (1 - evaporation_rate)
    for i, indices in enumerate(all_indices):
        reward = 1 / (all_errors[i] + 1e-6)
        for f in range(num_features):
            pheromones[f, indices[f]] += reward

    iteration_best = np.argmin(all_errors)
    iteration_best_weights = all_solutions[iteration_best]
    iteration_best_error = all_errors[iteration_best]

    if iteration_best_error < best_error:
        best_error = iteration_best_error
        best_weights = iteration_best_weights

    print(f"Iteration {iteration+1}/{num_iterations} | Best MSE: {best_error:.6f}")

print("\nBest weights found:")
print(f"Weights: {best_weights[:-1]}, Bias: {best_weights[-1]}")
print(f"True weights: {true_weights}, True bias: 2")
print(f"Final MSE: {best_error:.6f}")
