import numpy as np
import matplotlib.pyplot as plt
import time
import random

# ---------------- Parameters ----------------
GRID_SIZE = 30
INFECTION_PROB = 0.35     # Probability of infection from neighbor
RECOVERY_TIME = 6         # Steps before recovery
ITERATIONS = 40           # Number of simulation steps
VACCINATED_RATIO = 0.1    # 10% vaccinated
QUARANTINED_RATIO = 0.1   # 10% quarantined

# States: 0=Healthy, 1=Infected, 2=Recovered, 3=Vaccinated, 4=Quarantined

# ---------------- Initialization ----------------
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
infection_timer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Infect one person in the center
grid[GRID_SIZE//2, GRID_SIZE//2] = 1

# Random vaccination and quarantine
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        r = random.random()
        if r < VACCINATED_RATIO:
            grid[x][y] = 3
        elif r < VACCINATED_RATIO + QUARANTINED_RATIO:
            grid[x][y] = 4

# ---------------- Neighbor Helper ----------------
def get_neighbors(x, y):
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

# ---------------- Visualization Setup ----------------
plt.figure(figsize=(6,6))
cmap = plt.cm.get_cmap('viridis', 5)  # 5 discrete colors for 5 states
plt.ion()  # interactive mode

# ---------------- Simulation ----------------
for step in range(ITERATIONS):
    plt.clf()
    plt.title(f"Disease Spread Simulation (Step {step+1})")
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    cbar = plt.colorbar(ticks=[0,1,2,3,4])
    cbar.ax.set_yticklabels(['Healthy','Infected','Recovered','Vaccinated','Quarantined'])
    plt.pause(0.3)

    new_grid = grid.copy()

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            state = grid[x, y]

            if state == 0:  # Healthy
                for nx, ny in get_neighbors(x, y):
                    if grid[nx, ny] == 1 and np.random.rand() < INFECTION_PROB:
                        new_grid[x, y] = 1
                        break

            elif state == 1:  # Infected
                infection_timer[x, y] += 1
                if infection_timer[x, y] >= RECOVERY_TIME:
                    new_grid[x, y] = 2  # Recovered

            # Vaccinated (3) & Quarantined (4) remain unchanged

    grid = new_grid.copy()

plt.ioff()
plt.show()
print("\nSimulation complete ✅")
