# Lab-6
import numpy as np
import random

def objective_function(cell):
    return sum(cell)  

# Initialize the population
def initialize_population(grid_size, num_resources):
    return np.random.randint(0, 10, size=(grid_size[0], grid_size[1], num_resources))

# Evaluate fitness for all cells in the grid
def evaluate_fitness(grid):
    fitness = np.zeros(grid.shape[:2])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            fitness[i, j] = objective_function(grid[i, j])
    return fitness

# Update the state of a cell based on neighbors and add mutation
def update_cell_state(grid, fitness, x, y, radius, mutation_prob=0.1):
    neighbors = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if 0 <= x + i < grid.shape[0] and 0 <= y + j < grid.shape[1] and (i != 0 or j != 0):
                neighbors.append((x + i, y + j))
    
    # Find the best neighbor based on fitness
    best_neighbor = max(neighbors, key=lambda n: fitness[n[0], n[1]])
    new_state = grid[best_neighbor[0], best_neighbor[1]].copy()
    
    # Add random mutation with a certain probability
    if random.random() < mutation_prob:
        mutation_index = random.randint(0, len(new_state) - 1)
        new_state[mutation_index] = random.randint(0, 10)  # Randomize one resource value
    
    return new_state

# Update grid states
def update_grid(grid, fitness, radius, mutation_prob):
    new_grid = np.copy(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            new_grid[i, j] = update_cell_state(grid, fitness, i, j, radius, mutation_prob)
    return new_grid

# Run the algorithm
def parallel_cellular_algorithm(grid_size, num_resources, num_iterations, radius, mutation_prob):
    grid = initialize_population(grid_size, num_resources)
    best_solution = None
    best_fitness = -np.inf
    
    for iteration in range(num_iterations):
        fitness = evaluate_fitness(grid)
        grid = update_grid(grid, fitness, radius, mutation_prob)
        
        # Track the best solution
        max_fitness = np.max(fitness)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = grid[np.unravel_index(np.argmax(fitness), fitness.shape)]
        
        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")
    
    return best_solution, best_fitness

if __name__ == "__main__":
    print("1BM22CS092\t\t Dipesh Sah")
    print("\tLab Experiment-6")
    print("Implementation of Resource Allocation \nUsing Parallel Cellular Algorithm with Mutation.\n")

    grid_rows = int(input("Enter the number of rows in the grid: "))
    grid_cols = int(input("Enter the number of columns in the grid: "))
    num_resources = int(input("Enter the number of resources per cell: "))
    num_iterations = int(input("Enter the number of iterations: "))
    neighborhood_radius = int(input("Enter the neighborhood radius: "))
    mutation_prob = float(input("Enter the mutation probability (e.g., 0.1 for 10%): "))
    print()
    
    # Execute the algorithm
    best_solution, best_fitness = parallel_cellular_algorithm(
        (grid_rows, grid_cols),
        num_resources,
        num_iterations,
        neighborhood_radius,
        mutation_prob
    )

    print("\nBest Solution Found:", best_solution)
    print("Best Fitness Achieved:", best_fitness)
