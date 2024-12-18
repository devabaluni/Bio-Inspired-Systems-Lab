#Lab-4
import numpy as np

def fitness_function(path, obstacles, target):
    path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    collision_penalty = 0

    for point in path:
        for obs in obstacles:
            obs_x, obs_y, radius = obs
            distance_to_obs = np.linalg.norm(point - np.array([obs_x, obs_y]))
            if distance_to_obs < radius:
                collision_penalty += 1e6

    distance_to_target = np.linalg.norm(path[-1] - target)
    return path_length + collision_penalty + distance_to_target

def levy_flight(dim):
    beta = 1.5
    u = np.random.normal(0, 1, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step

def cuckoo_search_robot(num_nests, max_iter, waypoints, lower_bound, upper_bound, start, target, obstacles):
    dim = waypoints * 2
    nests = np.random.uniform(lower_bound, upper_bound, (num_nests, dim))
    nests = nests.reshape((num_nests, waypoints, 2))
    fitness = np.array([fitness_function(np.vstack([start, nest, target]), obstacles, target) for nest in nests])

    best_nest = nests[np.argmin(fitness)]
    best_fitness = min(fitness)

    for _ in range(max_iter):
        for i in range(num_nests):
            new_nest = nests[i] + levy_flight(dim).reshape(waypoints, 2)
            new_nest = np.clip(new_nest, lower_bound, upper_bound)
            new_fitness = fitness_function(np.vstack([start, new_nest, target]), obstacles, target)
            
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_nest = nests[current_best_idx]

        abandon_prob = 0.25
        worst_nests_idx = np.argsort(fitness)[-int(abandon_prob * num_nests):]
        for idx in worst_nests_idx:
            nests[idx] = np.random.uniform(lower_bound, upper_bound, (waypoints, 2))
            fitness[idx] = fitness_function(np.vstack([start, nests[idx], target]), obstacles, target)

    return best_nest, best_fitness

print("1BM22CS092\t\t Dipesh Sah")
print("Lab Experiment-4")
print("Implementation of Robot Motion Planning for finding \nthe most efficient paths Using Cuckoo Search.\n")

num_nests = int(input("Enter the number of nests: "))
max_iter = int(input("Enter the number of iterations: "))
waypoints = int(input("Enter the number of waypoints: "))
lower_bound = float(input("Enter the lower bound of the search space: "))
upper_bound = float(input("Enter the upper bound of the search space: "))
start_x, start_y = map(float, input("Enter the start position (x y): ").split())
target_x, target_y = map(float, input("Enter the target position (x y): ").split())
num_obstacles = int(input("Enter the number of obstacles: "))
obstacles = []

for i in range(num_obstacles):
    obs_x, obs_y, radius = map(float, input(f"Enter obstacle {i+1} (x y radius): ").split())
    obstacles.append((obs_x, obs_y, radius))

start = np.array([start_x, start_y])
target = np.array([target_x, target_y])

best_path, best_value = cuckoo_search_robot(num_nests, max_iter, waypoints, lower_bound, upper_bound, start, target, obstacles)

print("\nBest Path (Waypoints):")
print(np.vstack([start, best_path, target]))
print("Best Fitness Value:", best_value)
