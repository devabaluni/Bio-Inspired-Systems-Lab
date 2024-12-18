#Lab-3
import numpy as np

def euclidean_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def create_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = euclidean_distance(cities[i], cities[j])
    return distance_matrix

def ant_colony_optimization(cities, num_ants, alpha, beta, rho, pheromone_init, iterations):
    num_cities = len(cities)
    distance_matrix = create_distance_matrix(cities)
    pheromones = np.full((num_cities, num_cities), pheromone_init)
    best_route = None
    best_distance = float('inf')

    def calculate_transition_probabilities(current_city, visited):
        probabilities = []
        for next_city in range(num_cities):
            if next_city not in visited:
                pheromone = pheromones[current_city][next_city] ** alpha
                heuristic = (1 / distance_matrix[current_city][next_city]) ** beta
                probabilities.append(pheromone * heuristic)
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum()

    for _ in range(iterations):
        all_routes = []
        all_distances = []

        for ant in range(num_ants):
            visited = []
            current_city = np.random.randint(0, num_cities)
            visited.append(current_city)

            for _ in range(num_cities - 1):
                probabilities = calculate_transition_probabilities(current_city, visited)
                next_city = np.random.choice(range(num_cities), p=probabilities)
                visited.append(next_city)
                current_city = next_city

            visited.append(visited[0])
            all_routes.append(visited)
            total_distance = sum(distance_matrix[visited[i]][visited[i + 1]] for i in range(num_cities))
            all_distances.append(total_distance)

            if total_distance < best_distance:
                best_distance = total_distance
                best_route = visited

        pheromones *= (1 - rho)
        for route, distance in zip(all_routes, all_distances):
            for i in range(num_cities):
                pheromones[route[i]][route[i + 1]] += 1 / distance

    return best_route, best_distance

print("1BM22CS092\t\t Dipesh Sah")
print("Lab Experiment-3")
print("Implementation of Traveling Salesman Problem Using Ant Colony Optimization.\n")

num_cities = int(input("Enter the number of cities: "))
cities = []
for i in range(num_cities):
    x, y = map(float, input(f"Enter coordinates of city {i + 1} (x y): ").split())
    cities.append(np.array([x, y]))

num_ants = int(input("Enter the number of ants: "))
alpha = float(input("Enter the importance of pheromone (alpha): "))
beta = float(input("Enter the importance of heuristic information (beta): "))
rho = float(input("Enter the evaporation rate (rho): "))
pheromone_init = float(input("Enter the initial pheromone value: "))
iterations = int(input("Enter the number of iterations: "))

best_route, best_distance = ant_colony_optimization(cities, num_ants, alpha, beta, rho, pheromone_init, iterations)

print("\nBest Route (Order of Cities):", best_route)
print("Best Distance:", best_distance)
