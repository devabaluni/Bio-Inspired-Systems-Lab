#Lab-1
import random

def fitness(path, distance_matrix):
    cost = 0
    for i in range(len(path) - 1):
        cost += distance_matrix[path[i]][path[i + 1]]
    return cost

def random_route(num_locations):
    route = list(range(1, num_locations))
    random.shuffle(route)
    return [0] + route + [0]

def crossover(parent1, parent2):
    split = random.randint(1, len(parent1) - 2)
    child = parent1[:split] + [node for node in parent2 if node not in parent1[:split]]
    return child

def mutate(route):
    if random.random() < MUTATION_RATE:
        index1 = random.randint(1, len(route) - 2)
        index2 = random.randint(1, len(route) - 2)
        route[index1], route[index2] = route[index2], route[index1]
    return route

def initialize_population(num_locations):
    return [random_route(num_locations) for _ in range(POPULATION_SIZE)]

def genetic_algorithm(distance_matrix, num_locations):
    population = initialize_population(num_locations)
    
    for generation in range(GENERATIONS):
        population = sorted(population, key=lambda x: fitness(x, distance_matrix))
        print(f"Generation {generation}: Best route {population[0]} with cost {fitness(population[0], distance_matrix)}")
        
        top_half = population[:len(population) // 2]
        new_population = top_half[:]
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(top_half, 2)
            child = crossover(parent1, parent2)
            new_population.append(mutate(child))
        
        population = new_population

    best_route = min(population, key=lambda x: fitness(x, distance_matrix))
    return best_route, fitness(best_route, distance_matrix)

POPULATION_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.3

print("1BM22CS092\t\t Dipesh Sah")
print("Lab Experiment-1")
print("Implementation of Vehicle Routing Using Genetic Algorithms.\n")
num_locations = int(input("Enter the number of locations (including depot): "))
distance_matrix = []

print("Enter the distance matrix:")
for i in range(num_locations):
    row = list(map(int, input(f"Enter distances from location {i}: ").split()))
    distance_matrix.append(row)

start_location = 0
best_route, best_cost = genetic_algorithm(distance_matrix, num_locations)
print(f"\nBest route found: {best_route} with cost: {best_cost}")
