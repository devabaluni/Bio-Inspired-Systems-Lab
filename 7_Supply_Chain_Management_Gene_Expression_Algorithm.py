#Lab-7

import random

# Fitness function: calculates the total cost
def evaluate(individual, demands, holding_cost, order_cost, transport_cost):
    """
    Evaluate the total supply chain cost.
    individual: List of order quantities for each warehouse.
    """
    total_holding_cost = 0
    total_order_cost = len(demands) * order_cost
    total_transport_cost = 0
    penalty = 0  

    # costs
    for i in range(len(demands)):
        total_holding_cost += individual[i] * holding_cost
        total_transport_cost += individual[i] * transport_cost

        # penalty if the order quantity does not meet demand
        if individual[i] < demands[i]:
            penalty += 1000  # Heavy penalty for unmet demand

    total_cost = total_holding_cost + total_order_cost + total_transport_cost + penalty
    return total_cost

def initialize_population(pop_size, num_warehouses, max_order):
    return [[random.randint(0, max_order) for _ in range(num_warehouses)] for _ in range(pop_size)]

# Selection: Tournament selection
def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])  # Select individual with lowest cost
        selected.append(winner[0])
    return selected

# Crossover: Uniform crossover
def crossover(parent1, parent2, crossover_prob=0.5):
    child1, child2 = parent1[:], parent2[:]
    if random.random() < crossover_prob:
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = parent2[i], parent1[i]
    return child1, child2

# Mutation: Random mutation
def mutate(individual, max_order, mutation_prob=0.2):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = random.randint(0, max_order)
    return individual

# Main Gene Expression Algorithm
def gene_expression_algorithm():
    random.seed(42)  
    
    print("1BM22CS092\t\t Dipesh Sah")
    print("Lab Experiment-7")
    print("Implementation of Supply Chain Management \nUsing Gene Expression Algorithm.\n")
    
    num_warehouses = int(input("Enter the number of warehouses: "))
    max_order = int(input("Enter the maximum order quantity per warehouse: "))
    holding_cost = float(input("Enter the holding cost per unit: "))
    order_cost = float(input("Enter the fixed order cost per warehouse: "))
    transport_cost = float(input("Enter the transportation cost per unit: "))

    demands = []
    print("\nEnter the demand for each warehouse:")
    for i in range(num_warehouses):
        demand = int(input(f"Warehouse {i+1} demand: "))
        demands.append(demand)

    # Algorithm Parameters
    population_size = 50
    num_generations = 20
    crossover_prob = 0.8
    mutation_prob = 0.2

    population = initialize_population(population_size, num_warehouses, max_order)

    print("\nRunning Gene Expression Algorithm...")
    for generation in range(num_generations):
        fitnesses = [evaluate(ind, demands, holding_cost, order_cost, transport_cost) for ind in population]

        selected_population = selection(population, fitnesses)

        next_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2, crossover_prob)
            next_population.append(mutate(child1, max_order, mutation_prob))
            next_population.append(mutate(child2, max_order, mutation_prob))
        
        population = next_population

        best_cost = min(fitnesses)
        print(f"Generation {generation+1}, Best Cost: {best_cost}")

    final_fitnesses = [evaluate(ind, demands, holding_cost, order_cost, transport_cost) for ind in population]
    best_solution = population[final_fitnesses.index(min(final_fitnesses))]
    print("\nBest Solution Found:")
    for i, qty in enumerate(best_solution):
        print(f"Warehouse {i+1} Order Quantity: {qty}")
    print(f"Total Cost: {min(final_fitnesses)}")

if __name__ == "__main__":
    gene_expression_algorithm()
