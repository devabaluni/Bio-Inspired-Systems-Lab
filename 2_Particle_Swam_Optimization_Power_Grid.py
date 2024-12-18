#Lab-2
import random

# Define the fitness function for power grid optimization
def fitness_function(positions, loads, cost_coefficients, loss_factor=0.1):
    """
    Fitness function to evaluate power grid optimization.

    :param positions: List of generator outputs (particle positions).
    :param loads: List of energy demands at each zone.
    :param cost_coefficients: Coefficients for generator cost [a, b, c] for quadratic cost.
    :param loss_factor: Proportional factor for energy loss.
    :return: Total cost (fitness value).
    """
    total_power = sum(positions)
    total_load = sum(loads)
    
    loss = loss_factor * (total_power - total_load) ** 2

    balance_penalty = abs(total_power - total_load) * 100

    generation_cost = sum(
        cost_coefficients[i][0] * positions[i]**2 + cost_coefficients[i][1] * positions[i] + cost_coefficients[i][2]
        for i in range(len(positions))
    )

    return generation_cost + balance_penalty + loss


# PSO Algorithm for Power Grid Optimization
def pso(num_particles, num_dimensions, bounds, max_iterations, loads, cost_coefficients):
    # PSO Parameters
    w_max = 0.9  # Initial inertia weight
    w_min = 0.4  # Final inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    max_velocity = (bounds[1] - bounds[0]) / 2  

    particles = [{'position': [random.uniform(bounds[0], bounds[1]) for _ in range(num_dimensions)],
                  'velocity': [random.uniform(-1, 1) for _ in range(num_dimensions)],
                  'best_position': None,
                  'best_fitness': float('inf')} for _ in range(num_particles)]

    global_best_position = None
    global_best_fitness = float('inf')

    for iteration in range(max_iterations):
        w = w_max - (w_max - w_min) * (iteration / max_iterations)

        for particle in particles:
            current_fitness = fitness_function(particle['position'], loads, cost_coefficients)

            if current_fitness < particle['best_fitness']:
                particle['best_fitness'] = current_fitness
                particle['best_position'] = particle['position'][:]

            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle['position'][:]
        
        for particle in particles:
            for d in range(num_dimensions):
                r1 = random.random()  
                r2 = random.random()  
                
                cognitive_velocity = c1 * r1 * (particle['best_position'][d] - particle['position'][d])
                social_velocity = c2 * r2 * (global_best_position[d] - particle['position'][d])
                particle['velocity'][d] = w * particle['velocity'][d] + cognitive_velocity + social_velocity

                particle['velocity'][d] = max(min(particle['velocity'][d], max_velocity), -max_velocity)

                particle['position'][d] += particle['velocity'][d]

                if particle['position'][d] < bounds[0]:
                    particle['position'][d] = bounds[0]
                elif particle['position'][d] > bounds[1]:
                    particle['position'][d] = bounds[1]

    return global_best_position, global_best_fitness


if __name__ == "__main__":
    
    print("1BM22CS092\t\t Dipesh Sah")
    print("Lab Experiment-2")
    print("Implementation of Power Grid Optimization to minimize power \nloss & balance demand supply using Particle Swam Optimization.\n")
    
    num_generators = int(input("Enter the number of generators: "))
    loads = list(map(float, input("Enter the energy demands (space-separated): ").split()))
    bounds = [float(input("Enter the lower bound of generator output: ")),
              float(input("Enter the upper bound of generator output: "))]
    max_iterations = int(input("Enter the maximum number of iterations: "))
    num_particles = int(input("Enter the number of particles: "))

    cost_coefficients = []
    for i in range(num_generators):
        print(f"Enter cost coefficients (a, b, c) for Generator {i + 1}:")
        cost_coefficients.append(list(map(float, input().split())))

    best_position, best_fitness = pso(num_particles, num_generators, bounds, max_iterations, loads, cost_coefficients)
    
    print("Optimal Generator Outputs:", best_position)
    print("Minimum Total Cost:", best_fitness)
