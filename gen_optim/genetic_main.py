import genetic_utils as gu
import numpy as np

num_generations  = 10
population_size  = 40

epsilon_range = (0.1,  0.45)
kappa_range   = (1.0,  1.7)
delta_range   = (-0.3, 0.3)

population = gu.initialize_population(population_size,
                                      epsilon_range, kappa_range, delta_range)

best_params  = np.zeros(3)
best_fitness = -np.inf

for gen in range(num_generations):
    fitness = gu.evaluate_population(population, A=-0.5, method='contour', N=200)

    # track global best
    gen_best_idx = np.argmax(fitness)
    if fitness[gen_best_idx] > best_fitness:
        best_fitness = fitness[gen_best_idx]
        best_params  = population[gen_best_idx].copy()

    print(f"gen {gen+1:3d}/{num_generations}  "
          f"best={best_fitness:.6f}  "
          f"mean={fitness.mean():.6f}  "
          f"(eps={best_params[0]:.3f}, kap={best_params[1]:.3f}, dlt={best_params[2]:.3f})")

    parents_a, parents_b = gu.select_parents(population, fitness, k=10)

    offspring = gu.crossover(parents_a, parents_b, method='blx_alpha', alpha=0.05)

    mutated_offspring = gu.mutate(offspring, 0.05,
                                  epsilon_range, kappa_range, delta_range)

    # elitism: carry the top 1% of the current population into the next generation
    n_elite = max(1, int(0.01 * population_size))
    elite_idx = np.argsort(fitness)[-n_elite:]
    mutated_offspring[:n_elite] = population[elite_idx]
    population = mutated_offspring

print()
print("═" * 55)
print("Best individual found:")
print(f"  epsilon : {best_params[0]:.6f}")
print(f"  kappa   : {best_params[1]:.6f}")
print(f"  delta   : {best_params[2]:.6f}")
print(f"  fitness : {best_fitness:.6f}")
print("═" * 55)







