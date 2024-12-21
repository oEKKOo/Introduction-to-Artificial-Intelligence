import numpy as np
import matplotlib.pyplot as plt
import random

def objective_function(x):
    return 10 * np.sin(5 * x) + 7 * np.abs(x - 5) + 10

# 遗传算法参数
POPULATION_SIZE = 50      # 种群大小
GENES_LENGTH = 16         # 基因长度（二进制编码）
X_BOUND = [0, 10]         # x的取值范围
CROSSOVER_PROB = 0.7      # 交叉概率
MUTATION_PROB = 0.001     # 变异概率
MAX_GENERATIONS = 100     # 最大迭代次数

def binary_to_decimal(binary, x_bound):
    decimal = int(binary, 2)
    max_decimal = 2**GENES_LENGTH - 1
    return x_bound[0] + (x_bound[1] - x_bound[0]) * decimal / max_decimal

def decimal_to_binary(x, x_bound):
    max_decimal = 2**GENES_LENGTH - 1
    decimal = int((x - x_bound[0]) * max_decimal / (x_bound[1] - x_bound[0]))
    return format(decimal, f'0{GENES_LENGTH}b')

def initialize_population(pop_size, genes_length):
    population = []
    for _ in range(pop_size):
        individual = ''.join(random.choice(['0', '1']) for _ in range(genes_length))
        population.append(individual)
    return population

def evaluate_fitness(population, x_bound):
    fitness = []
    for individual in population:
        x = binary_to_decimal(individual, x_bound)
        y = objective_function(x)
        fitness.append(1 / (y + 1))
    return fitness

def selection(population, fitness):
    total_fitness = sum(fitness)
    selection_probs = [f / total_fitness for f in fitness]
    selected = np.random.choice(population, size=2, replace=False, p=selection_probs)
    return selected

def crossover(parent1, parent2, crossover_prob):
    if random.random() < crossover_prob:
        point = random.randint(1, GENES_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(individual, mutation_prob):
    mutated = ''
    for gene in individual:
        if random.random() < mutation_prob:
            mutated += '1' if gene == '0' else '0'
        else:
            mutated += gene
    return mutated

def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE, GENES_LENGTH)
    best_fitness_over_time = []
    best_individual = None
    best_y = float('inf')

    for generation in range(MAX_GENERATIONS):
        fitness = evaluate_fitness(population, X_BOUND)
        # 记录当前最优个体
        max_fitness = max(fitness)
        max_index = fitness.index(max_fitness)
        current_best = population[max_index]
        current_y = objective_function(binary_to_decimal(current_best, X_BOUND))
        if current_y < best_y:
            best_y = current_y
            best_individual = current_best

        best_fitness_over_time.append(best_y)

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = selection(population, fitness)
            child1, child2 = crossover(parent1, parent2, CROSSOVER_PROB)
            child1 = mutate(child1, MUTATION_PROB)
            child2 = mutate(child2, MUTATION_PROB)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

    # 解码最佳个体
    best_x = binary_to_decimal(best_individual, X_BOUND)
    return best_x, best_y, best_fitness_over_time

if __name__ == "__main__":
    best_x, best_y, fitness_over_time = genetic_algorithm()
    print(f"最优解x: {best_x:.5f}, y: {best_y:.5f}")

    # 绘制适应度随代数变化图
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_over_time, label='Best y over generations')
    plt.xlabel('Generation')
    plt.ylabel('y value')
    plt.title('遗传算法优化过程')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制目标函数和最优解
    x_values = np.linspace(X_BOUND[0], X_BOUND[1], 400)
    y_values = objective_function(x_values)

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label='y = 10*sin(5x) + 7|x-5| +10')
    plt.plot(best_x, best_y, 'ro', label=f'Best Solution (x={best_x:.5f}, y={best_y:.5f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('目标函数及最优解')
    plt.legend()
    plt.grid(True)
    plt.show()
