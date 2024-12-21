import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm  # 进度条库

# 设置中文字体以确保中文标签正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置随机种子以确保结果可重复
random.seed(42)
np.random.seed(42)

# 生成20个城市的坐标，范围在[0, 100)
City_Map = 100 * np.random.rand(20, 2)

def compute_distance_matrix(city_map):
    """
    预计算城市间的距离矩阵，以提高效率。
    """
    num_cities = len(city_map)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(city_map[i] - city_map[j])
    return distance_matrix

# 预计算距离矩阵
distance_matrix = compute_distance_matrix(City_Map)

def total_distance(path, distance_matrix):
    """
    计算给定路径的总长度。
    """
    distance = 0
    for i in range(len(path)):
        from_city = path[i]
        to_city = path[(i + 1) % len(path)]  # 回到起点
        distance += distance_matrix[from_city][to_city]
    return distance

def initialize_population(pop_size, city_count):
    """
    初始化种群，每个个体是一个城市索引的随机排列。
    """
    population = []
    base_path = list(range(city_count))
    for _ in range(pop_size):
        individual = base_path.copy()
        random.shuffle(individual)
        population.append(individual)
    return population

def evaluate_fitness(population, distance_matrix):
    """
    评估种群中每个个体的适应度。
    适应度定义为路径总长度的倒数。
    """
    fitness = []
    for individual in population:
        dist = total_distance(individual, distance_matrix)
        fitness.append(1 / dist)
    return fitness

def tournament_selection(population, fitness, tournament_size):
    """
    锦标赛选择操作，从种群中选择两个父代。
    """
    selected = []
    for _ in range(2):  # 每次选择两个父代
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament = [ (population[i], fitness[i]) for i in tournament_indices ]
        tournament.sort(key=lambda x: x[1], reverse=True)  # 按适应度降序排序
        selected.append(tournament[0][0])
    return selected

def ordered_crossover(parent1, parent2):
    """
    顺序交叉（Ordered Crossover, OX）操作，生成一个子代。
    """
    size = len(parent1)
    child = [None] * size

    # 随机选择两个交叉点
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size - 1)

    # 复制父代1的交叉段到子代
    child[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]

    # 填充剩余城市，按照父代2的顺序
    p2_pointer = cx_point2
    c_pointer = cx_point2
    while None in child:
        current_city = parent2[p2_pointer % size]
        if current_city not in child:
            child[c_pointer % size] = current_city
            c_pointer += 1
        p2_pointer += 1

    return child

def mutate(individual, mutation_rate):
    """
    交换变异操作，以增加种群多样性。
    """
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(individual) - 1)
            # 交换两个城市的位置
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual

def get_elites(population, fitness, elite_size):
    """
    精英保留策略，保留适应度最高的精英个体。
    """
    fitness_population = list(zip(population, fitness))
    fitness_population.sort(key=lambda x: x[1], reverse=True)
    elites = [individual for individual, fit in fitness_population[:elite_size]]
    return elites

def genetic_algorithm_tsp(pop_size=50, generations=200, elite_size=5, tournament_size=5, mutation_rate=0.02):
    """
    遗传算法主流程，用于解决TSP问题。
    """
    city_count = len(City_Map)
    population = initialize_population(pop_size, city_count)
    fitness = evaluate_fitness(population, distance_matrix)
    elites = get_elites(population, fitness, elite_size)

    best_distance = float('inf')
    best_path = None
    distance_history = []
    average_fitness_history = []
    diversity_history = []

    for generation in tqdm(range(1, generations + 1), desc="迭代进度"):
        # 评估适应度
        fitness = evaluate_fitness(population, distance_matrix)
        
        # 获取精英
        elites = get_elites(population, fitness, elite_size)
        
        # 记录当前最优解
        current_best_fitness = max(fitness)
        current_best_index = fitness.index(current_best_fitness)
        current_best_path = population[current_best_index]
        current_best_distance = total_distance(current_best_path, distance_matrix)
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_path = current_best_path.copy()
        
        distance_history.append(best_distance)
        
        # 记录平均适应度
        average_fitness = sum(fitness) / len(fitness)
        average_fitness_history.append(average_fitness)
        
        # 记录种群多样性（唯一路径数量）
        unique_population = set(tuple(individual) for individual in population)
        diversity_history.append(len(unique_population))
        
        # 创建新种群，首先加入精英
        new_population = elites.copy()
        
        # 生成其余个体
        while len(new_population) < pop_size:
            parent1, parent2 = tournament_selection(population, fitness, tournament_size)
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # 更新种群
        population = new_population[:pop_size]
    
    return best_path, best_distance, distance_history, average_fitness_history, diversity_history

def plot_distance_history(distance_history):
    """
    绘制路径长度随迭代次数变化的图表。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(distance_history, color='blue', linewidth=2)
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('路径总长度', fontsize=14)
    plt.title('遗传算法优化过程中的路径长度变化', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  # 自动调整子图参数以适应图像
    plt.show()

def plot_best_path(best_path):
    """
    绘制最终最优路径的图表。
    """
    plt.figure(figsize=(10, 10))
    # 绘制路径线条
    for i in range(len(best_path)):
        from_city = City_Map[best_path[i]]
        to_city = City_Map[best_path[(i + 1) % len(best_path)]]
        plt.plot([from_city[0], to_city[0]], [from_city[1], to_city[1]], 'b-', linewidth=1)
    
    # 绘制城市点
    plt.scatter(City_Map[:,0], City_Map[:,1], c='red', s=50, zorder=5)
    
    # 标注城市索引
    for idx, (x, y) in enumerate(City_Map):
        plt.text(x + 1, y + 1, str(idx), fontsize=12, color='black')
    
    plt.xlabel('X 坐标', fontsize=14)
    plt.ylabel('Y 坐标', fontsize=14)
    plt.title('遗传算法求解的最优路径', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-5, 105)  # 设置x轴范围，增加边距
    plt.ylim(-5, 105)  # 设置y轴范围，增加边距
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_average_fitness(average_fitness_history):
    """
    绘制种群平均适应度随迭代次数变化的图表。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(average_fitness_history, color='green', linewidth=2)
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('平均适应度', fontsize=14)
    plt.title('遗传算法优化过程中的平均适应度变化', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_population_diversity(diversity_history):
    """
    绘制种群多样性随迭代次数变化的图表。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(diversity_history, color='purple', linewidth=2)
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('种群多样性', fontsize=14)
    plt.title('遗传算法优化过程中的种群多样性变化', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    # 遗传算法参数
    POPULATION_SIZE = 50        # 种群大小
    GENERATIONS = 200           # 最大迭代次数
    ELITE_SIZE = 5              # 精英保留数量
    TOURNAMENT_SIZE = 5         # 锦标赛选择的参与者数量
    MUTATION_RATE = 0.02        # 变异概率

    # 运行遗传算法
    best_path, best_distance, distance_history, average_fitness_history, diversity_history = genetic_algorithm_tsp(
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_size=ELITE_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        mutation_rate=MUTATION_RATE
    )

    # 输出结果
    print("\n最优路径的城市索引顺序：")
    print(best_path)
    print(f"\n最优路径的总长度：{best_distance:.2f}")

    # 可视化结果
    plot_distance_history(distance_history)
    plot_average_fitness(average_fitness_history)
    plot_population_diversity(diversity_history)
    plot_best_path(best_path)

if __name__ == "__main__":
    main()
