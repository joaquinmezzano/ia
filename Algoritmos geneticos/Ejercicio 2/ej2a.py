import random

# -------------------------------
# PARÁMETROS
# -------------------------------
POP_SIZE = 20       # Tamaño población
GENS = 30           # Numero de generaciones
CXPB = 0.8          # % de cruce
MUTPB = 0.05        # % de mutación
GENOME_LENGTH = 6  # Longitud de cada individuo (en bits)

# -------------------------------
# FUNCIÓN FITNESS
# -------------------------------
def fitness(individual):
    result = 0
    for index, value in enumerate(reversed(individual)):
        if (value == 1):
            result += index+1**2
    return abs(result-num)

# -------------------------------
# CREAR POBLACIÓN
# -------------------------------
def create_individual():
    return [random.randint(0, 1) for _ in range(GENOME_LENGTH)]

def create_population():
    return [create_individual() for _ in range(POP_SIZE)]

# -------------------------------
# OPERADORES GENÉTICOS
# -------------------------------
def selection(population):
    # Selección por torneo
    k = 3
    selected = []
    for _ in range(POP_SIZE):
        aspirants = random.sample(population, k)
        winner = max(aspirants, key=fitness)
        selected.append(winner)
    return selected

def crossover(p1, p2):
    # Cross de un punto
    if random.random() < CXPB:
        point = random.randint(1, GENOME_LENGTH-1)
        return p1[:point]+p2[point:], p2[:point]+p1[point:]
    return p1[:], p2[:]

def mutate(individual):
    # Mutación switcheando bit
    for i in range(GENOME_LENGTH):
        if random.random() < MUTPB:
            individual[i] = 1 - individual[i]
    return individual

# -------------------------------
# ALGORITMO PRINCIPAL
# -------------------------------
def genetic_algorithm():
    population = create_population()
    for gen in range(GENS):
        # Evaluar y mostrar el mejor
        population.sort(key=fitness, reverse=True)
        print(f"Gen {gen}: Mejor = {population[0]} Fitness = {fitness(population[0])}")

        # Selección
        selected = selection(population)

        # Reproducción
        next_gen = []
        for i in range(0, POP_SIZE, 2):
            offspring1, offspring2 = crossover(selected[i], selected[i+1])
            next_gen.append(mutate(offspring1))
            next_gen.append(mutate(offspring2))
        
        population = next_gen
    return max(population, key=fitness)

# -------------------------------
# EJECUTAR
# -------------------------------
num = int(input("Indique el número a comprobar: "))
best = genetic_algorithm()
print(f"Mejor individuo encontrado: {best}, Fitness = {fitness(best)}")