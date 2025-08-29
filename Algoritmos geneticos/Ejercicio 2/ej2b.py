import random
from deap import base, creator, tools, algorithms

# Crear clase de fitness y clase de individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox: atributos, individuos, población
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de evaluación: contar 1's
def eval_maxones(individual):
    result = 0
    for index, value in enumerate(reversed(individual)):
        if (value == 1):
            result += index+1**2
    return (-abs(result-num),)
toolbox.register("evaluate", eval_maxones)

# Operadores genéticos
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Algoritmo principal
num = int(input("Indique el número a comprobar: "))
def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

    best = tools.selBest(pop, 1)[0]
    print("Mejor individuo: ", best)
    print("Fitness: ", best.fitness.values[0])

if __name__ == "__main__":
    main()