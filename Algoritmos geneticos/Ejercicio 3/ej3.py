import random
from deap import base, creator, tools, algorithms

# Crear clase de fitness y clase de individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox: atributos, individuos, población
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 255)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de evaluación
def eval_maxones(individual):
    r,g,b = individual
    result = ((r-tripla[0])**2+(g-tripla[1])**2+(b-tripla[2])**2) ** 0.5
    return (-result,)
toolbox.register("evaluate", eval_maxones)

# Operadores genéticos
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Algoritmo principal
def main():
    while True:
        num1 = int(input("Indique el primer número (R)GB entre 0 y 255: "))
        if (num1 >= 0 and num1 <= 255):
            break
        print("El número aceptable va de 0 a 255. Intente nuevamente.")

    while True:
        num2 = int(input("Indique el primer número R(G)B entre 0 y 255: "))
        if (num2 >= 0 and num2 <= 255):
            break
        print("El número aceptable va de 0 a 255. Intente nuevamente.")

    while True:
        num3 = int(input("Indique el primer número RG(B) entre 0 y 255: "))
        if (num3 >= 0 and num3 <= 255):
            break
        print("El número aceptable va de 0 a 255. Intente nuevamente.")

    global tripla
    tripla = (num1, num2, num3)

    random.seed(42)
    pop = toolbox.population(n=50)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

    best = tools.selBest(pop, 1)[0]
    print("Mejor individuo: ", best)
    print("Fitness: ", best.fitness.values[0])

if __name__ == "__main__":
    main()