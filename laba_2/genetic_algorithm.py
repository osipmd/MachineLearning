import random

import math

from l2_utils import read_data, create_model, rms_error


def individual(length, min, max):
    return [random.uniform(min, max) for i in range(length)]


def population(count, length, min, max):
    return [individual(length, min, max) for i in range(count)]

"""
def fitness(individ, target):
    sum = 0
    for i in range(len(individ)):
        sum += individ[i]
    return abs(target - sum)
"""

def fitness(individ, target):
    model = create_model(individ)
    y = list(map(lambda flat: flat.price, flats))
    predicted_y = list(map(lambda flat: model(flat), flats))
    error = rms_error(y, predicted_y)
    return abs(target - error)

def grade(pop, target):
    sum = 0
    for individ in pop:
        sum += fitness(individ, target)
    return sum / (len(pop) * 1.0)


def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    # get fitness for current population
    graded = [(fitness(x, target), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    # define retain size
    retain_length = int(len(graded) * retain)
    # get the best individuals for crossover
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic diversity
    # it's need for avoiding local minimum
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)

    # mutate some individuals
    # it's need for avoiding local minimum
    for individual in parents:
        if mutate > random.random():
            position_to_mutate = random.randint(0, len(individual) - 1)
            individual[position_to_mutate] = random.uniform(
                min(individual), max(individual))

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        father = random.randint(0, parents_length - 1)
        mather = random.randint(0, parents_length - 1)
        if father != mather:
            father = parents[father]
            mather = parents[mather]
            half = int(len(father) / 2)
            child = father[:half] + mather[half:]
            children.append(child)

    parents.extend(children)
    return parents

flats = read_data()

target = 0
p_count = 100
i_length = 3
i_min = -1000
i_max = 1000
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target), ]
best_fitness = 100000000000000000
best_population = []
for i in range(1000):
    p = evolve(p, target)
    grade_result = grade(p, target)
    fitness_history.append(grade_result)
    if grade_result < best_fitness:
        best_fitness = grade_result
        best_population = p

#for datum in fitness_history:
#   print(datum)

print("Best fitness : ", best_fitness)
print("Best population : ", best_population)
graded = [(fitness(x, target), x) for x in best_population]
graded_result = [x[0] for x in sorted(graded)][0]
graded_vector = [x[1] for x in sorted(graded)][0]
print("The best result : ", graded_result)
print("The best vector : ", graded_vector)
print("Sqrt of the best result : ", math.sqrt(graded_result))
