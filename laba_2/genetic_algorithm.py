import random
import math

import numpy as np

from l2_utils import read_data, create_model, rms_error, read_normalized_data, read_data_to_two_set, get_error

best_fitness = 100000000000000000
best_individ = []


def individual(length, min, max):
    area = random.uniform(1, 2)
    rooms = random.uniform(-80, -70)
    free = random.uniform(60, 80)
    return [area, rooms, free]


def population(count, length, min, max):
    individ_arr = [individual(length, min, max) for i in range(count)]
    return individ_arr


"""
def fitness(individ, target):
    sum = 0
    for i in range(len(individ)):
        sum += individ[i]
    return abs(target - sum)
"""


def fitness(individ, target, x, y):
    individ = np.array(individ).reshape(3, 1)
    error = get_error(x, y, individ)
    return abs(target - error)


def grade(pop, target, x, y):
    global best_fitness
    global best_individ
    sum = 0
    for individ in pop:
        fit = fitness(individ, target, x, y)
        if best_fitness > fit:
            best_fitness = fit
            best_individ = individ
        sum += fit
    return sum / (len(pop) * 1.0)


def evolve(pop, target, x, y, retain=0.2, random_select=0.05, mutate=0.01):
    # get fitness for current population
    graded = [(fitness(individ, target, x, y), individ) for individ in pop]
    graded = [individ[1] for individ in sorted(graded)]
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
x, y = read_data_to_two_set()
x_n = (x - x.mean()) / x.std()
y_n = np.copy(y)
y_n = (y_n - y.mean()) / y.std()

target = 0
population_count = 100
vector_length = 3
min_coeff = -100
max_coeff = 100
p = population(population_count, vector_length, min_coeff, max_coeff)
fitness_history = [grade(p, target, x_n, y), ]
best_population = []

for i in range(3000):
    p = evolve(p, target, x_n, y)
    grade_result = grade(p, target, x_n, y)
    fitness_history.append(grade_result)
    if grade_result < best_fitness:
        best_fitness = grade_result
        best_population = p

print("The best result: ", math.sqrt(best_fitness))
