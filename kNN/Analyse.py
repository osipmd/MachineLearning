import math
import random
from kNN import *
from Point import *
from Classification import *
from Drawer import *


def print_classification(classification):
    for unit in classification.units:
        print(unit)
        for point in unit.test:
            print(point, end=" , ")
        print()
        for point in unit.classified:
            print(point, end=" , ")
        print()
        print(unit.calculate_confusion_matrix())
        matrix = unit.calculate_confusion_matrix()
        print(Statistics.count_recall(matrix))
        print(Statistics.count_precision(matrix))
        print(Statistics.count_f_measure(matrix))
    print('f_measure ', classification.calc_f_measure())


# read data from file
def read_data_from_file(filename):
    data = []
    with open(filename) as input_file:
        for line in input_file:
            point = Point.from_str(line)
            data.append(point)
    return data

def data_to_polar(data):
    new_data = []
    for point in data:
        r = math.sqrt(point.x ** 2 + point.y ** 2)
        tg = math.atan(point.y / point.x)
        new_point = Point(r, tg, point.class_number)
        new_data.append(new_point)
    return new_data

def mult_data(data, multiplier):
    new_data = []
    for point in data:
        new_x = point.x * multiplier
        new_y = point.y * multiplier
        new_point = Point(new_x, new_y, point.class_number)
        new_data.append(new_point)
    return new_data

# We can use different distance. Default - Euclidean distance
minkowski_distance = lambda a, b, p: (abs((b.x - a.x) ** p) + abs((b.y - a.y) ** p)) ** (1 / p)

distances = {
    'manhattan': lambda a, b: minkowski_distance(a, b, 1),
    'euclidian': lambda a, b: minkowski_distance(a, b, 2),
    'third': lambda a, b: minkowski_distance(a, b, 3)
}

# https://en.wikipedia.org/wiki/Kernel_(statistics)
kernels = {
    'epanechnikov': lambda val: 0.75 * (1 - val * val),
    'gaussian': lambda val: 1 / math.sqrt(2 * math.pi) * math.exp((-1 / 2) * val * val),
    # 'uniform': lambda val: 1 / 2,
    # 'triangular': lambda val: 1 - abs(val),
    # 'quartic': lambda val: 15 / 16 * (1 - val ** 2) ** 2
}

number_of_classes = 2

data = read_data_from_file('chips.txt')

# data = data_to_polar(data)
data = mult_data(data, 2)
random.shuffle(data)

best_number_k = 0
best_f_measure = 0

best_dist = distances.get('euclidian')
best_dist_name = ''

best_kernel = kernels.get('epanechnikov')
best_kernel_name = ''

best_cross_validation_number = 0

build_graphics = True
f_measure_of_results = []

for kernel_name, kernel in kernels.items():
    for dist_name, dist in distances.items():
        for cross_validation_number in range(5, 10):
            file_name = "out_data/f-measure/k_results_" + str(cross_validation_number) + ".png"
            f_measure_of_results = []
            for number_k_Neighbor in range(5, 10):
                kNN = k_Nearest_Neighbor(number_of_classes, cross_validation_number, number_k_Neighbor, dist,
                                         kernel,
                                         data)
                classification = kNN.calculate_classification()
                cur_f_measure = classification.calc_f_measure()
                f_measure_of_results.append(cur_f_measure)

                if cur_f_measure > best_f_measure:
                    best_f_measure = cur_f_measure
                    best_number_k = number_k_Neighbor
                    best_dist = dist
                    best_dist_name = dist_name
                    best_kernel = kernel
                    best_kernel_name = kernel_name
                    best_cross_validation_number = cross_validation_number

            if build_graphics:
                Drawer.draw_graphic(file_name, f_measure_of_results, 5)

kNN = k_Nearest_Neighbor(number_of_classes, best_cross_validation_number, best_number_k, best_dist,
                         best_kernel,
                         data)
classification = kNN.calculate_classification()

if build_graphics:
    Drawer.define_class_of_space(classification)

print('best k', best_number_k)
print('best dist ', best_dist_name)
print('best kernel ', best_kernel_name)
print('best cross validation number ', best_cross_validation_number)
print('f_measure ', classification.calc_f_measure())
