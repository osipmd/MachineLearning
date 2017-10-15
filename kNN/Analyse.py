import math
from Drawer import *
from kNNTree import *
from kNN import *


def read_data_from_file(filename):
    data = []
    with open(filename) as input_file:
        for line in input_file:
            point = Point.from_str(line)
            data.append(point)
    return data


def transform(data, transform):
    result = []
    for point in data:
        result.append(transform(point))
    return result


# We can use different distance. Default - Euclidean distance
minkowski_distance = lambda a, b, p: (abs((b.x - a.x) ** p) + abs((b.y - a.y) ** p)) ** (1 / p)

distances = {
    # 'manhattan': lambda a, b: minkowski_distance(a, b, 1),
    'euclidian': lambda a, b: minkowski_distance(a, b, 2),
    # 'third' : lambda a, b: minkowski_distance(a, b, 3)
}

# https://en.wikipedia.org/wiki/Kernel_(statistics)
epanechnikov_kernel = lambda val: 0.75 * (1 - val * val)
gaussian_kernel = lambda val: 1 / math.sqrt(2 * math.pi) * math.exp((-1 / 2) * val * val)

multiplier = 2
transforms = {
    'multiply': lambda point: Point(point.x * multiplier, point.y * multiplier, point.class_number),
    'polar': lambda point: Point(math.sqrt(point.x ** 2 + point.y ** 2), math.atan(point.y / point.x),
                                 point.class_number)
}

data = read_data_from_file('chips.txt')

data = transform(data, transforms['multiply'])
# Drawer.draw_data(data)

number_of_classes = 2
number_cross_validation = 5
number_k_Neighbor = 10

for name, dist in distances.items():
    print(name)

    _kNNTree = kNNTree(number_of_classes, number_cross_validation, number_k_Neighbor, dist, epanechnikov_kernel, data)
    results = _kNNTree.calculate_classification()

    # kNN = k_Nearest_Neighbor(number_of_classes, number_cross_validation, number_k_Neighbor, dist, epanechnikov_kernel,
    #                          data)
    # results = kNN.calculate_classification()

    for result in results:
        train_data = result[0]
        test_data = result[1]
        for point in test_data:
            train_data.append(point)
        # Drawer.draw_data(train_data, True, 1)
        tp = Statistics.count_tp(result[2], test_data)
        fp = Statistics.count_fp(result[2], test_data)
        tn = Statistics.count_tn(result[2], test_data)
        fn = Statistics.count_fn(result[2], test_data)

        print('\t Positive Negative')
        print('C Pos : {} {}'.format(tp, fp))
        print('C neg : {} {}'.format(fn, tn))
        print("F_measure: ", Statistics.count_f_measure(result[2], test_data))
    print()  # read data from file
