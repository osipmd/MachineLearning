import math
from Drawer import *

from kNN import *

def read_data_from_file(filename):
    data = []
    with open(filename) as input_file:
        for line in input_file:
            point = Point.from_str(line)
            data.append(point)
    return data


# We can use different distance. Default - Euclidean distance
minkowski_distance = lambda a, b, p: (abs((b.x - a.x) ** p) + abs((b.y - a.y) ** p)) ** (1 / p)

distances = {
    # 'manhattan': lambda a, b: minkowski_distance(a, b, 1),
    'euclidian': lambda a, b: minkowski_distance(a, b, 2),
    # 'third' : lambda a, b: minkowski_distance(a, b, 3)
}

# https://en.wikipedia.org/wiki/Kernel_(statistics)
epanechnikov_kernel = lambda val: 0.75 * (1 - val * val)
gaussian_kernel = lambda val: 1/math.sqrt(2*math.pi)*math.exp((-1/2)*val*val)

number_of_classes = 2
number_cross_validation = 5
number_k_Neighbor = 10
data = read_data_from_file('chips.txt')
Drawer.draw_data(data)
for name, dist in distances.items():
    print(name)
    kNN = k_Nearest_Neighbor(number_of_classes, number_cross_validation, number_k_Neighbor, dist, epanechnikov_kernel,
                             data)
    kNN.calculate_classification()
    print()  # read data from file
