from kNN import *

# We can use different distance. Default - Euclidean distance
minkowski_distance = lambda a, b, p: (abs((b.x - a.x) ** p) + abs((b.y - a.y) ** p)) ** (1 / p)

distances = {
    # 'manhattan': lambda a, b: minkowski_distance(a, b, 1),
    'euclidian': lambda a, b: minkowski_distance(a, b, 2),
    # 'third' : lambda a, b: minkowski_distance(a, b, 3)
}


# https://en.wikipedia.org/wiki/Kernel_(statistics)
epanechnikov_kernel = lambda val : 0.75 * (1 - val * val)


number_of_classes = 2
number_cross_validation = 5
number_k_Neighbor = 10

for name, dist in distances.items():
    print(name)
    kNN = k_Nearest_Neighbor(number_of_classes, number_cross_validation, number_k_Neighbor, dist, epanechnikov_kernel)
    kNN.calculate_classification()
    print()