# We can use different distance. Default - Euclidean distance
minkowski_distance = lambda a, b, p: (abs((b.x - a.x) ** p) + abs((b.y - a.y) ** p)) ** (1 / p)

distances = {
    'manhattan': lambda a, b: minkowski_distance(a, b, 1),
    'euclidian': lambda a, b: minkowski_distance(a, b, 2),
    'third': lambda a, b: minkowski_distance(a, b, 3)
}

minkowski_distance_3d = lambda a, b, p: (abs((b.x - a.x) ** p) + abs((b.y - a.y) ** p) + abs((b.z - a.z) ** p)) ** (
1 / p)

distances_3d = {
    'manhattan': lambda a, b: minkowski_distance_3d(a, b, 1),
    'euclidian': lambda a, b: minkowski_distance_3d(a, b, 2),
    'third': lambda a, b: minkowski_distance_3d(a, b, 3)
}
