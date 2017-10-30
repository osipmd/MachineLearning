from point import *


def read_data(path_to_file='src_data/prices_without_head.txt'):
    points = []
    with open(path_to_file) as f:
        for line in f:
            area, rooms, price = map(int, line.split(','))
            point = Point(area, rooms, price)
            points.append(point)
    return points
