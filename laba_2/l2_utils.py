from point import *


def read_data(path_to_file='src_data/prices_without_head.txt'):
    points = []
    with open(path_to_file) as f:
        for line in f:
            area, rooms, price = map(int, line.split(','))
            point = Point(area, rooms, price)
            points.append(point)
    return points


def rms_error(real, predicted):
    def sqr_fun(elems): return (elems[0] - elems[1]) ** 2

    l = len(real)
    return sum(map(sqr_fun, zip(real, predicted))) / l

def create_model(coeffs):
    return lambda point: point.area * coeffs[0] + point.rooms * coeffs[1] + 1 * coeffs[2]
