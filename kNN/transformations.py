import math
from kNN import *


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


def data_to_elliptic(data, a=1, b=2):
    new_data = []
