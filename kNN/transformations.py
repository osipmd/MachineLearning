import math
from kNN import *
from Point3D import *


def data_to_polar(data):
    r = lambda p: math.sqrt(p.x ** 2 + p.y ** 2)
    angle = lambda p: math.atan(p.y / p.x)
    return list(map(lambda p: Point2D(r(p), angle(p), p.class_number), data))


def mult_data(data, multiplier):
    return list(map(lambda p: Point2D(p.x * multiplier, p.y * multiplier, p.class_number), data))


def data_to_elliptic(data, a=1, b=1):
    z = lambda x, y: (x / a) ** 2 + (y / b) ** 2
    return list(map(lambda p: Point3D(p.x, p.y, z(p.x, p.y), p.class_number), data))


def data_to_hyperbolic(data, a=1, b=1):
    z = lambda x, y: (x / a) ** 2 - (y / b) ** 2
    return list(map(lambda p: Point3D(p.x, p.y, z(p.x, p.y), p.class_number), data))
