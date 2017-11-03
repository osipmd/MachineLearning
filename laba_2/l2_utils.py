from flat import *
from sklearn.preprocessing import StandardScaler
import numpy as np


def read_data(path_to_file='src_data/prices_without_head.txt'):
    flats = []
    with open(path_to_file) as f:
        for line in f:
            area, rooms, price = map(int, line.split(','))
            flat = Flat(area, rooms, price)
            flats.append(flat)
    return flats


def normalize_data(path_to_file='src_data/prices_without_head.txt'):
    flats = []
    with open(path_to_file) as f:
        for line in f:
            area, rooms, price = map(int, line.split(','))
            flat = [area, rooms, price]
            flats.append(flat)
    scaler = StandardScaler()
    scaler.fit(flats)
    return scaler.transform(flats)


def read_normalized_data(normalized_flats=normalize_data()):
    flats = []
    for normalized_flat in normalized_flats:
        area = normalized_flat[0]
        rooms = normalized_flat[1]
        price = normalized_flat[2]
        flat = Flat(area, rooms, price)
        flats.append(flat)
    return flats


def rms_error(real, predicted):
    def sqr_fun(elems): return (elems[0] - elems[1]) ** 2

    l = len(real)
    return sum(map(sqr_fun, zip(real, predicted))) / l


def create_model(coeffs):
    return lambda flat: flat.area * coeffs[0] + flat.rooms * coeffs[1] + 1 * coeffs[2]
