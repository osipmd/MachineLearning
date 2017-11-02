import math

from flat import *


def read_data(path_to_file='src_data/prices_without_head.txt'):
    flats = []
    with open(path_to_file) as f:
        for line in f:
            area, rooms, price = map(int, line.split(','))
            flat = Flat(area, rooms, price)
            flats.append(flat)
    return flats


def rms_error(real, predicted):
    def sqr_fun(elems): return (elems[0] - elems[1]) ** 2

    l = len(real)
    return sum(map(sqr_fun, zip(real, predicted))) / l


def create_model(coeffs):
    return lambda flat: flat.area * coeffs[0] + flat.rooms * coeffs[1] + 1 * coeffs[2]


def step_error(all_coeffs, flats):
    errors = []
    for coeffs in all_coeffs:
        model = create_model(coeffs)

        y = list(map(lambda flat: flat.price, flats))
        predicted_y = list(map(lambda flat: model(flat), flats))

        error = rms_error(y, predicted_y)
        if error < 0:
            error = 0
        errors.append(error)
        # errors.append(math.sqrt(error))
    return errors


def normalize(flats):
    max_area = max(list(map(lambda flat: flat.area, flats)))
    max_rooms = max(list(map(lambda flat: flat.rooms, flats)))
    max_price = max(list(map(lambda flat: flat.price, flats)))
    for flat in flats:
        flat.area /= max_area
        flat.rooms /= max_rooms
        flat.price /= max_price
    return flats, max_area, max_rooms, max_price
