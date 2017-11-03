import random

import numpy as np


class GradientDescent:
    def __init__(self):
        self.all_coeffs = []

    def calc(self, data, eps=0.001, k=0.0000001, by_hand=False):
        w1 = random.uniform(-1 / 6, 1 / 6)
        w2 = random.uniform(-1 / 6, 1 / 6)
        w0 = random.uniform(-1 / 6, 1 / 6)

        w = np.array([w1, w2, w0]).reshape(3, 1)
        x = np.array(list(map(lambda point: point.get_attrs_list(), data)))
        y = np.array(list(map(lambda point: [point.price], data)))
        w_prev = None

        MAX_STEP = 1500
        step = 0
        while w_prev is None or np.linalg.norm(w_prev - w) >= eps and step < MAX_STEP:
            self.all_coeffs.append(w.reshape(1, 3)[0])
            step += 1
            w_prev = w
            eta = k / step
            if by_hand:
                gradient = self.calc_gradient_by_hand(data, w_prev)
            else:
                gradient = self.calc_gradient(x, w_prev, y)
            w = w_prev - eta * gradient

        # self.all_coeffs.append(w.reshape(1, 3)[0])
        return w.reshape(1, 3)[0]

    def calc_gradient(self, x, w, y):
        l = len(y)
        # метод наименьших квадратов
        return 2 / l * x.transpose().dot(x.dot(w) - y)

    def calc_gradient_by_hand(self, data, w):
        l = len(data)
        dQ_dw0 = 2 / l * sum(
            map(lambda flat: flat.rooms * w.item(1) + flat.area * w.item(0) + w.item(2) - flat.price, data))
        dQ_dw1 = 2 / l * sum(
            map(lambda flat: (flat.rooms * w.item(1) + flat.area * w.item(0) + w.item(2) - flat.price) * flat.area,
                data))
        dQ_dw2 = 2 / l * sum(
            map(lambda flat: (flat.rooms * w.item(1) + flat.area * w.item(0) + w.item(2) - flat.price) * flat.rooms,
                data))
        return np.array([dQ_dw1, dQ_dw2, dQ_dw0]).reshape(3, 1)
