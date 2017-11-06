import random

import numpy as np


class GradientDescent:
    def __init__(self):
        self.all_coeffs = []
        self.y_std = 0
        self.y_mean = 0

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

        return w.reshape(1, 3)[0]

    def calc_normalize(self, x, y, iterations = 5000):
        w = np.ones(x.shape[1]).reshape(3, 1)
        self.y_mean = y.mean()
        self.y_std = y.std()
        y_normal = (y - self.y_mean) / self.y_std
        q_prev = None
        q = None

        step = 0
        while q_prev is None or step < iterations and abs(q - q_prev) > 1:
            step += 1
            hypothesis = np.dot(x, w)
            loss = hypothesis - y_normal
            q_prev = q
            w_prev = w
            alpha = 0.1
            if step > 0:
                q = self.error(x, y_normal, w)
            if step > 1:
                alpha = self.chose_alpha(q, q_prev)
            gradient = np.dot(x.transpose(), loss) / x.shape[0] * 2
            w = w_prev - alpha * gradient

        return w

    def chose_alpha(self, q, q_prev):
        diff = np.abs(q - q_prev)

        if diff < 10:
            return 1400000
        if diff < 100:
            return 0.13
        if diff < 1000:
            return 0.01
        return 0.1

    def error(self, x, y, w):
        predict = np.dot(x, w) * self.y_std + self.y_mean
        err = np.dot((y - predict).transpose(), (y - predict))
        return err / predict.shape[0]

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
