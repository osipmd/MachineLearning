import numpy as np


class GradientDescent:
    @staticmethod
    def calc(data, eps=0.001, k=0.0000001, by_hand=False):
        w = np.array([0, 0, 0]).reshape(3, 1)
        x = np.array(list(map(lambda point: point.get_attrs_list(), data)))
        y = np.array(list(map(lambda point: [point.price], data)))
        w_prev = None

        step = 0
        while w_prev is None or np.linalg.norm(w_prev - w) >= eps:
            step += 1
            w_prev = w
            eta = k / step
            if by_hand:
                gradient = GradientDescent.calc_gradient_by_hand(data, w_prev)
            else:
                gradient = GradientDescent.calc_gradient(x, w_prev, y)
            w = w_prev - eta * gradient

        return w.reshape(1, 3)[0]

    @staticmethod
    def calc_gradient(x, w, y):
        l = len(y)
        return 2 / l * x.transpose().dot(x.dot(w) - y)

    @staticmethod
    def calc_gradient_by_hand(data, w):
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

