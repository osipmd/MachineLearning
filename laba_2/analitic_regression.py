import numpy as np


class AnalyticRegression:
    # w = ( ( X_t * X ) ^ (-1) ) * X_t * y
    @staticmethod
    def calc(data):
        x = np.matrix(list(map(lambda point: point.get_attrs_list(), data)))
        y = np.matrix(list(map(lambda point: [point.price], data)))
        x_t = x.transpose()
        return np.array(np.linalg.inv(x_t.dot(x)).dot(x_t).dot(y).reshape(1, 3))[0]
