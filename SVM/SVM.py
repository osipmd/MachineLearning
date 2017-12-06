import numpy as np
import scipy as sp
from numpy import linalg
import cvxopt
import cvxopt.solvers

import statistics
from utils import read_data_from_file


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

def core(x):
    return x


class SVM(object):
    def __init__(self, C=None):
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def get_params(self, deep=False):
        return {"C": self.C}

    def set_params(self, **params):
        self.C = params["C"]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                x_ = X[i] * X[j]
                K[i, j] = core(X[i]*X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lambd = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = lambd > 1e-5
        ind = np.arange(len(lambd))[sv]
        self.lambd = lambd[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.lambd), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.lambd)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.lambd * self.sv_y * K[ind[n], sv])
        self.b /= len(self.lambd)

        self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for lambd, sv_y, sv in zip(self.lambd, self.sv_y, self.sv):
                    s += lambd * sv_y * core(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
