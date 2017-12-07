import numpy as np
import scipy as sp
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl

import f_measure_statistics
from analyse import do_analyse
from statistics import create_statistics
from utils import read_features_from_file
from utils import read_labels_from_file


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SVM(object):
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
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
                K[i, j] = self.kernel(X[i], X[j])

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

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.lambd)):
                self.w += self.lambd[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for lambd, sv_y, sv in zip(self.lambd, self.sv_y, self.sv):
                    s += lambd * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


def split_data(X, Y, test_number):
    X_train = np.vstack((X[:test_number], X[(test_number + 1):]))
    Y_train = np.hstack((Y[:test_number], Y[(test_number + 1):]))
    X_test = X[test_number:(test_number + 1)]
    Y_test = Y[test_number:(test_number + 1)]
    return X_train, Y_train, X_test, Y_test

def union_features(X, features_index):
    union = X[:, features_index[0]]
    for i in range(len(features_index) - 1):
        union = np.vstack((union, X[:, features_index[i]]))
    return union.transpose()

def read_arrays():
    with open("results.txt", 'r') as input:
        for line in input:
            line = line.rstrip()[line.index(':')].strip()[1:]
            print(line)

def test_soft():
    features_train_file = 'Practice_FS/arcene_train.data'
    feature_test_file = 'Practice_FS/arcene_valid.data'
    labels_train_file = 'Practice_FS/arcene_train.labels'
    labels_test_file = 'Practice_FS/arcene_valid.labels'

    features_train = read_features_from_file(features_train_file)
    features_test = read_features_from_file(feature_test_file)
    labels_train = read_labels_from_file(labels_train_file)
    labels_test = read_labels_from_file(labels_test_file)

    pearson_sorted_keys, spearman_sorted_keys, IG_sorted_keys = do_analyse()
    All, pearson_spearman, pearson_IG, spearman_IG, pearson, spearman, IG = create_statistics()
    pearson_sorted_keys = pearson_sorted_keys[:100]
    spearman_sorted_keys = spearman_sorted_keys[:100]
    IG_sorted_keys = IG_sorted_keys[:50]

    X_train = union_features(features_train, spearman_IG)
    X_test = np.vstack((features_test[:, pearson_sorted_keys[0]], features_test[:, pearson_sorted_keys[1]])).transpose()
    Y_train = labels_train
    Y_test = labels_test

    conflusion_matrix = sp.array([[0, 0], [0, 0]])
    predicted_array = []
    X_array = []
    for i in range(len(X_train)):
        X_train, Y_train, X_test, Y_test = split_data(X_train, Y_train, i)
        clf = SVM(C=1.0)
        clf.fit(X_train, Y_train)

        y_predict = clf.predict(X_test)

        predicted_array.append(y_predict)
        X_array.append(X_test)
        if y_predict == Y_test == 1:
            conflusion_matrix[0][0] += 1
        elif y_predict == Y_test == -1:
            conflusion_matrix[1][1] += 1
        elif y_predict == 1:
            conflusion_matrix[0][1] += 1
        else:
            conflusion_matrix[1][0] += 1

    f_measure = f_measure_statistics.Statistics.count_f_measure(conflusion_matrix)

    print("F-measure : ", f_measure)

    return predicted_array, X_array

predicted_array, X_array = test_soft()
