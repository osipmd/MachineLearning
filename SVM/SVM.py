import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split, cross_val_score, KFold


from utils import read_data_from_file


class SVMClassifier(BaseEstimator):
    w = None
    w0 = None

    def __init__(self, C=1.0):
        self.C = C

    def get_params(self, deep=False):
        return {"C": self.C}

    def set_params(self, **params):
        self.C = params["C"]

    def fit(self, X, y):
        # data['R'] = (X[0]**2 + X[1]**2) ** 0.5
        # data['Φ'] = np.arctan2(X[0], X[1])
        # X0 = X
        # X = data[['R', 'Φ']].values
        # X = data[['X', 'Y']].values
        N = len(X)
        y_ = np.apply_along_axis(lambda t: 2 * t - 1, 0, y)

        construct_H = lambda i, j: (np.dot(X[i], X[j])) * y_[i] * y_[j]
        H = np.fromfunction(np.vectorize(construct_H), (N, N), dtype=int)
        c = -np.ones(N)
        x0 = np.random.randn(N)
        cons = [{"type": "ineq", "fun": lambda x: self.C * np.ones(N) - x, "jac": lambda x: -np.eye(N)}
            , {"type": "ineq", "fun": lambda x: x, "jac": lambda x: np.eye(N)}
            , {"type": "eq", "fun": lambda x: np.dot(y_, x), "jac": lambda x: y_}]
        opt = {"disp": False}

        # solve constrained minimization problem using quadratic programming solver
        loss = lambda x: 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(c, x)
        jac = lambda x: 0.5 * np.dot(x.T, H) + c
        res = sp.optimize.minimize(loss, x0, jac=jac, constraints=cons, method="SLSQP", options=opt)
        # find w, w0
        self.w = np.dot(res.x * y_, X)
        for i, w_i in enumerate(res.x):
            if (w_i > 0):
                self.w0 = np.dot(self.w, X[i]) - y_[i]
                break

    def predict(self, X):
        y_pred_ = np.sign(np.dot(X, self.w) - self.w0)
        save(X, y_pred_)

        return 0.5 * y_pred_ + 0.5

def core(x):
    return x
    # return (abs(x) + 1) ** 2 * np.sign(x)


x2_p = []
x2_n = []
x2 = []
y2 = []


def save(x, y):
    for i in range(len(x)):
        x1 = x[i]
        y1 = y[i]
        if y1 > 0:
            y2.append(1)
            x2_p.append(x1)
        else:
            y2.append(-1)
            x2_n.append(x1)


X, Y = read_data_from_file('chips.txt')
C = 1.0
#cv_score_2 = cross_val_score(SVMClassifier(C), X, Y, scoring="accuracy", cv=118)

clf = SVMClassifier(C)
clf.fit(X, Y)
Y_predict = clf.predict(X)
print(confusion_matrix(Y, Y_predict))
recall = confusion_matrix(Y, Y_predict)[0][0] / (confusion_matrix(Y, Y_predict)[0][0] + confusion_matrix(Y, Y_predict)[0][1])
precision = confusion_matrix(Y, Y_predict)[0][0] / (confusion_matrix(Y, Y_predict)[0][0] + confusion_matrix(Y, Y_predict)[1][0])
print(2*recall*precision/(recall+precision))
