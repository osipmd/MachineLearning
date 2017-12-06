import math


def count_all_pearson_correlation_coefficient(X, Y):
    pearson_coefficients = []
    for i in range(len(X)):
        pearson_coefficients.append(count_pearson_correlation_coefficient(X[:, i], Y))
    return pearson_coefficients


def count_pearson_correlation_coefficient(X, Y):
    x_mean = X.mean()
    y_mean = Y.mean()
    x = 0
    x_2 = 0
    y = 0
    y_2 = 0
    for x_element in X:
        x += (x_element - x_mean)
        x_2 += (x_element - x_mean) ** 2
    for y_element in Y:
        y += (y_element - y_mean)
        y_2 += (y_element - y_mean) ** 2
    pearson_coefficient = x * y / math.sqrt(x_2 * y_2)
    return pearson_coefficient