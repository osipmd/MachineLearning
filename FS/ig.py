import math


def get_IG_for_all_features(X, Y):
    IG = []
    for i in range(len(X)):
        IG.append(get_IG(X[:, i], Y))
    return IG


def get_IG(X, Y):
    Hc = get_class_entropy(Y)
    Sum = get_features_entropy(X)
    M = get_class_to_feature_entropy(X, Y)
    H_C_E = (-1) * Sum * (-1) * M
    IG = Hc - H_C_E
    return IG


def get_class_entropy(Y):
    positive_class_count = 0
    negative_class_count = 0
    for y_element in Y:
        if y_element > 0:
            positive_class_count += 1
        if y_element < 0:
            negative_class_count += 1
    positive_probability = positive_class_count / len(Y)
    negative_probability = negative_class_count / len(Y)
    entropy = positive_probability * math.log2(positive_probability) + \
              negative_probability * math.log2(negative_probability)
    return entropy


def get_features_entropy(X):
    x_count_dict = {}
    n = len(X)
    entropy = 0
    for x_element in X:
        x_count_dict[x_element] = 0
    for x_element in X:
        x_count_dict[x_element] += 1
    for key in x_count_dict.keys():
        probability = x_count_dict[key] / n
        entropy += probability * math.log2(probability)
    return entropy


def get_matrix_class_feature(X, Y):
    dict_positive = {}
    dict_negative = {}
    for x_element in X:
        dict_positive[x_element] = 0
        dict_negative[x_element] = 0
    for i, x_element in enumerate(X):
        if (Y[i]) < 0:
            dict_negative[x_element] += 1
        else:
            dict_positive[x_element] += 1
    return dict_positive, dict_negative


def get_class_to_feature_entropy(X, Y):
    dict_positive, dict_negative = get_matrix_class_feature(X, Y)
    M = 0
    for i, y_element in enumerate(Y):
        if y_element > 0:
            probability = dict_positive[X[i]] / (dict_positive[X[i]] + dict_negative[X[i]])
            M += probability * math.log2(probability)
        if y_element < 0:
            probability = dict_negative[X[i]] / (dict_positive[X[i]] + dict_negative[X[i]])
            M += probability * math.log2(probability)
    return M
