def count_all_spearman_correlation_coefficient(X, Y):
    spearman_coefficients = []
    rank_dict_labels, ligaments_labels = get_rank_ligaments(Y)
    for i in range(len(X)):
        spearman_coefficients.append(count_spearman_correlation_coefficient_2(X[i], Y, rank_dict_labels, ligaments_labels))
    return spearman_coefficients


def count_spearman_correlation_coefficient_2(X, Y, rank_dict_labels, ligaments_labels):
    rank_dict_features, ligaments_features = get_rank_ligaments(X)
    n = len(X)
    diff_rank = 0
    for i, x in enumerate(X):
        diff_rank = (rank_dict_features[x] - (n + 1) / 2) * (rank_dict_labels[Y[i]] - (n + 1) / 2)
    spearman_coefficient = diff_rank / (n * (n - 1) * (n + 1) - (ligaments_features + ligaments_labels))
    return spearman_coefficient


def count_spearman_correlation_coefficient(X, Y, rank_dict_labels, ligaments_labels):
    rank_dict_features = define_rank(X)
    n = len(X)
    diff_rank = 0
    for i, x in enumerate(X):
        diff_rank = (rank_dict_features[x] - rank_dict_labels[Y[i]]) ** 2
    spearman_coefficient = 1 - (6 / (n * (n - 1) * (n + 1))) * diff_rank
    return spearman_coefficient


def define_rank(Y):
    rank_dict = {}
    count_dict = {}
    y = sorted(Y)
    for i, y_element in enumerate(y):
        rank_dict[y_element] = 0
        count_dict[y_element] = 0
    for i, y_element in enumerate(y):
        rank_dict[y_element] = i
        count_dict[y_element] += 1
    for key in count_dict.keys():
        if count_dict[key] != 1:
            rank_dict[key] /= count_dict[key]
    return rank_dict, count_dict


def get_rank_ligaments(Y):
    rank_dict, count_dict = define_rank(Y)
    ligaments = 0
    for key in count_dict.keys():
        if count_dict[key] != 1:
            ligaments += count_dict[key] * ((count_dict[key] ** 2) - 1)
    ligaments *= 1 / 2
    return rank_dict, ligaments
