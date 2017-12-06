from scipy.stats import entropy

from ig import get_IG_for_all_features
from pearson import count_all_pearson_correlation_coefficient
from spearman import define_rank, count_all_spearman_correlation_coefficient
from utils import read_features_from_file
from utils import read_labels_from_file

import numpy as np

features_train_file = 'Practice_FS/arcene_train.data'
feature_test_file = 'Practice_FS/arcene_valid.data'
labels_train_file = 'Practice_FS/arcene_train.labels'
labels_test_file = 'Practice_FS/arcene_valid.labels'

features_train = read_features_from_file(features_train_file)
features_test = read_features_from_file(feature_test_file)
labels_train = read_labels_from_file(labels_train_file)
labels_test = read_labels_from_file(labels_test_file)


pearson_coefficients = count_all_pearson_correlation_coefficient(features_train, labels_train)
#print(sorted(np.abs(pearson_coefficients), reverse=True))
pearson_dict = {}
for i, x in enumerate(pearson_coefficients):
    pearson_dict[i] = np.abs(x)

sorted_keys = sorted(pearson_dict, key=lambda x: pearson_dict[x], reverse=True)
print(sorted_keys)


spearman_coefficients = count_all_spearman_correlation_coefficient(features_train, labels_train)
#print(sorted(np.abs(spearman_coefficients), reverse=True))
spearman_dict = {}
for i, x in enumerate(spearman_coefficients):
    spearman_dict[i] = np.abs(x)

sorted_keys = sorted(spearman_dict, key=lambda x: spearman_dict[x], reverse=True)
print(sorted_keys)


IG = get_IG_for_all_features(features_train, labels_train)
#print(IG)
IG_dict = {}
for i, x in enumerate(IG):
    IG_dict[i] = x

sorted_keys = sorted(IG_dict, key=lambda x: IG_dict[x], reverse=True)
print(sorted_keys)
