from pearson import count_all_pearson_correlation_coefficient
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
print(pearson_coefficients)
print(np.sort(pearson_coefficients))