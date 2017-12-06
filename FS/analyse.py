from utils import read_features_from_file
from utils import read_labels_from_file

features_train_file = 'Practice_FS/arcene_train.data'
feature_test_file = 'Practice_FS/arcene_train.labels'
labels_train_file = 'Practice_FS/arcene_valid.data'
labels_test_file = 'FPractice_FS/arcene_valid.labels'

features_train = read_features_from_file(features_train_file)
features_test = read_features_from_file(feature_test_file)
labels_train = read_labels_from_file(labels_train_file)
labels_test = read_labels_from_file(labels_test_file)
