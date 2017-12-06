import numpy as np


def read_labels_from_file(filename):
    labels = []
    with open(filename) as file:
        for line in file:
            labels.append(int(line))
    return np.asanyarray(labels)



def read_features_from_file(filename):
    features = []
    with open(filename) as file:
        for line in file:
            row = [int(i) for i in line.split()]
            features.append(row)
    return np.asanyarray(features)

